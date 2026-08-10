"""
benchmark.py
────────────
Evaluate SigLIP2 attribute detection against ground-truth labels from
pedestrian_1k/metadata.jsonl.

Usage
─────
    # Full benchmark (all 1k images)
    python benchmark.py --image_dir pedestrian_1k --metadata pedestrian_1k/metadata.jsonl

    # Quick smoke-test on 50 images
    python benchmark.py --image_dir pedestrian_1k --metadata pedestrian_1k/metadata.jsonl --limit 50

    # Save detailed results to JSON
    python benchmark.py ... --out_json results/benchmark_out.json

Output
──────
  • Per-group accuracy (top-1 for multi-class, binary F1 for accessories)
  • Per-class confusion within each group
  • Weighted mean accuracy across all groups
  • Throughput  (images / second)
  • Optional JSON export of every per-image prediction + GT
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from attribute_detector import AttributeDetector
from vlm_detector import VLMAttributeDetector
from attribute_schemas import PERSON_ATTRIBUTE_GROUPS, AttributeGroup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth → schema label mapping
# ──────────────────────────────────────────────────────────────────────────────

# Maps dataset attribute key → {gt_value: label_string_in_schema}
# The label_string_in_schema must appear verbatim in the AttributeGroup.labels list.

GT_LABEL_MAP: Dict[str, Dict[str, str]] = {
    "gender": {
        "man":   "a photo of a man",
        "woman": "a photo of a woman",
    },
    "age": {
        "teenager":    "a photo of a teenager",
        "young adult": "a photo of a young adult",
        "middle-aged": "a photo of a middle-aged person",
        "elderly":     "a photo of an elderly person",
    },
    "angle": {
        "seen from the front": "a photo of a person seen from the front",
        "seen from the back":  "a photo of a person seen from the back",
        "seen from the side":  "a photo of a person seen from the side",
    },
    "hair": {
        "black":  "a photo of a person with black hair",
        "brown":  "a photo of a person with brown hair",
        "blonde": "a photo of a person with blonde hair",
        "grey":   "a photo of a person with grey or white hair",
        "shaved": "a photo of a person with shaved or very short hair",
    },
    "top_type": {
        "t-shirt":       "a photo of a person wearing a t-shirt",
        "tank top":      "a photo of a person wearing a tank top",
        "hoodie":        "a photo of a person wearing a hoodie",
        "knit sweater":  "a photo of a person wearing a knit sweater",
        "leather jacket":"a photo of a person wearing a leather jacket",
        "puffer jacket": "a photo of a person wearing a puffer jacket",
    },
    "top_color": {
        "white":        "a photo of a person wearing a white top",
        "black":        "a photo of a person wearing a black top",
        "grey":         "a photo of a person wearing a grey top",
        "navy blue":    "a photo of a person wearing a navy blue top",
        "bright red":   "a photo of a person wearing a bright red top",
        "forest green": "a photo of a person wearing a forest green top",
        "yellow":       "a photo of a person wearing a yellow top",
    },
    "bottom_type": {
        "denim jeans":    "a photo of a person wearing denim jeans",
        "formal trousers":"a photo of a person wearing formal trousers",
        "cargo pants":    "a photo of a person wearing cargo pants",
        "sweatpants":     "a photo of a person wearing sweatpants",
        "shorts":         "a photo of a person wearing shorts",
        "leggings":       "a photo of a person wearing leggings",
    },
    "bottom_color": {
        "white":     "a photo of a person wearing white trousers or bottoms",
        "black":     "a photo of a person wearing black trousers or bottoms",
        "blue":      "a photo of a person wearing blue or dark blue trousers or bottoms",
        "dark grey": "a photo of a person wearing dark grey trousers or bottoms",
        "khaki":     "a photo of a person wearing khaki or beige trousers or bottoms",
    },
}

# Maps accessory GT values → which binary group names should be True
ACCESSORY_TO_GROUPS: Dict[str, List[str]] = {
    "a black backpack":      ["carrying_backpack"],
    "a brown leather handbag": ["carrying_handbag"],
    "a colorful umbrella":   ["holding_umbrella"],
    "holding a smartphone":  ["holding_smartphone"],
    "wearing a baseball cap":["wearing_cap"],
    "wearing sunglasses":    ["wearing_sunglasses"],
    "none":                  [],
}

# All binary group names tracked in this benchmark
BINARY_GROUPS = [
    "carrying_backpack", "carrying_handbag", "holding_umbrella",
    "holding_smartphone", "wearing_cap", "wearing_sunglasses",
]

# Multi-class groups tracked (must match keys in GT_LABEL_MAP)
MULTICLASS_GROUPS = [
    "gender", "age", "angle", "hair",
    "top_type", "top_color", "bottom_type", "bottom_color",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _group_by_name(name: str) -> Optional[AttributeGroup]:
    for g in PERSON_ATTRIBUTE_GROUPS:
        if g.name == name:
            return g
    return None


def _gt_label_idx(group: AttributeGroup, gt_label_str: str) -> Optional[int]:
    """Return the index of gt_label_str inside group.labels, or None."""
    try:
        return group.labels.index(gt_label_str)
    except ValueError:
        return None


def _gt_for_binary_groups(accessory_value: str) -> Dict[str, bool]:
    """Return {group_name: bool} for all binary groups given the accessory GT."""
    true_groups = set(ACCESSORY_TO_GROUPS.get(accessory_value, []))
    return {g: (g in true_groups) for g in BINARY_GROUPS}


# ──────────────────────────────────────────────────────────────────────────────
# Metrics accumulators
# ──────────────────────────────────────────────────────────────────────────────

class MulticlassMetrics:
    """Top-1 accuracy + confusion matrix for a multi-class group."""

    def __init__(self, name: str, labels: List[str]) -> None:
        self.name = name
        self.labels = labels
        n = len(labels)
        self.confusion = [[0] * n for _ in range(n)]   # [gt][pred]
        self.total = 0
        self.correct = 0

    def update(self, gt_idx: int, pred_idx: int) -> None:
        self.confusion[gt_idx][pred_idx] += 1
        self.total += 1
        if gt_idx == pred_idx:
            self.correct += 1

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def short_label(self, idx: int) -> str:
        """Trim prompt boilerplate for display."""
        lbl = self.labels[idx]
        for pfx in ["a photo of a person wearing ", "a photo of a person with ",
                     "a photo of a person seen from ", "a photo of a person ",
                     "a photo of a "]:
            if lbl.startswith(pfx):
                return lbl[len(pfx):]
        return lbl


class BinaryMetrics:
    """Precision / Recall / F1 for a binary (sigmoid) group."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.tp = self.fp = self.tn = self.fn = 0

    def update(self, gt: bool, pred: bool) -> None:
        if gt and pred:
            self.tp += 1
        elif not gt and pred:
            self.fp += 1
        elif gt and not pred:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    detector: AttributeDetector,
    image_dir: Path,
    metadata_path: Path,
    limit: Optional[int] = None,
    batch_size: int = 8,
) -> Tuple[Dict[str, MulticlassMetrics], Dict[str, BinaryMetrics], List[Dict], float]:
    """
    Returns
    -------
    mc_metrics   : {group_name: MulticlassMetrics}
    bin_metrics  : {group_name: BinaryMetrics}
    per_image    : list of per-image dicts (image, gt_attrs, pred_attrs)
    throughput   : images / second
    """
    # Init metrics objects
    mc_metrics: Dict[str, MulticlassMetrics] = {}
    for gname in MULTICLASS_GROUPS:
        grp = _group_by_name(gname)
        if grp:
            mc_metrics[gname] = MulticlassMetrics(gname, grp.labels)

    bin_metrics: Dict[str, BinaryMetrics] = {
        g: BinaryMetrics(g) for g in BINARY_GROUPS
    }

    # Load metadata — de-duplicate image filenames.
    # (5 images in the dataset have 2 conflicting GT rows; keep first only.)
    records = []
    seen_images: set = set()
    skipped = 0
    with open(metadata_path) as f:
        for line in f:
            obj = json.loads(line)
            img_path = image_dir / obj["image"]
            if not img_path.exists():
                continue
            if obj["image"] in seen_images:
                skipped += 1
                continue
            seen_images.add(obj["image"])
            records.append((img_path, obj["attributes"]))

    if skipped:
        logger.warning(
            "Skipped %d duplicate image entries (same filename, different GT).",
            skipped,
        )

    if limit:
        records = records[:limit]

    logger.info("Running benchmark on %d images …", len(records))
    per_image_results = []
    t0 = time.perf_counter()

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        paths, gt_attrs_list = zip(*batch)

        # Load images
        pil_imgs = [Image.open(p).convert("RGB") for p in paths]

        # Run detector (batch)
        pred_results = detector.detect_batch(pil_imgs, category="person")

        for path, gt_attrs, pred in zip(paths, gt_attrs_list, pred_results):
            row: Dict[str, Any] = {
                "image": path.name,
                "gt": {},
                "pred": {},
                "correct": {},
            }

            # ── Multi-class groups ──────────────────────────────────────────
            for gname in MULTICLASS_GROUPS:
                gt_val = gt_attrs.get(gname)
                if gt_val is None:
                    continue
                gt_label_str = GT_LABEL_MAP.get(gname, {}).get(gt_val)
                if gt_label_str is None:
                    logger.debug("Unknown GT value '%s' for group '%s'", gt_val, gname)
                    continue

                grp = _group_by_name(gname)
                if grp is None:
                    continue
                gt_idx = _gt_label_idx(grp, gt_label_str)
                if gt_idx is None:
                    logger.debug("Label not in schema: '%s'", gt_label_str)
                    continue

                pred_label_str = pred.get(gname, {}).get("label", "")
                pred_idx = _gt_label_idx(grp, pred_label_str)
                if pred_idx is None:
                    pred_idx = -1

                mc_metrics[gname].update(gt_idx, pred_idx if pred_idx >= 0 else gt_idx ^ 1)
                row["gt"][gname] = gt_val
                row["pred"][gname] = pred.get(gname, {}).get("short", "?")
                row["correct"][gname] = (gt_idx == pred_idx)

            # ── Binary / accessory groups ───────────────────────────────────
            accessory = gt_attrs.get("accessory", "none")
            gt_binary = _gt_for_binary_groups(accessory)
            for gname, gt_bool in gt_binary.items():
                pred_attr = pred.get(gname, {})
                pred_bool = pred_attr.get("present", False)
                bin_metrics[gname].update(gt_bool, pred_bool)
                row["gt"][gname] = gt_bool
                row["pred"][gname] = pred_bool
                row["correct"][gname] = (gt_bool == pred_bool)

            per_image_results.append(row)

        processed = min(start + batch_size, len(records))
        if processed % 100 == 0 or processed == len(records):
            elapsed = time.perf_counter() - t0
            logger.info(
                "  %d / %d  (%.1f img/s)",
                processed, len(records), processed / elapsed,
            )

    total_time = time.perf_counter() - t0
    throughput = len(records) / total_time
    return mc_metrics, bin_metrics, per_image_results, throughput


# ──────────────────────────────────────────────────────────────────────────────
# Report printer
# ──────────────────────────────────────────────────────────────────────────────

_W = 72   # console width

def _bar(acc: float, width: int = 30) -> str:
    filled = int(acc * width)
    return "█" * filled + "░" * (width - filled)


def print_report(
    mc_metrics: Dict[str, MulticlassMetrics],
    bin_metrics: Dict[str, BinaryMetrics],
    throughput: float,
) -> None:
    sep = "─" * _W

    print(f"\n{'SigLIP2 Attribute Detection — Benchmark Report':^{_W}}")
    print(sep)

    # ── Multi-class groups ────────────────────────────────────────────────
    print("\n  MULTI-CLASS GROUPS  (top-1 accuracy)\n")
    mc_accs = []
    for gname, m in mc_metrics.items():
        acc = m.accuracy
        mc_accs.append(acc)
        bar = _bar(acc)
        print(f"  {gname:<16} {bar}  {acc:6.1%}  ({m.correct}/{m.total})")

        # Per-class recall
        for i, lbl in enumerate(m.labels):
            row_total = sum(m.confusion[i])
            if row_total == 0:
                continue
            class_acc = m.confusion[i][i] / row_total
            short = m.short_label(i)
            print(f"    {'':2}{short:<34} {class_acc:5.1%}  (n={row_total})")

    # ── Binary groups ────────────────────────────────────────────────────
    print(f"\n  BINARY GROUPS  (sigmoid threshold)\n")
    print(f"  {'Group':<22} {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'TP':>4}  {'FP':>4}  {'TN':>4}  {'FN':>4}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}")
    bin_accs = []
    for gname, m in bin_metrics.items():
        bin_accs.append(m.accuracy)
        print(
            f"  {gname:<22}  {m.accuracy:6.1%}  {m.precision:6.1%}  "
            f"{m.recall:6.1%}  {m.f1:6.1%}  {m.tp:4d}  {m.fp:4d}  {m.tn:4d}  {m.fn:4d}"
        )

    # ── Summary ──────────────────────────────────────────────────────────
    all_accs = mc_accs + bin_accs
    mean_acc = sum(all_accs) / len(all_accs) if all_accs else 0.0

    print(f"\n{sep}")
    print(f"  Mean accuracy (all groups) : {mean_acc:.1%}")
    print(f"  Mean accuracy (multi-class): {sum(mc_accs)/len(mc_accs):.1%}" if mc_accs else "")
    print(f"  Mean accuracy (binary)     : {sum(bin_accs)/len(bin_accs):.1%}" if bin_accs else "")
    print(f"  Throughput                 : {throughput:.1f} images / second")
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark SigLIP2 attribute detection against ground truth."
    )
    p.add_argument(
        "--image_dir",
        required=True,
        help="Directory containing the crop images (e.g. pedestrian_1k/).",
    )
    p.add_argument(
        "--metadata",
        required=True,
        help="Path to metadata.jsonl with ground-truth attributes.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of images to evaluate (default: all).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for detector inference (default: 8).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help='Force device: "cuda" or "cpu". Default: auto.',
    )
    p.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Optional path to save per-image results as JSON.",
    )
    p.add_argument(
        "--use_vlm",
        action="store_true",
        help="Use VLM (Ollama Qwen-VL) instead of SigLIP2.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.use_vlm:
        import os
        if os.path.exists("/.dockerenv"):
            ollama_host = "http://host.docker.internal:11434"
        else:
            ollama_host = "http://localhost:11434"
        logger.info(f"Using VLMAttributeDetector with Ollama at {ollama_host}")
        detector = VLMAttributeDetector(ollama_host=ollama_host)
    else:
        detector = AttributeDetector(device=args.device)

    mc, bn, per_image, tput = run_benchmark(
        detector=detector,
        image_dir=Path(args.image_dir),
        metadata_path=Path(args.metadata),
        limit=args.limit,
        batch_size=args.batch_size,
    )

    print_report(mc, bn, tput)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialise metrics
        summary = {
            "throughput_img_per_sec": round(tput, 2),
            "multiclass": {
                gname: {
                    "accuracy": round(m.accuracy, 4),
                    "correct": m.correct,
                    "total": m.total,
                    "confusion": m.confusion,
                    "labels": m.labels,
                }
                for gname, m in mc.items()
            },
            "binary": {
                gname: {
                    "accuracy": round(m.accuracy, 4),
                    "precision": round(m.precision, 4),
                    "recall": round(m.recall, 4),
                    "f1": round(m.f1, 4),
                    "tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn,
                }
                for gname, m in bn.items()
            },
            "per_image": per_image,
        }
        out_path.write_text(json.dumps(summary, indent=2))
        logger.info("Saved detailed results → %s", out_path)
