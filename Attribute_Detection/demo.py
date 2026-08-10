"""
demo.py
───────
End-to-end demo for the SigLIP2 Attribute Detection pipeline.

Run modes
─────────
1. Single image file (person or vehicle):
       python demo.py --image path/to/crop.jpg --category person

2. Folder of crops:
       python demo.py --image_dir path/to/crops/ --category vehicle

3. Interactive webcam test (grab a frame, show result):
       python demo.py --webcam --category person

4. Batch benchmark on a folder:
       python demo.py --image_dir path/to/crops/ --benchmark

The script annotates images with the top attributes and saves results to
a JSON file alongside each image.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

from attribute_detector import AttributeDetector, format_attributes
from attribute_schemas import VEHICLE_ATTRIBUTE_GROUPS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")

# ──────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────────────────────

_CATEGORY_COLOR = {"person": (0, 200, 80), "vehicle": (0, 160, 255)}

_DISPLAY_GROUPS_PERSON = [
    "gender", "age_range", "top_type", "top_color",
    "bottom_type", "bottom_color",
    "footwear_type", "footwear_color",
    "wearing_cap_or_hat", "wearing_helmet", "carrying_bag",
]
_DISPLAY_GROUPS_VEHICLE = [
    "vehicle_type", "vehicle_color",
    "motorbike_passengers", "vehicle_orientation",
]


def _is_motorbike_result(result: dict) -> bool:
    """Return True if the detected vehicle type is a motorbike/scooter."""
    vtype = result.get("vehicle_type", {}).get("short", "")
    return any(k in vtype.lower() for k in ("motorbike", "motorcycle", "scooter"))


def draw_attributes(
    frame: np.ndarray,
    result: dict,
    category: str,
    x: int = 10,
    y: int = 10,
) -> np.ndarray:
    """Overlay attribute text on an image (OpenCV BGR frame)."""
    vis = frame.copy()
    groups = (
        _DISPLAY_GROUPS_PERSON
        if category == "person"
        else _DISPLAY_GROUPS_VEHICLE
    )

    color = _CATEGORY_COLOR.get(category, (255, 255, 255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    line_h = 22
    pad = 8

    # Build text lines
    lines: List[str] = [f"[ {category.upper()} ]"]
    for gname in groups:
        if gname not in result:
            continue
        # Skip motorbike passengers unless it's a motorbike
        if gname == "motorbike_passengers" and not _is_motorbike_result(result):
            continue
        attr = result[gname]
        short = attr["short"]
        score = attr["score"]
        if "present" in attr:
            val = ("✓ " + short) if attr["present"] else ("✗ " + short)
        else:
            val = short
        lines.append(f"{gname}: {val}  ({score:.0%})")

    # Draw semi-transparent background box
    max_w = max(
        cv2.getTextSize(ln, font, scale, thick)[0][0] for ln in lines
    ) + 2 * pad
    box_h = line_h * len(lines) + 2 * pad
    overlay = vis.copy()
    cv2.rectangle(overlay, (x, y), (x + max_w, y + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, vis, 0.35, 0, vis)

    for i, line in enumerate(lines):
        cy = y + pad + (i + 1) * line_h
        cv2.putText(vis, line, (x + pad, cy), font, scale, color, thick, cv2.LINE_AA)

    return vis


# ──────────────────────────────────────────────────────────────────────────────
# Main routines
# ──────────────────────────────────────────────────────────────────────────────

def run_single(args: argparse.Namespace, detector: AttributeDetector) -> None:
    img_path = Path(args.image)
    assert img_path.exists(), f"File not found: {img_path}"

    pil_img = Image.open(img_path).convert("RGB")
    logger.info("Running on %s  (%dx%d) …", img_path.name, *pil_img.size)

    t0 = time.perf_counter()
    result = detector.detect(pil_img, category=args.category)
    elapsed = time.perf_counter() - t0

    print(format_attributes(result, category=args.category))
    print(f"\nInference time: {elapsed*1000:.1f} ms")

    # Save annotated image
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    vis = draw_attributes(frame, result, args.category)
    out_path = img_path.parent / (img_path.stem + "_attrs.jpg")
    cv2.imwrite(str(out_path), vis)
    logger.info("Saved annotated image → %s", out_path)

    # Save JSON
    json_path = img_path.parent / (img_path.stem + "_attrs.json")
    json_path.write_text(json.dumps(result, indent=2))
    logger.info("Saved attribute JSON → %s", json_path)


def run_folder(args: argparse.Namespace, detector: AttributeDetector) -> None:
    folder = Path(args.image_dir)
    assert folder.is_dir(), f"Not a directory: {folder}"
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    assert paths, f"No images found in {folder}"

    logger.info("Found %d images in %s", len(paths), folder)
    all_results = {}

    if args.benchmark:
        # Batch mode
        batch_size = 16
        timings = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            pil_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            t0 = time.perf_counter()
            results = detector.detect_batch(pil_imgs, category=args.category)
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)
            for path, res in zip(batch_paths, results):
                all_results[path.name] = res
        total = sum(timings)
        fps = len(paths) / total
        logger.info(
            "Batch benchmark: %d images in %.2f s  →  %.1f imgs/s",
            len(paths), total, fps,
        )
    else:
        for p in paths:
            pil_img = Image.open(p).convert("RGB")
            result = detector.detect(pil_img, category=args.category)
            all_results[p.name] = result

            # Annotate & save
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            vis = draw_attributes(frame, result, args.category)
            out_p = folder / (p.stem + "_attrs.jpg")
            cv2.imwrite(str(out_p), vis)

    summary_path = folder / "attribute_results.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Saved summary → %s", summary_path)


def run_webcam(args: argparse.Namespace, detector: AttributeDetector) -> None:
    """Grab live frames and show attributes in real-time."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam.")
        return

    logger.info("Webcam open. Press 'q' to quit, 'space' to freeze a frame.")
    frozen = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            frozen = frame.copy()

        display_frame = frozen if frozen is not None else frame
        pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(pil, category=args.category)
        vis = draw_attributes(display_frame, result, args.category)
        cv2.imshow("SigLIP2 Attribute Detection", vis)

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SigLIP2 attribute detection demo."
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=str, help="Path to a single image crop.")
    mode.add_argument("--image_dir", type=str, help="Folder of image crops.")
    mode.add_argument(
        "--webcam", action="store_true", help="Use webcam as input."
    )

    p.add_argument(
        "--category",
        choices=["person", "vehicle"],
        default="person",
        help="Attribute schema to use.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help='Force device: "cuda", "cpu". Default: auto.',
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Batch-process folder and report throughput (requires --image_dir).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detector = AttributeDetector(device=args.device)

    if args.image:
        run_single(args, detector)
    elif args.image_dir:
        run_folder(args, detector)
    elif args.webcam:
        run_webcam(args, detector)
