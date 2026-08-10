"""
attribute_detector.py
─────────────────────
Attribute detection for person and vehicle crops using
google/siglip2-base-patch16-224 (zero-shot image classification).

Usage
-----
    from attribute_detector import AttributeDetector

    detector = AttributeDetector()

    # Single crop (PIL.Image or numpy HWC-BGR/RGB)
    result = detector.detect(image, category="person")
    print(result)

    # Batch of crops
    results = detector.detect_batch(images, category="vehicle")
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Literal, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

from attribute_schemas import (
    PERSON_ATTRIBUTE_GROUPS,
    VEHICLE_ATTRIBUTE_GROUPS,
    AttributeGroup,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────────

ImageInput = Union[Image.Image, np.ndarray]
Category = Literal["person", "vehicle"]

AttributeResult = Dict[str, Union[str, float, bool]]
"""
Per-group output dict, e.g.:
    {
        "label": "a person wearing a t-shirt",
        "short": "t-shirt",               # cleaned label
        "score": 0.83,
        # For binary groups only:
        "present": True
    }
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helper: clean a candidate label to a short human-readable string
# ──────────────────────────────────────────────────────────────────────────────

_STRIP_PREFIXES = [
    "a photo of an ",
    "a photo of a ",
    "a photo of ",
    "a person wearing ",
    "a person with ",
    "a person carrying ",
    "a person facing ",
    "a person viewed ",
    "a person not wearing ",
    "a person not carrying ",
    "seen from ",
    "a person ",
    "a motorbike with ",
    "a vehicle facing ",
    "a vehicle viewed ",
    "a vehicle ",
    "a ",
    "an ",
]


def _short_label(label: str) -> str:
    """Strip boilerplate from a candidate label for display."""
    label = label.strip()
    for prefix in _STRIP_PREFIXES:
        if label.lower().startswith(prefix):
            return label[len(prefix):].strip()
    return label


# ──────────────────────────────────────────────────────────────────────────────
# Core detector
# ──────────────────────────────────────────────────────────────────────────────

class AttributeDetector:
    """
    Wraps google/siglip2-base-patch16-224 to perform multi-group zero-shot
    attribute classification on person / vehicle crops.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str | None
        "cuda", "cpu", or None (auto-select).
    batch_labels : bool
        If True, pack all candidate labels from all groups into a single
        model forward pass (faster, slightly more memory).  If False, run
        one forward pass per group (slower but lower peak VRAM).
    """

    MODEL_NAME = "google/siglip2-base-patch16-384"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        batch_labels: bool = True,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_labels = batch_labels

        logger.info("Loading SigLIP2 model: %s on %s …", model_name, device)
        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = (
            AutoModel.from_pretrained(model_name)
            .to(device)
            .eval()
        )
        logger.info("Model loaded in %.2f s", time.perf_counter() - t0)

        # ── Extract SigLIP2's learned logit_scale and logit_bias ──────────
        # Without these, raw cosine similarities are ~0.1–0.3, producing a
        # near-uniform softmax distribution and destroying confidence scores.
        # logit_scale is typically exp(log_scale) ≈ 10–100 after training.
        with torch.no_grad():
            raw_scale = getattr(self.model, "logit_scale", None)
            self._logit_scale: float = (
                float(raw_scale.exp().item()) if raw_scale is not None else 1.0
            )
            raw_bias = getattr(self.model, "logit_bias", None)
            self._logit_bias: float | None = (
                float(raw_bias.item()) if raw_bias is not None else None
            )
        logger.info(
            "Logit scale: %.2f  bias: %s",
            self._logit_scale,
            f"{self._logit_bias:.3f}" if self._logit_bias is not None else "none",
        )

        # Pre-cache text encodings for each category to avoid re-encoding
        # the same prompts on every inference call.
        self._text_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._encode_all_labels()

    # ── Private helpers ────────────────────────────────────────────────────

    def _encode_all_labels(self) -> None:
        """Pre-compute and cache text embeddings for all attribute groups."""
        for category, groups in [
            ("person", PERSON_ATTRIBUTE_GROUPS),
            ("vehicle", VEHICLE_ATTRIBUTE_GROUPS),
        ]:
            self._text_cache[category] = {}
            for group in groups:
                self._text_cache[category][group.name] = (
                    self._encode_texts(group.labels)
                )
        logger.debug("Text embeddings cached for all attribute groups.")

    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Return normalised text embeddings, shape (N, D)."""
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self.device)
        out = self.model.get_text_features(**inputs)
        # AutoModel returns a BaseModelOutputWithPooling dataclass;
        # extract the pooled CLS embedding.
        text_feats = out.pooler_output if hasattr(out, "pooler_output") else out
        return text_feats / text_feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Return normalised image embedding, shape (1, D)."""
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)
        out = self.model.get_image_features(**inputs)
        img_feats = out.pooler_output if hasattr(out, "pooler_output") else out
        return img_feats / img_feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _encode_images_batch(
        self, images: List[Image.Image]
    ) -> torch.Tensor:
        """Return normalised image embeddings, shape (B, D)."""
        inputs = self.processor(
            images=images,
            return_tensors="pt",
        ).to(self.device)
        out = self.model.get_image_features(**inputs)
        img_feats = out.pooler_output if hasattr(out, "pooler_output") else out
        return img_feats / img_feats.norm(dim=-1, keepdim=True)

    @staticmethod
    def _to_pil(image: ImageInput) -> Image.Image:
        """Convert numpy HWC-BGR or HWC-RGB array to PIL.Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                # Greyscale
                return Image.fromarray(image).convert("RGB")
            if image.shape[2] == 4:
                return Image.fromarray(image[..., :3]).convert("RGB")
            # Detect BGR (OpenCV default) by a simple heuristic: caller must
            # explicitly state bgr=True; otherwise assume RGB.
            return Image.fromarray(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    # ── Public API ─────────────────────────────────────────────────────────

    def detect(
        self,
        image: ImageInput,
        category: Category = "person",
        bgr: bool = False,
    ) -> Dict[str, AttributeResult]:
        """
        Run attribute detection on a single crop.

        Parameters
        ----------
        image : PIL.Image | np.ndarray
            Crop of the person or vehicle.
        category : "person" | "vehicle"
            Which attribute schema to use.
        bgr : bool
            Set True if `image` is a numpy array in BGR channel order
            (e.g., directly from OpenCV).

        Returns
        -------
        dict  {group_name: AttributeResult}
        """
        pil = self._to_pil(image)
        if bgr and isinstance(image, np.ndarray):
            pil = Image.fromarray(image[..., ::-1]).convert("RGB")

        img_emb = self._encode_image(pil)  # (1, D)
        groups = (
            PERSON_ATTRIBUTE_GROUPS
            if category == "person"
            else VEHICLE_ATTRIBUTE_GROUPS
        )
        return self._score_groups(img_emb, groups, category)

    def detect_batch(
        self,
        images: List[ImageInput],
        category: Category = "person",
        bgr: bool = False,
    ) -> List[Dict[str, AttributeResult]]:
        """
        Run attribute detection on a batch of crops.

        Returns a list of result dicts, one per image.
        """
        pil_images = []
        for img in images:
            if bgr and isinstance(img, np.ndarray):
                pil_images.append(
                    Image.fromarray(img[..., ::-1]).convert("RGB")
                )
            else:
                pil_images.append(self._to_pil(img))

        img_embs = self._encode_images_batch(pil_images)  # (B, D)
        groups = (
            PERSON_ATTRIBUTE_GROUPS
            if category == "person"
            else VEHICLE_ATTRIBUTE_GROUPS
        )

        results = []
        for i in range(img_embs.shape[0]):
            emb_i = img_embs[i : i + 1]  # (1, D)
            results.append(self._score_groups(emb_i, groups, category))
        return results

    # ── Scoring logic ──────────────────────────────────────────────────────

    def _score_groups(
        self,
        img_emb: torch.Tensor,          # (1, D)
        groups: List[AttributeGroup],
        category: str,
    ) -> Dict[str, AttributeResult]:
        """
        For each attribute group, apply SigLIP2's learned logit_scale then:
          - Multi-class groups → softmax over scale*cos_sim logits (+ bias,
                                 which cancels in softmax anyway).
          - Binary groups      → softmax over scale*cos_sim for [positive,
                                 negative] labels WITHOUT bias.
                                 Rationale: logit_bias=-16.77 is trained for
                                 retrieval (near-zero base rate). For zero-shot
                                 binary classification we want a RELATIVE
                                 comparison ("which label fits better?"), not an
                                 absolute probability. Bias cancels in softmax,
                                 so we just skip it. Threshold = 0.5 = "positive
                                 beats negative".
        """
        output: Dict[str, AttributeResult] = {}
        text_cache = self._text_cache[category]

        for group in groups:
            text_embs = text_cache[group.name]  # (N, D)

            # Cosine similarity (both sides are L2-normalised): (N,)
            cos_sims = (img_emb @ text_embs.T).squeeze(0)

            if group.positive_idx is not None:
                # ── Binary attribute — relative softmax (NO bias) ──────────
                # logit_bias = -16.77 kills sigmoid for low cos_sim values
                # (needs cos_sim > 0.149 just to reach P=0.5).  Instead we
                # compare positive vs negative in a softmax: whichever label
                # has higher cosine similarity wins.  Threshold=0.5 means
                # "positive prompt wins the head-to-head comparison".
                scaled = cos_sims * self._logit_scale   # bias intentionally omitted
                probs = torch.softmax(scaled, dim=0)
                pos_prob = float(probs[group.positive_idx].item())
                present = pos_prob >= group.threshold   # threshold=0.5 by default
                output[group.name] = {
                    "label": group.labels[group.positive_idx],
                    "short": _short_label(group.labels[group.positive_idx]),
                    "score": round(pos_prob, 4),
                    "present": present,
                }
            else:
                # ── Multi-class — softmax over the candidate set ───────────
                # Bias cancels in softmax, scale sharpens the distribution.
                logits = cos_sims * self._logit_scale + (
                    self._logit_bias if self._logit_bias is not None else 0.0
                )
                probs = torch.softmax(logits, dim=0)
                best_idx = int(probs.argmax().item())
                output[group.name] = {
                    "label": group.labels[best_idx],
                    "short": _short_label(group.labels[best_idx]),
                    "score": round(float(probs[best_idx].item()), 4),
                }

        return output


# ──────────────────────────────────────────────────────────────────────────────
# Convenience formatting
# ──────────────────────────────────────────────────────────────────────────────

def format_attributes(
    result: Dict[str, AttributeResult],
    category: Category = "person",
    min_score: float = 0.0,
    skip_groups: List[str] | None = None,
) -> str:
    """
    Return a human-readable summary of a detection result.

    Parameters
    ----------
    result : dict
        Output from AttributeDetector.detect().
    category : "person" | "vehicle"
        Used to determine which groups to skip for vehicles.
    min_score : float
        Groups whose top score is below this value are shown with a ⚠ marker.
    skip_groups : list[str] | None
        Group names to exclude from output (e.g., irrelevant groups for
        non-motorbike vehicles).
    """
    skip_groups = skip_groups or []
    lines = [f"[{category.upper()} ATTRIBUTES]"]
    for group_name, attr in result.items():
        if group_name in skip_groups:
            continue
        score = attr["score"]
        short = attr["short"]
        if "present" in attr:
            value = "✓ yes" if attr["present"] else "✗ no"
        else:
            value = short
        marker = "⚠" if score < min_score else " "
        lines.append(f"  {marker} {group_name:<25} {value:<35} ({score:.2%})")
    return "\n".join(lines)
