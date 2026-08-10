"""
pipeline.py
───────────
Drop-in integration module.

Exposes a `CropAttributePipeline` that accepts detector track crops
(numpy BGR arrays) and a category label, and returns structured attribute
dictionaries ready for downstream consumers (Kafka, Redis, database, etc.).

Typical usage in a tracking loop:
──────────────────────────────────
    from pipeline import CropAttributePipeline

    pipe = CropAttributePipeline(device="cuda")

    # Inside your per-frame loop:
    for track in active_tracks:
        crop_bgr = frame[y1:y2, x1:x2]
        attrs = pipe.process(
            crop_bgr=crop_bgr,
            track_id=track.id,
            category=track.category,   # "person" or "vehicle"
        )
        # attrs is a flat dict ready for JSON serialisation
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal

import numpy as np

from attribute_detector import AttributeDetector
from attribute_schemas import PERSON_ATTRIBUTE_GROUPS, VEHICLE_ATTRIBUTE_GROUPS

logger = logging.getLogger(__name__)

Category = Literal["person", "vehicle"]


def _flatten_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert nested AttributeResult dict to a flat key-value dict.

    Example output:
        {
            "gender": "female person",
            "gender_score": 0.91,
            "top_type": "t-shirt",
            "top_type_score": 0.77,
            "wearing_cap_or_hat": True,
            "wearing_cap_or_hat_score": 0.62,
            ...
        }
    """
    flat: Dict[str, Any] = {}
    for group_name, attr in result.items():
        if "present" in attr:
            flat[group_name] = attr["present"]
        else:
            flat[group_name] = attr["short"]
        flat[f"{group_name}_score"] = attr["score"]
    return flat


def _is_motorbike(flat: Dict[str, Any]) -> bool:
    vtype = str(flat.get("vehicle_type", "")).lower()
    return any(k in vtype for k in ("motorbike", "motorcycle", "scooter"))


class CropAttributePipeline:
    """
    Stateless pipeline wrapper around AttributeDetector.

    Parameters
    ----------
    device : str | None
        "cuda" / "cpu" / None (auto).
    run_person_groups : list[str] | None
        Subset of person attribute group names to run.  Defaults to all.
    run_vehicle_groups : list[str] | None
        Subset of vehicle attribute group names to run.  Defaults to all.
    motorbike_passenger_threshold : float
        Minimum vehicle_type score for the motorbike passenger group to be
        included in the output.  Set to 0.0 to always include it.
    """

    def __init__(
        self,
        device: str | None = None,
        run_person_groups: List[str] | None = None,
        run_vehicle_groups: List[str] | None = None,
        motorbike_passenger_threshold: float = 0.0,
    ) -> None:
        self.detector = AttributeDetector(device=device)
        self._person_groups = run_person_groups
        self._vehicle_groups = run_vehicle_groups
        self._moto_thresh = motorbike_passenger_threshold

    # ── Public API ─────────────────────────────────────────────────────────

    def process(
        self,
        crop_bgr: np.ndarray,
        track_id: int | str,
        category: Category,
    ) -> Dict[str, Any]:
        """
        Process a single BGR crop.

        Returns
        -------
        dict with keys:
            track_id, category, <flat attributes>
        """
        result = self.detector.detect(crop_bgr, category=category, bgr=True)
        result = self._filter_groups(result, category)
        flat = _flatten_result(result)

        # For vehicles: suppress motorbike_passengers if not a motorbike
        if category == "vehicle" and not _is_motorbike(flat):
            flat.pop("motorbike_passengers", None)
            flat.pop("motorbike_passengers_score", None)

        return {"track_id": track_id, "category": category, **flat}

    def process_batch(
        self,
        crops_bgr: List[np.ndarray],
        track_ids: List[int | str],
        category: Category,
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of BGR crops (more efficient GPU utilisation).
        """
        assert len(crops_bgr) == len(track_ids), \
            "crops_bgr and track_ids must have the same length."

        results = self.detector.detect_batch(crops_bgr, category=category, bgr=True)
        outputs = []
        for tid, result in zip(track_ids, results):
            result = self._filter_groups(result, category)
            flat = _flatten_result(result)
            if category == "vehicle" and not _is_motorbike(flat):
                flat.pop("motorbike_passengers", None)
                flat.pop("motorbike_passengers_score", None)
            outputs.append({"track_id": tid, "category": category, **flat})
        return outputs

    # ── Private helpers ────────────────────────────────────────────────────

    def _filter_groups(
        self,
        result: Dict[str, Any],
        category: Category,
    ) -> Dict[str, Any]:
        """Keep only the requested group subset (if configured)."""
        whitelist = (
            self._person_groups
            if category == "person"
            else self._vehicle_groups
        )
        if whitelist is None:
            return result
        return {k: v for k, v in result.items() if k in whitelist}
