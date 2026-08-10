"""
test_attribute_detection.py
───────────────────────────
Lightweight self-tests that run without a GPU and without real images.

Run:
    pytest test_attribute_detection.py -v

What is tested
──────────────
1. Schema integrity – every label group has at least 2 labels.
2. _short_label – strips prefixes correctly.
3. _flatten_result – output shape and types.
4. CropAttributePipeline.process / process_batch – with a mock detector.
5. format_attributes – output is non-empty string.
6. Motorbike passenger filtering – suppressed for non-motorbike vehicles.
"""

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Make local modules importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from attribute_schemas import (
    PERSON_ATTRIBUTE_GROUPS,
    VEHICLE_ATTRIBUTE_GROUPS,
    AttributeGroup,
)
from attribute_detector import _short_label, _flatten_result  # type: ignore[attr-defined]
from attribute_detector import format_attributes, AttributeResult


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_multiclass_result(label: str, score: float) -> AttributeResult:
    return {"label": label, "short": _short_label(label), "score": score}


def _make_binary_result(label: str, score: float, present: bool) -> AttributeResult:
    return {"label": label, "short": _short_label(label), "score": score, "present": present}


SAMPLE_PERSON_RESULT: Dict[str, AttributeResult] = {
    "gender": _make_multiclass_result("a photo of a male person", 0.87),
    "age_range": _make_multiclass_result("a photo of a young adult", 0.72),
    "top_type": _make_multiclass_result("a person wearing a t-shirt", 0.81),
    "top_color": _make_multiclass_result("a person wearing a blue top", 0.65),
    "bottom_type": _make_multiclass_result("a person wearing jeans", 0.78),
    "bottom_color": _make_multiclass_result("a person wearing blue trousers or bottoms", 0.61),
    "footwear_type": _make_multiclass_result("a person wearing shoes or sneakers", 0.55),
    "footwear_color": _make_multiclass_result("a person with white footwear", 0.52),
    "wearing_cap_or_hat": _make_binary_result(
        "a person wearing a cap or hat", 0.30, False
    ),
    "wearing_helmet": _make_binary_result(
        "a person wearing a helmet", 0.10, False
    ),
    "carrying_bag": _make_binary_result(
        "a person carrying a bag or backpack", 0.70, True
    ),
    "body_orientation": _make_multiclass_result(
        "a person facing the camera from the front", 0.84
    ),
}

SAMPLE_VEHICLE_RESULT_MOTORBIKE: Dict[str, AttributeResult] = {
    "vehicle_type": _make_multiclass_result(
        "a photo of a motorbike or motorcycle", 0.91
    ),
    "vehicle_color": _make_multiclass_result("a black vehicle", 0.77),
    "motorbike_passengers": _make_multiclass_result(
        "a motorbike with two people (rider and one pillion passenger)", 0.68
    ),
    "vehicle_orientation": _make_multiclass_result(
        "a vehicle viewed from the side", 0.55
    ),
}

SAMPLE_VEHICLE_RESULT_BUS: Dict[str, AttributeResult] = {
    "vehicle_type": _make_multiclass_result("a photo of a bus", 0.88),
    "vehicle_color": _make_multiclass_result("a white vehicle", 0.71),
    "motorbike_passengers": _make_multiclass_result(
        "a motorbike with one rider and no passenger", 0.33
    ),
    "vehicle_orientation": _make_multiclass_result(
        "a vehicle facing towards the camera", 0.60
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSchemas:
    def test_person_groups_non_empty(self):
        assert len(PERSON_ATTRIBUTE_GROUPS) >= 8

    def test_vehicle_groups_non_empty(self):
        assert len(VEHICLE_ATTRIBUTE_GROUPS) >= 4

    @pytest.mark.parametrize("group", PERSON_ATTRIBUTE_GROUPS)
    def test_person_group_has_multiple_labels(self, group: AttributeGroup):
        assert len(group.labels) >= 2, f"Group '{group.name}' has < 2 labels"

    @pytest.mark.parametrize("group", VEHICLE_ATTRIBUTE_GROUPS)
    def test_vehicle_group_has_multiple_labels(self, group: AttributeGroup):
        assert len(group.labels) >= 2, f"Group '{group.name}' has < 2 labels"

    def test_binary_groups_have_positive_idx(self):
        binary_groups = [g for g in PERSON_ATTRIBUTE_GROUPS if g.positive_idx is not None]
        assert len(binary_groups) >= 1, "Expected at least one binary person group"

    def test_positive_idx_in_range(self):
        for group in PERSON_ATTRIBUTE_GROUPS + VEHICLE_ATTRIBUTE_GROUPS:
            if group.positive_idx is not None:
                assert 0 <= group.positive_idx < len(group.labels), \
                    f"positive_idx out of range in group '{group.name}'"


class TestShortLabel:
    def test_strips_person_wearing(self):
        assert _short_label("a person wearing a t-shirt") == "a t-shirt"

    def test_strips_photo_of_a(self):
        assert _short_label("a photo of a bus") == "bus"

    def test_strips_photo_of(self):
        result = _short_label("a photo of a male person")
        assert "photo" not in result.lower()

    def test_no_change_when_no_prefix(self):
        label = "male"
        assert _short_label(label) == "male"


class TestFlattenResult:
    def test_multiclass_group_produces_string_value(self):
        flat = _flatten_result(SAMPLE_PERSON_RESULT)
        assert isinstance(flat["gender"], str)
        assert isinstance(flat["gender_score"], float)

    def test_binary_group_produces_bool_value(self):
        flat = _flatten_result(SAMPLE_PERSON_RESULT)
        assert isinstance(flat["wearing_cap_or_hat"], bool)
        assert isinstance(flat["wearing_cap_or_hat_score"], float)

    def test_score_in_range(self):
        flat = _flatten_result(SAMPLE_PERSON_RESULT)
        for key, val in flat.items():
            if key.endswith("_score"):
                assert 0.0 <= val <= 1.0, f"{key} score out of range: {val}"

    def test_all_groups_present(self):
        flat = _flatten_result(SAMPLE_PERSON_RESULT)
        for group_name in SAMPLE_PERSON_RESULT:
            assert group_name in flat


class TestFormatAttributes:
    def test_returns_non_empty_string(self):
        out = format_attributes(SAMPLE_PERSON_RESULT, category="person")
        assert isinstance(out, str)
        assert len(out) > 0

    def test_contains_category_header(self):
        out = format_attributes(SAMPLE_PERSON_RESULT, category="person")
        assert "PERSON" in out

    def test_vehicle_format(self):
        out = format_attributes(SAMPLE_VEHICLE_RESULT_MOTORBIKE, category="vehicle")
        assert "VEHICLE" in out


class TestMotorbikePassengerFiltering:
    """Motorbike passengers should be suppressed for non-motorbike vehicles."""

    def _mock_pipeline(self, mock_result: Dict[str, Any]):
        """Build a CropAttributePipeline whose inner detector is mocked."""
        from pipeline import CropAttributePipeline, _flatten_result

        with patch(
            "pipeline.AttributeDetector.__init__", return_value=None
        ):
            pipe = CropAttributePipeline.__new__(CropAttributePipeline)
            pipe.detector = MagicMock()
            pipe.detector.detect.return_value = mock_result
            pipe.detector.detect_batch.return_value = [mock_result]
            pipe._person_groups = None
            pipe._vehicle_groups = None
            pipe._moto_thresh = 0.0
        return pipe

    def test_passengers_included_for_motorbike(self):
        pipe = self._mock_pipeline(SAMPLE_VEHICLE_RESULT_MOTORBIKE)
        flat = pipe.process(np.zeros((64, 64, 3), dtype=np.uint8), 1, "vehicle")
        assert "motorbike_passengers" in flat

    def test_passengers_suppressed_for_bus(self):
        pipe = self._mock_pipeline(SAMPLE_VEHICLE_RESULT_BUS)
        flat = pipe.process(np.zeros((64, 64, 3), dtype=np.uint8), 2, "vehicle")
        assert "motorbike_passengers" not in flat

    def test_batch_returns_correct_length(self):
        pipe = self._mock_pipeline(SAMPLE_VEHICLE_RESULT_MOTORBIKE)
        crops = [np.zeros((64, 64, 3), dtype=np.uint8)] * 3
        results = pipe.process_batch(crops, [1, 2, 3], "vehicle")
        assert len(results) == 3
