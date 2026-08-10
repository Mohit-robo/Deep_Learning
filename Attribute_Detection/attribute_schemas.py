"""
Attribute schemas for person and vehicle crops.

Candidate labels derived directly from the pedestrian_1k dataset vocabulary
(metadata.jsonl).  Every label in a group is a candidate for SigLIP2
zero-shot scoring.

Binary groups (positive_idx set) use sigmoid on the positive label's scaled
logit — matches SigLIP's training objective.  threshold=0.5 is correct
for the calibrated logit_scale (softmax gives a relative comparison, so 0.5 means positive beats negative).

Prompt style: "a photo of a person …" aligns with SigLIP's WebLI pre-training
distribution and yields better scores than bare noun phrases.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AttributeGroup:
    """A single mutually-exclusive attribute group."""
    name: str
    labels: List[str]
    positive_idx: Optional[int] = None   # set → binary (sigmoid) group
    threshold: float = 0.5              # sigmoid threshold for binary groups


# ---------------------------------------------------------------------------
# PERSON attribute groups  (derived from pedestrian_1k metadata.jsonl)
# ---------------------------------------------------------------------------

PERSON_ATTRIBUTE_GROUPS: List[AttributeGroup] = [

    # ── Gender  (2 values: man / woman) ───────────────────────────────────
    AttributeGroup(
        name="gender",
        labels=[
            "a photo of a man",
            "a photo of a woman",
        ],
    ),

    # ── Age  (4 values) ───────────────────────────────────────────────────
    AttributeGroup(
        name="age",
        labels=[
            "a photo of a teenager",
            "a photo of a young adult",
            "a photo of a middle-aged person",
            "a photo of an elderly person",
        ],
    ),

    # ── Viewing angle  (3 values) ─────────────────────────────────────────
    AttributeGroup(
        name="angle",
        labels=[
            "a photo of a person seen from the front",
            "a photo of a person seen from the back",
            "a photo of a person seen from the side",
        ],
    ),

    # ── Hair colour  (6 values) ───────────────────────────────────────────
    AttributeGroup(
        name="hair",
        labels=[
            "a photo of a person with black hair",
            "a photo of a person with brown hair",
            "a photo of a person with blonde hair",
            "a photo of a person with grey or white hair",
            "a photo of a person with red or auburn hair",
            "a photo of a person with shaved or very short hair",
        ],
    ),

    # ── Top / upper-body garment type  (6 values) ─────────────────────────
    AttributeGroup(
        name="top_type",
        labels=[
            "a photo of a person wearing a t-shirt",
            "a photo of a person wearing a tank top",
            "a photo of a person wearing a hoodie",
            "a photo of a person wearing a knit sweater",
            "a photo of a person wearing a leather jacket",
            "a photo of a person wearing a puffer jacket",
        ],
    ),

    # ── Top / upper-body garment colour  (7 values) ───────────────────────
    AttributeGroup(
        name="top_color",
        labels=[
            "a photo of a person wearing a white top",
            "a photo of a person wearing a black top",
            "a photo of a person wearing a grey top",
            "a photo of a person wearing a navy blue top",
            "a photo of a person wearing a bright red top",
            "a photo of a person wearing a forest green top",
            "a photo of a person wearing a yellow top",
        ],
    ),

    # ── Bottom / lower-body garment type  (6 values) ──────────────────────
    AttributeGroup(
        name="bottom_type",
        labels=[
            "a photo of a person wearing denim jeans",
            "a photo of a person wearing formal trousers",
            "a photo of a person wearing cargo pants",
            "a photo of a person wearing sweatpants",
            "a photo of a person wearing shorts",
            "a photo of a person wearing leggings",
        ],
    ),

    # ── Bottom / lower-body garment colour  (5 values) ────────────────────
    AttributeGroup(
        name="bottom_color",
        labels=[
            "a photo of a person wearing white trousers or bottoms",
            "a photo of a person wearing black trousers or bottoms",
            "a photo of a person wearing blue or dark blue trousers or bottoms",
            "a photo of a person wearing dark grey trousers or bottoms",
            "a photo of a person wearing khaki or beige trousers or bottoms",
        ],
    ),

    # ── Accessory / carrying  (6 values + none)  ── binary groups ─────────
    AttributeGroup(
        name="carrying_backpack",
        labels=[
            "a photo of a person carrying a backpack",
            "a photo of a person not carrying a backpack",
        ],
        positive_idx=0,
        threshold=0.5,
    ),
    AttributeGroup(
        name="carrying_handbag",
        labels=[
            "a photo of a person carrying a handbag or purse",
            "a photo of a person not carrying a handbag",
        ],
        positive_idx=0,
        threshold=0.5,
    ),
    AttributeGroup(
        name="holding_umbrella",
        labels=[
            "a photo of a person holding an umbrella",
            "a photo of a person not holding an umbrella",
        ],
        positive_idx=0,
        threshold=0.5,
    ),
    AttributeGroup(
        name="holding_smartphone",
        labels=[
            "a photo of a person holding a smartphone or phone",
            "a photo of a person not holding a phone",
        ],
        positive_idx=0,
        threshold=0.5,
    ),
    AttributeGroup(
        name="wearing_cap",
        labels=[
            "a photo of a person wearing a baseball cap or hat",
            "a photo of a person with no cap or hat",
        ],
        positive_idx=0,
        threshold=0.5,
    ),
    AttributeGroup(
        name="wearing_sunglasses",
        labels=[
            "a photo of a person wearing sunglasses",
            "a photo of a person not wearing sunglasses",
        ],
        positive_idx=0,
        threshold=0.5,
    ),
]


# ---------------------------------------------------------------------------
# VEHICLE attribute groups
# ---------------------------------------------------------------------------

VEHICLE_ATTRIBUTE_GROUPS: List[AttributeGroup] = [

    # ── Vehicle type ───────────────────────────────────────────────────────
    AttributeGroup(
        name="vehicle_type",
        labels=[
            "a photo of a car or sedan",
            "a photo of a truck or lorry",
            "a photo of a tempo or mini-truck",
            "a photo of a bus",
            "a photo of a motorbike or motorcycle",
            "a photo of a bicycle or cycle",
            "a photo of an auto-rickshaw or tuk-tuk",
            "a photo of a van or minivan",
            "a photo of an SUV or jeep",
            "a photo of a scooter",
        ],
    ),

    # ── Vehicle colour ─────────────────────────────────────────────────────
    AttributeGroup(
        name="vehicle_color",
        labels=[
            "a white vehicle",
            "a black vehicle",
            "a silver or grey vehicle",
            "a red vehicle",
            "a blue or navy vehicle",
            "a green vehicle",
            "a yellow vehicle",
            "an orange vehicle",
            "a brown vehicle",
            "a multi-colored or painted vehicle",
        ],
    ),

    # ── Motorbike passenger count ──────────────────────────────────────────
    AttributeGroup(
        name="motorbike_passengers",
        labels=[
            "a motorbike with one rider only",
            "a motorbike with two people, a rider and one pillion passenger",
            "a motorbike with three or more people",
        ],
    ),

    # ── Vehicle orientation ────────────────────────────────────────────────
    AttributeGroup(
        name="vehicle_orientation",
        labels=[
            "a vehicle facing towards the camera head-on",
            "a vehicle facing away from the camera",
            "a vehicle viewed from the side",
            "a vehicle viewed at a diagonal angle",
        ],
    ),
]
