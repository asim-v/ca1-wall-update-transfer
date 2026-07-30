from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from ca1_geometry.seams import SeamState


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_boundary_fragment_experience_did",
    ROOT / "scripts" / "run_boundary_fragment_experience_did.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (SeamState.WALL, SeamState.WALL, "shared_wall"),
        (SeamState.OPEN, SeamState.OPEN, "shared_open"),
        (SeamState.WALL, SeamState.OPEN, "wall_open"),
        (SeamState.OPEN, SeamState.WALL, "wall_open"),
        (SeamState.REVERSE_WALL, SeamState.WALL, None),
        (SeamState.OPEN, SeamState.CLOSED, None),
    ],
)
def test_state_pair_label(first, second, expected):
    assert MODULE.state_pair_label(first, second) == expected


def test_difference_in_differences():
    assert MODULE.difference_in_differences(
        first_wall=0.2,
        first_control=0.1,
        later_wall=0.5,
        later_control=0.2,
    ) == pytest.approx(0.2)


def test_difference_in_differences_rejects_nonfinite():
    assert math.isnan(
        MODULE.difference_in_differences(
            first_wall=float("nan"),
            first_control=0.1,
            later_wall=0.5,
            later_control=0.2,
        )
    )
