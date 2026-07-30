from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_boundary_fragment_experience_raw_split",
    ROOT / "scripts" / "run_boundary_fragment_experience_raw_split.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_split_correlation_uses_four_crossed_assignments():
    value = {}
    for half in (0, 1):
        value[0, half] = np.zeros(10, dtype=float)
        value[1, half] = np.arange(10, dtype=float)
        value[2, half] = 2.0 * np.arange(10, dtype=float)
        value[3, half] = np.zeros(10, dtype=float)
    mean, parts = MODULE._split_correlation(
        value,
        pre_square=0,
        first_target=1,
        second_target=2,
        post_square=3,
    )
    assert len(parts) == 4
    assert mean == pytest.approx(1.0)


def test_common_half_support_requires_every_half():
    occupancy = {
        session: [
            np.ones((2, 2), dtype=float),
            np.ones((2, 2), dtype=float),
        ]
        for session in range(2)
    }
    occupancy[1][1][0, 1] = 0.1
    assert MODULE._common_half_support(
        (0, 1),
        occupancy,
        ((0, 0), (0, 1)),
        minimum_seconds=0.25,
    ) == ((0, 0),)
