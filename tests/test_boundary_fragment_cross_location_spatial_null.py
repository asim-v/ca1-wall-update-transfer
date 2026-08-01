"""Tests for the animal-level spatial-null calibration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    path = (
        ROOT
        / "scripts"
        / "summarize_boundary_fragment_cross_location_spatial_null.py"
    )
    spec = importlib.util.spec_from_file_location("spatial_null", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SPATIAL_NULL = _load_script()


def test_exact_sign_flip_enumerates_all_assignments() -> None:
    result = SPATIAL_NULL.exact_sign_flip({"b": 2.0, "a": 1.0})

    assert result["animals"] == 2
    assert result["sign_assignments"] == 4
    assert result["positive_animals"] == 2
    assert np.isclose(result["observed_animal_mean"], 1.5)
    assert np.isclose(result["one_sided_tail_fraction"], 0.25)
    assert np.isclose(result["two_sided_tail_fraction"], 0.5)
    assert result["leave_one_animal_out_means"] == {"a": 2.0, "b": 1.0}


def test_tracked_spatial_null_records_strict_and_heterogeneous_controls() -> None:
    report = json.loads(
        (
            ROOT
            / "results"
            / "source_data"
            / "boundary_fragment_cross_location_spatial_null.json"
        ).read_text(encoding="utf-8")
    )
    primary = report["modes"]["global_rate_demeaned"]

    pooled = primary["exact_midpoint_wrong_orientation_null"]
    assert pooled["animals"] == 7
    assert pooled["positive_animals"] == 7
    assert np.isclose(pooled["observed_animal_mean"], 0.10095348102264133)
    assert np.isclose(pooled["one_sided_tail_fraction"], 1 / 128)

    tangential = primary["tangential_five_cm_strip_lag_null"]
    assert tangential["animals"] == 6
    assert tangential["positive_animals"] == 3

    mirror = primary["same_session_reflected_open_null"]
    assert mirror["animals"] == 5
    assert mirror["positive_animals"] == 5
    assert np.isclose(mirror["one_sided_tail_fraction"], 1 / 32)
