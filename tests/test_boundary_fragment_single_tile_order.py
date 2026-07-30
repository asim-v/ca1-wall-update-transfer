"""Focused tests for the fixed-order single-tile falsification."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    path = (
        ROOT
        / "scripts"
        / "audit_boundary_fragment_single_tile_order.py"
    )
    spec = importlib.util.spec_from_file_location(
        "boundary_fragment_single_tile_order",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ORDER = _load_script()


def test_exact_lag_pairs_are_same_state_and_exclude_target() -> None:
    seam = ORDER.OrientedSeam(5, 2)
    blocked = [()] * 10
    # Offsets 1..9 produce states:
    # open, open, wall, wall, reverse, open, wall, wall, wall.
    blocked = list(blocked)
    blocked[1] = ()
    blocked[2] = ()
    blocked[3] = (5,)
    blocked[4] = (5,)
    blocked[5] = (2,)
    blocked[6] = ()
    blocked[7] = (5,)
    blocked[8] = (5,)
    blocked[9] = (5,)

    pairs = ORDER._candidate_order_pairs(
        cycle_start=0,
        target_offset=4,
        blocked=blocked,
        seam=seam,
        exact_lag=2,
    )

    assert all(pair.lag == 2 for pair in pairs)
    assert all(4 not in (pair.earlier, pair.later) for pair in pairs)
    assert all(
        pair.state in (ORDER.SeamState.OPEN, ORDER.SeamState.WALL)
        for pair in pairs
    )
    assert {(pair.earlier, pair.later) for pair in pairs} == {
        (7, 9),
    }


def test_broader_pairs_use_all_lags_without_crossing_state() -> None:
    seam = ORDER.OrientedSeam(5, 2)
    blocked = [()] * 10
    blocked = list(blocked)
    blocked[2] = (5,)
    blocked[4] = (5,)
    blocked[6] = (2,)  # reverse wall: deliberately ineligible

    pairs = ORDER._candidate_order_pairs(
        cycle_start=0,
        target_offset=9,
        blocked=blocked,
        seam=seam,
        exact_lag=None,
    )

    assert pairs
    assert all(pair.earlier < pair.later for pair in pairs)
    assert all(9 not in (pair.earlier, pair.later) for pair in pairs)
    assert all(6 not in (pair.earlier, pair.later) for pair in pairs)
    assert (2, 4) in {
        (pair.earlier, pair.later) for pair in pairs
    }
    assert not any(
        {pair.earlier, pair.later} == {1, 2} for pair in pairs
    )


def test_lag_standardization_averages_pairwise_slopes() -> None:
    first = np.array([2.0, 4.0])
    second = np.array([8.0, 16.0])
    value = ORDER._lag_standardized_order_vector(
        [first, second],
        [1, 4],
        actual_lag=2,
    )
    # Scaled edits are [4, 8] and [4, 8].
    np.testing.assert_allclose(value, [4.0, 8.0])


def test_required_sessions_are_one_deduplicated_union() -> None:
    pair = ORDER.OrderPair(
        earlier=3,
        later=5,
        state=ORDER.SeamState.OPEN,
    )
    required = ORDER._required_sessions(
        train_baseline=0,
        test_baseline=20,
        target_test=12,
        open_session=3,
        wall_session=5,
        pairs=(pair,),
    )
    assert required == (0, 3, 5, 12, 20)


def test_animal_summary_equal_weights_records() -> None:
    def record(actual: float, placebo: float) -> dict:
        modes = {}
        for mode in ORDER.validation.RATE_MODES:
            metrics = {key: 0.0 for key in ORDER.PRIMARY_METRICS}
            metrics["actual_edit_r"] = actual
            metrics["placebo_order_r"] = placebo
            modes[mode] = metrics
        return {
            "placebo_pairs": 2,
            "cells": 30,
            "bins": 8,
            "modes": modes,
        }

    summary = ORDER._animal_variant_summary(
        [record(1.0, -1.0), record(0.0, 1.0)]
    )
    for mode in ORDER.validation.RATE_MODES:
        assert np.isclose(
            summary["modes"][mode]["actual_edit_r"]["record_mean"],
            0.5,
        )
        assert np.isclose(
            summary["modes"][mode]["placebo_order_r"]["record_mean"],
            0.0,
        )


def test_checked_in_artifact_has_one_canonical_schema() -> None:
    path = (
        ROOT
        / "results"
        / "source_data"
        / "boundary_fragment_single_tile_order_placebo.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["status"] == (
        "post_outcome_fixed_order_falsification_and_sensitivity"
    )
    assert set(report["cohort"]) == set(ORDER.VARIANTS)
    assert report["order_census"]["animals"] == 7
    assert report["order_census"]["animals_u_after_o"] == 7
    assert report["cohort"]["exact_lag"]["analyzed_records"] == 24
    assert report["cohort"]["broader_all_lags"]["analyzed_records"] == 46

    for animal in report["animals"]:
        lag = animal["order_census"]["u_minus_o_lag"]
        assert animal["order_census"]["u_after_o"]
        for variant in ORDER.VARIANTS:
            for record in animal["variants"][variant]["records"]:
                assert record[
                    "identical_cells_and_support_within_query"
                ]
                withheld = record["withheld_training_session"]
                for pair in record["pairs"]:
                    assert pair["earlier_session"] < pair["later_session"]
                    assert withheld not in (
                        pair["earlier_session"],
                        pair["later_session"],
                    )
                    assert pair["seam_state_both_endpoints"] in {
                        "open",
                        "wall",
                    }
                    if variant == "exact_lag":
                        assert pair["lag_in_geometry_sequence"] == lag
