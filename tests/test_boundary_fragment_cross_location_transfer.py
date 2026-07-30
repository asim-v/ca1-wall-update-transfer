"""Focused tests for strict cross-location wall-profile transfer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    path = (
        ROOT
        / "scripts"
        / "run_boundary_fragment_cross_location_transfer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "boundary_fragment_cross_location_transfer",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRANSFER = _load_script()


def test_orientation_relation_uses_signed_normal() -> None:
    target = TRANSFER.OrientedSeam(0, 1)
    assert (
        TRANSFER._orientation_relation(
            target,
            TRANSFER.OrientedSeam(3, 4),
        )
        == "same_signed_normal"
    )
    assert (
        TRANSFER._orientation_relation(
            target,
            TRANSFER.OrientedSeam(4, 3),
        )
        == "opposite_normal"
    )
    assert (
        TRANSFER._orientation_relation(
            target,
            TRANSFER.OrientedSeam(0, 3),
        )
        == "orthogonal_axis"
    )


def test_template_sessions_exclude_target_geometry() -> None:
    target = TRANSFER.OrientedSeam(0, 1)
    source = TRANSFER.OrientedSeam(3, 4)
    blocked = [()] * 10
    blocked[1] = (0,)
    blocked[2] = (3,)
    blocked[3] = ()
    blocked[4] = (0, 3)
    blocked[5] = ()
    blocked[6] = (0,)
    blocked[7] = (3,)
    blocked[8] = ()
    blocked[9] = (0, 3)
    result = TRANSFER._template_sessions(
        training_start=0,
        target_offset=4,
        target_seam=target,
        source_seam=source,
        blocked=blocked,
    )

    assert result is not None
    assert 4 not in {session for values in result for session in values}
    assert result.target_wall == (1, 6, 9)
    assert result.source_wall == (2, 7, 9)
    assert result.target_open == (2, 3, 5, 7, 8)
    assert result.source_open == (1, 3, 5, 6, 8)


def test_query_cell_mask_is_shared_and_excludes_target_training() -> None:
    registered = {
        session: np.ones(5, dtype=bool)
        for session in range(31)
    }
    registered[12][0] = False
    registered[30][1] = False
    registered[24][2] = False
    # Session 14 is the withheld target-training geometry and must not affect
    # the common mask.
    registered[14][3] = False
    cells = TRANSFER._query_common_cells(
        training_start=10,
        test_start=20,
        target_offset=4,
        registered=registered,
    )

    np.testing.assert_array_equal(cells, [3, 4])


def test_place_adjusted_sensitivity_recovers_shared_cell_order() -> None:
    source_open = np.array([5.0, 1.0, 8.0, 3.0, 7.0])
    target_open = np.array([2.0, 9.0, 4.0, 8.0, 1.0])
    wall_effect = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    metrics = TRANSFER._mode_metrics(
        source_wall=source_open + wall_effect,
        source_open=source_open,
        target_wall=target_open + 3.0 * wall_effect,
        target_open=target_open,
        exact_wall=target_open + 2.0 * wall_effect,
    )

    assert metrics is not None
    assert np.isclose(
        metrics["shared_training_place_adjusted_effect_r_sensitivity"],
        1.0,
    )
    assert np.isclose(
        metrics[
            "shared_training_exact_place_adjusted_effect_r_sensitivity"
        ],
        1.0,
    )


def test_query_primary_uses_one_grid_step_same_direction_sources() -> None:
    def record(
        relation: str,
        distance: float,
        effect: float,
    ) -> dict[str, object]:
        metrics = {
            "source_effect_r_to_target_residual": effect,
            "exact_effect_r_to_target_residual": 0.8,
            "shared_training_place_adjusted_effect_r_sensitivity": effect,
            (
                "shared_training_exact_place_adjusted_effect_r_"
                "sensitivity"
            ): 0.8,
            "source_wall_r_to_target_residual": 0.5,
            "source_open_r_to_target_residual": 0.1,
            "target_exact_open_r_to_target_residual": 0.0,
            "source_effect_r_to_target_open_profile": 0.0,
            "source_open_r_to_target_open_profile": 0.0,
            "source_wall_minus_open_r_to_target_residual": 0.4,
            "source_wall_minus_target_exact_open_r": 0.5,
            (
                "source_effect_minus_exact_effect_r_to_target_residual"
            ): effect - 0.8,
            "source_effect_specificity_over_target_open_r": effect,
        }
        return {
            "orientation_relation": relation,
            "midpoint_distance_cm": distance,
            "cells": 30,
            "target_bins": 10,
            "source_bins": 10,
            "metrics": {
                mode: metrics.copy()
                for mode in TRANSFER.RATE_MODES
            },
        }

    summary = TRANSFER._query_summary(
        [
            record("same_signed_normal", 25.0, 0.6),
            record("same_signed_normal", 25.0, 0.2),
            record("same_signed_normal", 50.0, -0.9),
            record("opposite_normal", 25.0, 0.1),
            record("orthogonal_axis", 25.0, -0.1),
        ]
    )

    assert summary is not None
    for mode in TRANSFER.RATE_MODES:
        contrasts = summary["modes"][mode]["contrasts"]
        assert np.isclose(
            contrasts[
                "one_grid_step_same_direction_source_effect_r_to_"
                "target_residual"
            ],
            0.4,
        )
        assert np.isclose(
            contrasts[
                "one_grid_step_same_direction_minus_exact_distance_"
                "nonsame_source_effect_r"
            ],
            0.4,
        )
