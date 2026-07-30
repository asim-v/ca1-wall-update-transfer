"""Focused tests for the compositional-wall identifiability audit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    path = ROOT / "scripts" / "audit_boundary_compositional_counterfactual.py"
    spec = importlib.util.spec_from_file_location(
        "boundary_compositional_counterfactual",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_script()


GEOMETRY = {
    "square": (),
    "o": (4,),
    "t": (3, 5, 6, 8),
    "u": (4, 5),
    "rectangle": (0, 3, 6),
    "+": (0, 2, 6, 8),
    "i": (3, 5),
    "l": (1, 2, 4, 5),
    "bit donut": (0, 4),
    "glenn": (0, 8),
}


def test_global_additive_wall_design_is_not_leave_one_shape_estimable() -> None:
    result = AUDIT.geometry_audit(GEOMETRY)
    oriented = result["oriented_wall_design"]
    physical = result["physical_boundary_design"]

    assert oriented["rank_without_intercept"] == 9
    assert oriented["rank_with_intercept"] == 10
    assert oriented["parameters_with_intercept"] == 25
    assert oriented["nullity_with_intercept"] == 15
    assert physical["rank_without_intercept"] == 9
    assert physical["rank_with_intercept"] == 10
    assert physical["parameters_with_intercept"] == 13
    assert physical["nullity_with_intercept"] == 3

    for design in (oriented, physical):
        assert not design[
            "all_deformed_targets_estimable_from_other_shapes"
        ]
        assert all(
            not item["target_row_in_training_row_span"]
            for item in design["leave_one_deformed_shape_out"].values()
        )
        assert all(
            item[
                "training_equivalent_models_can_change_target_arbitrarily"
            ]
            for item in design["leave_one_deformed_shape_out"].values()
        )


def test_no_global_one_wall_condition_and_only_two_local_factorials() -> None:
    result = AUDIT.geometry_audit(GEOMETRY)

    assert result["globally_isolated_one_boundary_environments"] == []
    assert result["minimum_nonzero_boundary_count"] == 3
    assert result["complete_local_factorial_count"] == 2
    assert result["distinct_ww_target_environments"] == ["bit donut"]

    queries = {
        item["query"]: item
        for item in result["complete_local_two_wall_factorials"]
    }
    assert set(queries) == {
        "0_to_3__4_to_3",
        "0_to_1__4_to_1",
    }
    assert all(len(item["overlap_bins"]) == 9 for item in queries.values())
    assert all(
        item["state_environments"]["WW"] == ["bit donut"]
        for item in queries.values()
    )


def test_interaction_free_predictor_can_recover_known_sum() -> None:
    wall_a = np.array([2.0, -1.0, 0.5, 3.0, 1.0, -2.0])
    wall_b = np.array([-0.5, 2.0, 1.0, -1.0, 2.5, 0.25])
    target = wall_a + wall_b

    score = AUDIT._score_predictors(
        additive=target,
        wall_a_only=wall_a,
        wall_b_only=wall_b,
        target=target,
    )

    assert score is not None
    assert np.isclose(
        score["predictors"]["interaction_free_additive"]["spearman_r"],
        1.0,
    )
    assert np.isclose(
        score["predictors"]["interaction_free_additive"][
            "normalized_rmse"
        ],
        0.0,
    )
    assert score["additive_minus_best_single_spearman_r"] > 0


def test_final_square_only_cycle_is_a_valid_post_test_baseline() -> None:
    environment = [
        name
        for _ in range(3)
        for name in GEOMETRY
    ] + ["square"]

    assert AUDIT._cycle_session(
        environment,
        cycle=3,
        name="square",
    ) == 30


def test_tracked_counterfactual_keeps_target_training_rates_out() -> None:
    path = (
        ROOT
        / "results"
        / "source_data"
        / "boundary_compositional_counterfactual.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["status"] == "stopped_underidentified_compositional_branch"
    assert sum(
        animal["attempted_records"] for animal in report["animals"]
    ) == 26
    assert sum(
        animal["eligible_records"] for animal in report["animals"]
    ) == 14
    for animal in report["animals"]:
        for record in animal["records"]:
            training_sessions = {
                session
                for sessions in record[
                    "training_state_sessions"
                ].values()
                for session in sessions
            }
            assert (
                record["target_training_session_excluded"]
                not in training_sessions
            )
            assert (
                record["target_training_session_excluded"]
                != record["training_square_session"]
            )
