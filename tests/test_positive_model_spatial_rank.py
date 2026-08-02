"""Tests for the frozen fully enumerated empirical spatial baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from ca1_geometry.positive_models import (
    exact_sign_flip,
    hierarchical_animal_summary,
    midrank_percentile,
    query_empirical_rank,
)


def _load_script() -> ModuleType:
    path = ROOT / "scripts" / "run_boundary_positive_spatial_rank.py"
    spec = importlib.util.spec_from_file_location("positive_rank", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RANK_SCRIPT = _load_script()


def _record(
    relation: str,
    score: float,
    *,
    distance: float = 25.0,
    translation: str = "tangential",
) -> dict[str, object]:
    return {
        "orientation_relation": relation,
        "midpoint_distance_cm": distance,
        "translation_axis": translation,
        "metrics": {
            "global_rate_demeaned": {
                "source_effect_r_to_target_residual": score,
            }
        },
    }


def test_midrank_percentile_handles_complete_and_partial_ties() -> None:
    assert np.isclose(midrank_percentile(2.0, [1.0, 2.0, 2.0, 3.0]), 0.5)
    assert np.isclose(midrank_percentile(5.0, [1.0, 2.0]), 1.0)
    assert np.isclose(midrank_percentile(0.0, [1.0, 2.0]), 0.0)


def test_query_rank_uses_every_exact_distance_candidate() -> None:
    result = query_empirical_rank(
        [
            _record("same_signed_normal", 0.8),
            _record("same_signed_normal", 0.4),
            _record("opposite_normal", 0.2),
            _record("orthogonal_axis", -0.2),
            _record("opposite_normal", 0.99, distance=50.0),
        ],
        mode="global_rate_demeaned",
    )

    assert result is not None
    assert result["candidate_sources"] == 4
    assert result["correct_sources"] == 2
    assert result["alternative_sources"] == 2
    assert np.isclose(result["correct_relation_mean"], 0.6)
    assert np.isclose(result["correct_minus_mean_alternative"], 0.6)
    assert np.isclose(result["correct_minus_best_alternative"], 0.4)
    assert np.isclose(result["correct_percentile_rank"], 0.75)
    assert result["correct_beats_every_alternative"]
    assert np.isclose(result["centered_correct_win"], 0.5)


def test_tangential_tier_does_not_relax_to_normal_sources() -> None:
    result = query_empirical_rank(
        [
            _record("same_signed_normal", 0.4),
            _record("opposite_normal", 0.1),
            _record(
                "same_signed_normal", 0.9, translation="normal"
            ),
            _record("opposite_normal", 1.0, translation="normal"),
        ],
        mode="global_rate_demeaned",
        tier="tier2_tangential",
    )

    assert result is not None
    assert result["candidate_sources"] == 2
    assert np.isclose(result["correct_minus_mean_alternative"], 0.3)


def test_hierarchical_aggregation_weights_exposure_pairs_equally() -> None:
    queries = [
        {
            "training_exposure": 1,
            "test_exposure": 2,
            "score": 1.0,
        },
        {
            "training_exposure": 1,
            "test_exposure": 2,
            "score": 1.0,
        },
        {
            "training_exposure": 2,
            "test_exposure": 3,
            "score": -1.0,
        },
    ]
    result = hierarchical_animal_summary(
        queries,
        metric_names=("score",),
    )

    assert result["eligible_queries"] == 3
    assert result["eligible_exposure_pairs"] == 2
    assert np.isclose(
        result["metrics"]["score"]["animal_value"],
        0.0,
    )


def test_exact_sign_flip_uses_animals_not_queries() -> None:
    result = exact_sign_flip({"mouse_b": 2.0, "mouse_a": 1.0})

    assert result["animals"] == 2
    assert result["sign_assignments"] == 4
    assert result["positive_animals"] == 2
    assert np.isclose(result["observed_animal_mean"], 1.5)
    assert np.isclose(result["one_sided_tail_fraction"], 0.25)
    assert np.isclose(result["two_sided_tail_fraction"], 0.5)


def test_translation_axis_is_geometry_only() -> None:
    target = RANK_SCRIPT.OrientedSeam(0, 1)
    tangential = RANK_SCRIPT.OrientedSeam(3, 4)
    normal = RANK_SCRIPT.OrientedSeam(1, 2)

    assert RANK_SCRIPT.translation_axis(target, tangential) == "tangential"
    assert RANK_SCRIPT.translation_axis(target, normal) == "normal"
