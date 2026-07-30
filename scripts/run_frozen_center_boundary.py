"""Run a frozen longitudinal internal-boundary analysis on one animal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ca1_geometry.arena import (
    introduced_boundaries,
    positions_on_accessible_support,
    segment_boundary_queries,
)
from ca1_geometry.io import Mat73Animal
from ca1_geometry.local_linear import LocalMapConfig
from ca1_geometry.metrics import anisotropy_profile
from ca1_geometry.pilot import (
    estimate_session_metric,
    occupancy_balance_weights,
    query_balanced_block_folds,
    residual_directional_reliability,
)


DISTANCES_CM = np.array([2.5, 7.5, 12.5])
DISTANCE_EDGES_CM = np.array([0.0, 5.0, 10.0, 15.0])
BANDWIDTHS_CM = (7.5, 10.0)
PRIMARY_BANDWIDTH_CM = 10.0
N_FOLD = 4
BLOCK_FRAMES = 60 * 30
GUARD_FRAMES = 1 * 30


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _finite_list(value: np.ndarray) -> list[float | None]:
    return [_finite(float(item)) for item in np.asarray(value).ravel()]


def _profile_dict(profile: Any) -> dict[str, Any]:
    return {
        "distance_cm": _finite_list(profile.center),
        "anisotropy": _finite_list(profile.anisotropy),
        "normal_magnification": _finite_list(
            profile.normal_magnification
        ),
        "tangential_change": _finite_list(profile.tangential_change),
        "n_query": [int(item) for item in profile.n_query],
    }


def _reliability_dict(value: Any) -> dict[str, Any]:
    return {
        "normal": _finite(value.normal),
        "tangent": _finite(value.tangent),
        "contrast": _finite(value.contrast),
        "n_query": int(value.n_query),
    }


def _discover_sequences(
    animal: Mat73Animal,
    condition_offset: int | None,
    condition_name: str | None,
) -> tuple[list[tuple[int, int, int]], str, tuple[int, ...]]:
    if condition_name is not None:
        square_sessions = [
            session
            for session in range(animal.n_sessions)
            if animal.environment(session) == "square"
        ]
        sequence = []
        for square_pre, square_post in zip(
            square_sessions[:-1], square_sessions[1:]
        ):
            matches = [
                session
                for session in range(square_pre + 1, square_post)
                if animal.environment(session) == condition_name
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one {condition_name!r} session between "
                    f"squares {square_pre + 1} and {square_post + 1}; "
                    f"found {len(matches)}"
                )
            sequence.append((square_pre, matches[0], square_post))
        if not sequence:
            raise ValueError("no complete square-condition-square sequence found")
        blocked_values = {
            animal.blocked(sessions[1]) for sessions in sequence
        }
        if len(blocked_values) != 1:
            raise ValueError(
                f"inconsistent blocked partitions for {condition_name!r}"
            )
        return sequence, condition_name, blocked_values.pop()

    if condition_offset is None:
        condition_offset = 1
    if not 1 <= condition_offset <= 9:
        raise ValueError("condition_offset must lie in [1, 9]")
    sequence = []
    condition_environment: str | None = None
    condition_blocked: tuple[int, ...] | None = None
    for square_pre in range(0, animal.n_sessions - 10, 10):
        candidate = (
            square_pre,
            square_pre + condition_offset,
            square_pre + 10,
        )
        environment = animal.environment(candidate[1])
        blocked = animal.blocked(candidate[1])
        if condition_environment is None:
            condition_environment = environment
            condition_blocked = blocked
        if (
            animal.environment(candidate[0]) != "square"
            or animal.environment(candidate[2]) != "square"
            or environment != condition_environment
            or blocked != condition_blocked
        ):
            raise ValueError(f"unexpected sequence structure: {candidate}")
        sequence.append(candidate)
    if (
        not sequence
        or condition_environment is None
        or condition_blocked is None
    ):
        raise ValueError("no complete square-condition-square sequence found")
    return sequence, condition_environment, condition_blocked


def _minimum_effective_n(estimate: Any) -> float | None:
    selected = estimate.effective_n[:, estimate.valid]
    return _finite(float(np.min(selected))) if selected.size else None


def _analyze_sequence(
    animal: Mat73Animal,
    sessions: tuple[int, int, int],
    cells: np.ndarray,
    queries: Any,
    blocked: tuple[int, ...],
) -> dict[str, Any]:
    position = [animal.position(session) for session in sessions]
    response = [animal.trace(session, cells) for session in sessions]
    keep = [
        positions_on_accessible_support(value, blocked)
        for value in position
    ]
    occupancy_weight = occupancy_balance_weights(position, keep)
    fold_assignment = [
        query_balanced_block_folds(
            position[index],
            keep[index],
            queries.position,
            bandwidth=PRIMARY_BANDWIDTH_CM,
            n_fold=N_FOLD,
            block_frames=BLOCK_FRAMES,
            guard_frames=GUARD_FRAMES,
        )
        for index in range(3)
    ]

    result: dict[str, Any] = {
        "sessions_one_based": [session + 1 for session in sessions],
        "n_cells": int(cells.size),
        "mean_event_rate_hz": [
            float(value.mean() * 30.0) for value in response
        ],
        "weight_modes": {},
    }
    for weight_name, weights in {
        "occupancy_balanced": occupancy_weight,
        "unweighted": [None, None, None],
    }.items():
        bandwidth_result: dict[str, Any] = {}
        for bandwidth in BANDWIDTHS_CM:
            config = LocalMapConfig(
                bandwidth=bandwidth,
                min_effective_samples=40.0,
                min_design_eigenratio=0.01,
            )
            estimate = [
                estimate_session_metric(
                    position[index],
                    response[index],
                    queries.position,
                    common_blocked=blocked,
                    config=config,
                    n_fold=N_FOLD,
                    sample_weight=weights[index],
                    fold_assignment=fold_assignment[index],
                )
                for index in range(3)
            ]
            square_pre, condition, square_post = estimate
            reference = 0.5 * (
                square_pre.metric + square_post.metric
            )
            pooled_reference = 0.5 * (
                square_pre.pooled + square_post.pooled
            )
            valid = (
                square_pre.valid & condition.valid & square_post.valid
            )
            null_valid = square_pre.valid & square_post.valid

            profile = anisotropy_profile(
                condition.metric,
                reference,
                queries.normal,
                queries.distance,
                DISTANCE_EDGES_CM,
                tangent=queries.tangent,
                valid=valid,
                denominator_metric=pooled_reference,
            )
            square_null = anisotropy_profile(
                square_post.metric,
                square_pre.metric,
                queries.normal,
                queries.distance,
                DISTANCE_EDGES_CM,
                tangent=queries.tangent,
                valid=null_valid,
                denominator_metric=pooled_reference,
            )
            pooled_visualization = anisotropy_profile(
                condition.pooled,
                pooled_reference,
                queries.normal,
                queries.distance,
                DISTANCE_EDGES_CM,
                tangent=queries.tangent,
                valid=valid,
                denominator_metric=pooled_reference,
            )

            target_reliability = residual_directional_reliability(
                condition.jacobians,
                [square_pre.jacobians, square_post.jacobians],
                queries.normal,
                queries.tangent,
                valid=valid,
            )
            near = valid & (queries.distance < 5.0)
            near_reliability = residual_directional_reliability(
                condition.jacobians,
                [square_pre.jacobians, square_post.jacobians],
                queries.normal,
                queries.tangent,
                valid=near,
            )
            null_reliability = residual_directional_reliability(
                square_post.jacobians,
                [square_pre.jacobians],
                queries.normal,
                queries.tangent,
                valid=null_valid,
            )

            near_segments = np.unique(
                queries.segment_index[near]
            ).size
            segment_total = np.unique(queries.segment_index).size
            near_query_total = int(
                np.count_nonzero(queries.distance < 5.0)
            )
            minimum_near_queries = max(
                4, int(np.ceil(0.25 * near_query_total))
            )
            minimum_near_segments = max(
                3, int(np.ceil(0.50 * segment_total))
            )
            near_a = profile.anisotropy[0]
            near_n = profile.normal_magnification[0]
            near_null = square_null.anisotropy[0]
            contrast_reliability = target_reliability.contrast
            gate = {
                "near_query_support": bool(
                    near.sum() >= minimum_near_queries
                ),
                "near_segment_support": bool(
                    near_segments >= minimum_near_segments
                ),
                "positive_normal_magnification": bool(
                    np.isfinite(near_n) and near_n > 0
                ),
                "positive_normal_minus_tangent": bool(
                    np.isfinite(near_a) and near_a > 0
                ),
                "contrast_reliability_above_0_3": bool(
                    np.isfinite(contrast_reliability)
                    and contrast_reliability > 0.3
                ),
                "target_larger_than_square_null": bool(
                    np.isfinite(near_a)
                    and np.isfinite(near_null)
                    and abs(near_a) > abs(near_null)
                ),
            }
            gate["passes_all"] = all(gate.values())

            bandwidth_result[f"{bandwidth:g}_cm"] = {
                "target": _profile_dict(profile),
                "square_pseudo_wall_null": _profile_dict(square_null),
                "pooled_metric_visualization_only": _profile_dict(
                    pooled_visualization
                ),
                "target_residual_reliability": _reliability_dict(
                    target_reliability
                ),
                "near_target_residual_reliability": _reliability_dict(
                    near_reliability
                ),
                "square_change_reliability": _reliability_dict(
                    null_reliability
                ),
                "valid_queries": int(valid.sum()),
                "near_valid_queries": int(near.sum()),
                "near_valid_segments": int(near_segments),
                "minimum_near_queries": minimum_near_queries,
                "minimum_near_segments": minimum_near_segments,
                "frames_per_fold": {
                    "square_pre": [
                        int(item) for item in square_pre.frames_per_fold
                    ],
                    "condition": [
                        int(item) for item in condition.frames_per_fold
                    ],
                    "square_post": [
                        int(item) for item in square_post.frames_per_fold
                    ],
                },
                "minimum_effective_n": {
                    "square_pre": _minimum_effective_n(square_pre),
                    "condition": _minimum_effective_n(condition),
                    "square_post": _minimum_effective_n(square_post),
                },
                "gate": gate,
            }
        result["weight_modes"][weight_name] = bandwidth_result
    return result


def _save_figure(result: dict[str, Any], output: Path) -> None:
    sequence = result["sequences"]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sequence)))
    for color, (name, value) in zip(colors, sequence.items(), strict=True):
        primary = value["weight_modes"]["occupancy_balanced"]["10_cm"]
        target = primary["target"]
        null = primary["square_pseudo_wall_null"]
        axes[0].plot(
            target["distance_cm"],
            target["anisotropy"],
            marker="o",
            color=color,
            label=name.replace("_", " "),
        )
        axes[0].plot(
            null["distance_cm"],
            null["anisotropy"],
            linestyle=":",
            color=color,
            alpha=0.8,
        )
        axes[1].scatter(
            int(name.rsplit("_", 1)[1]),
            (
                np.nan
                if primary["target_residual_reliability"]["contrast"] is None
                else primary["target_residual_reliability"]["contrast"]
            ),
            color=color,
            s=45,
        )
    axes[0].axhline(0.0, color="0.5", linewidth=1)
    axes[0].set(
        xlabel="Normal distance from wall (cm)",
        ylabel="Crossvalidated contrast / pooled square scale",
        title="Solid: true wall; dotted: square pseudo-wall",
    )
    axes[0].legend(frameon=False)
    axes[1].axhline(0.3, color="0.5", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="Exposure sequence",
        ylabel="Fold-pair correlation",
        title="Boundary-residual contrast reliability",
        xticks=range(1, len(sequence) + 1),
        ylim=(-1.0, 1.0),
    )
    figure.suptitle(
        f"{result['animal']} frozen {result['condition_environment']} analysis"
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("animal_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    condition_selector = parser.add_mutually_exclusive_group()
    condition_selector.add_argument(
        "--condition-offset",
        type=int,
        help="zero-based offset within each 10-condition sequence (1-9)",
    )
    condition_selector.add_argument(
        "--environment",
        help="condition label to locate independently within each square block",
    )
    parser.add_argument(
        "--selection-provenance",
        default="single_animal_condition_test",
        help="short machine-readable label recorded in the output",
    )
    argument = parser.parse_args()

    with Mat73Animal(argument.animal_file) as animal:
        sequences, environment, blocked = _discover_sequences(
            animal, argument.condition_offset, argument.environment
        )
        queries = segment_boundary_queries(
            introduced_boundaries(blocked),
            DISTANCES_CM,
            tangential_fractions=np.array([0.4, 0.5, 0.6]),
        )
        stable_sessions = sorted(
            {session for sequence in sequences for session in sequence}
        )
        stable_cells = animal.common_registered_cells(*stable_sessions)
        result = {
            "status": "single_animal_condition_test_not_cohort_inference",
            "selection_provenance": argument.selection_provenance,
            "animal": argument.animal_file.stem.replace(".complete", ""),
            "condition_environment": environment,
            "condition_offset": argument.condition_offset,
            "blocked_partitions": list(blocked),
            "boundary_segment_count": len(introduced_boundaries(blocked)),
            "stable_sessions_one_based": [
                session + 1 for session in stable_sessions
            ],
            "stable_cell_count": int(stable_cells.size),
            "frozen_estimator": {
                "folds": N_FOLD,
                "fold_scheme": (
                    "query-design-balanced whole temporal blocks using "
                    "positions only"
                ),
                "fold_balance_bandwidth_cm": PRIMARY_BANDWIDTH_CM,
                "block_seconds": BLOCK_FRAMES / 30,
                "guard_seconds_each_edge": GUARD_FRAMES / 30,
                "primary_bandwidth_cm": PRIMARY_BANDWIDTH_CM,
                "sensitivity_bandwidth_cm": 7.5,
                "query_distances_cm": _finite_list(DISTANCES_CM),
                "tangential_fractions": [0.4, 0.5, 0.6],
                "response": "raw binary rise-extracted events",
                "common_support": (
                    "condition-blocked partitions removed in all sessions"
                ),
                "primary_weighting": "coarse occupancy balanced",
                "normalizer": "pooled bracketing-square tensor, scale only",
            },
            "sequences": {
                f"sequence_{index + 1}": _analyze_sequence(
                    animal,
                    sequence,
                    stable_cells,
                    queries,
                    blocked,
                )
                for index, sequence in enumerate(sequences)
            },
        }

    primary_gate = [
        value["weight_modes"]["occupancy_balanced"]["10_cm"]["gate"]
        for value in result["sequences"].values()
    ]
    result["primary_gate_sequences_passed"] = int(
        sum(item["passes_all"] for item in primary_gate)
    )
    result["primary_gate_sequences_total"] = len(primary_gate)
    result["all_primary_sequence_gates_pass"] = all(
        item["passes_all"] for item in primary_gate
    )

    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    _save_figure(result, argument.figure)
    compact = {
        name: {
            "near_anisotropy": value["weight_modes"][
                "occupancy_balanced"
            ]["10_cm"]["target"]["anisotropy"][0],
            "near_normal_magnification": value["weight_modes"][
                "occupancy_balanced"
            ]["10_cm"]["target"]["normal_magnification"][0],
            "near_square_null": value["weight_modes"][
                "occupancy_balanced"
            ]["10_cm"]["square_pseudo_wall_null"]["anisotropy"][0],
            "contrast_reliability": value["weight_modes"][
                "occupancy_balanced"
            ]["10_cm"]["target_residual_reliability"]["contrast"],
            "near_queries": value["weight_modes"][
                "occupancy_balanced"
            ]["10_cm"]["near_valid_queries"],
            "near_segments": value["weight_modes"][
                "occupancy_balanced"
            ]["10_cm"]["near_valid_segments"],
            "passes_gate": value["weight_modes"][
                "occupancy_balanced"
            ]["10_cm"]["gate"]["passes_all"],
        }
        for name, value in result["sequences"].items()
    }
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
