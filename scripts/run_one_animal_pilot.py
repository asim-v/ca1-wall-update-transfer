"""Run the frozen center-boundary pilot on QLAK-CA1-51."""

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
from ca1_geometry.metrics import anisotropy_components, anisotropy_profile
from ca1_geometry.pilot import (
    estimate_session_metric,
    occupancy_balance_weights,
)


SEQUENCES = {
    "sequence_1": (0, 1, 10),
    "sequence_2": (10, 11, 20),
}
BLOCKED = (4,)
DISTANCES_CM = np.array([2.5, 7.5, 12.5])
DISTANCE_EDGES_CM = np.array([0.0, 5.0, 10.0, 15.0])
BANDWIDTHS_CM = (5.0, 7.5, 10.0)


def _finite_list(value: np.ndarray) -> list[float | None]:
    return [
        float(item) if np.isfinite(item) else None
        for item in np.asarray(value).ravel()
    ]


def _finite_scalar(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


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


def _minimum_valid(
    values: np.ndarray, valid: np.ndarray
) -> float | None:
    selected = np.asarray(values)[:, valid]
    if selected.size == 0:
        return None
    result = float(np.min(selected))
    return result if np.isfinite(result) else None


def _analyze_population(
    animal: Mat73Animal,
    cells_by_sequence: dict[str, np.ndarray],
    queries: Any,
    *,
    n_fold: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for sequence_name, sessions in SEQUENCES.items():
        cell = cells_by_sequence[sequence_name]
        position = [animal.position(session) for session in sessions]
        response = [
            animal.trace(session, cell) for session in sessions
        ]
        keep = [
            positions_on_accessible_support(item, BLOCKED)
            for item in position
        ]
        balanced = occupancy_balance_weights(position, keep)
        weight_modes = {
            "unweighted": [None, None, None],
            "occupancy_balanced": balanced,
        }
        sequence_output: dict[str, Any] = {
            "sessions_one_based": [item + 1 for item in sessions],
            "condition_environment": animal.environment(sessions[1]),
            "n_cells": int(cell.size),
            "weight_modes": {},
        }

        for weight_name, weights in weight_modes.items():
            bandwidth_output: dict[str, Any] = {}
            for bandwidth in BANDWIDTHS_CM:
                config = LocalMapConfig(
                    bandwidth=bandwidth,
                    min_effective_samples=40.0,
                    min_design_eigenratio=0.01,
                )
                estimates = [
                    estimate_session_metric(
                        position[index],
                        response[index],
                        queries.position,
                        common_blocked=BLOCKED,
                        config=config,
                        n_fold=n_fold,
                        sample_weight=weights[index],
                    )
                    for index in range(3)
                ]
                pre, condition, post = estimates
                reference = 0.5 * (pre.metric + post.metric)
                pooled_reference = 0.5 * (pre.pooled + post.pooled)
                valid = pre.valid & condition.valid & post.valid
                profile = anisotropy_profile(
                    condition.metric,
                    reference,
                    queries.normal,
                    queries.distance,
                    DISTANCE_EDGES_CM,
                    tangent=queries.tangent,
                    valid=valid,
                )
                pooled_scale_profile = anisotropy_profile(
                    condition.metric,
                    reference,
                    queries.normal,
                    queries.distance,
                    DISTANCE_EDGES_CM,
                    tangent=queries.tangent,
                    valid=valid,
                    denominator_metric=pooled_reference,
                )
                pooled_visualization_profile = anisotropy_profile(
                    condition.pooled,
                    pooled_reference,
                    queries.normal,
                    queries.distance,
                    DISTANCE_EDGES_CM,
                    tangent=queries.tangent,
                    valid=valid,
                    denominator_metric=pooled_reference,
                )
                null_valid = pre.valid & post.valid
                square_null = anisotropy_profile(
                    post.metric,
                    pre.metric,
                    queries.normal,
                    queries.distance,
                    DISTANCE_EDGES_CM,
                    tangent=queries.tangent,
                    valid=null_valid,
                )
                pooled_scale_square_null = anisotropy_profile(
                    post.metric,
                    pre.metric,
                    queries.normal,
                    queries.distance,
                    DISTANCE_EDGES_CM,
                    tangent=queries.tangent,
                    valid=null_valid,
                    denominator_metric=pooled_reference,
                )
                pooled_visualization_square_null = anisotropy_profile(
                    post.pooled,
                    pre.pooled,
                    queries.normal,
                    queries.distance,
                    DISTANCE_EDGES_CM,
                    tangent=queries.tangent,
                    valid=null_valid,
                    denominator_metric=pooled_reference,
                )
                component = anisotropy_components(
                    condition.metric,
                    reference,
                    queries.normal,
                    tangent=queries.tangent,
                )
                near_a = pooled_scale_profile.anisotropy[0]
                near_n = pooled_scale_profile.normal_magnification[0]
                null_a = pooled_scale_square_null.anisotropy[0]
                bandwidth_output[f"{bandwidth:g}_cm"] = {
                    "condition_vs_bracketing_squares": _profile_dict(profile),
                    "condition_vs_squares_pooled_reference_scale": (
                        _profile_dict(pooled_scale_profile)
                    ),
                    "pooled_metric_visualization_only": _profile_dict(
                        pooled_visualization_profile
                    ),
                    "square_to_square_null": _profile_dict(square_null),
                    "square_to_square_null_pooled_reference_scale": (
                        _profile_dict(pooled_scale_square_null)
                    ),
                    "pooled_square_null_visualization_only": _profile_dict(
                        pooled_visualization_square_null
                    ),
                    "valid_query_count": int(valid.sum()),
                    "valid_query_total": int(valid.size),
                    "near_sign_criterion": bool(
                        np.isfinite(near_a)
                        and np.isfinite(near_n)
                        and near_a > 0
                        and near_n > 0
                    ),
                    "near_null_is_smaller": bool(
                        np.isfinite(near_a)
                        and np.isfinite(null_a)
                        and abs(null_a) < abs(near_a)
                    ),
                    "session_split_pair_metric_reliability": {
                        "square_pre": _finite_scalar(
                            pre.split_pair_reliability
                        ),
                        "condition": _finite_scalar(
                            condition.split_pair_reliability
                        ),
                        "square_post": _finite_scalar(
                            post.split_pair_reliability
                        ),
                    },
                    "frames_per_fold_after_common_support": {
                        "square_pre": [
                            int(item) for item in pre.frames_per_fold
                        ],
                        "condition": [
                            int(item) for item in condition.frames_per_fold
                        ],
                        "square_post": [
                            int(item) for item in post.frames_per_fold
                        ],
                    },
                    "minimum_effective_n": {
                        "square_pre": _minimum_valid(
                            pre.effective_n, pre.valid
                        ),
                        "condition": _minimum_valid(
                            condition.effective_n, condition.valid
                        ),
                        "square_post": _minimum_valid(
                            post.effective_n, post.valid
                        ),
                    },
                    "minimum_design_eigenratio": {
                        "square_pre": _minimum_valid(
                            pre.design_eigenratio, pre.valid
                        ),
                        "condition": _minimum_valid(
                            condition.design_eigenratio, condition.valid
                        ),
                        "square_post": _minimum_valid(
                            post.design_eigenratio, post.valid
                        ),
                    },
                    "query_level": {
                        "distance_cm": _finite_list(queries.distance),
                        "segment_index": [
                            int(item) for item in queries.segment_index
                        ],
                        "delta_normal": _finite_list(
                            component["delta_normal"]
                        ),
                        "delta_tangent": _finite_list(
                            component["delta_tangent"]
                        ),
                        "trace_reference": _finite_list(
                            component["trace_reference"]
                        ),
                        "valid": [bool(item) for item in valid],
                    },
                }
            sequence_output["weight_modes"][weight_name] = bandwidth_output
        output[sequence_name] = sequence_output
    return output


def _plot_summary(result: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    colors = {"sequence_1": "#1f77b4", "sequence_2": "#d95f02"}
    for axis, population_name in zip(
        axes, ("sequence_common_cells", "stable_five_session_core"), strict=True
    ):
        population = result["populations"][population_name]
        for sequence_name, sequence in population.items():
            for bandwidth in BANDWIDTHS_CM:
                entry = sequence["weight_modes"]["unweighted"][
                    f"{bandwidth:g}_cm"
                ]
                profile = entry[
                    "condition_vs_squares_pooled_reference_scale"
                ]
                axis.plot(
                    profile["distance_cm"],
                    profile["anisotropy"],
                    color=colors[sequence_name],
                    marker="o",
                    alpha=0.35 + 0.05 * bandwidth,
                    label=(
                        f"{sequence_name.replace('_', ' ')}, h={bandwidth:g}"
                    ),
                )
        axis.axhline(0.0, color="0.5", linewidth=1)
        axis.set(
            xlabel="Normal distance from boundary (cm)",
            title=population_name.replace("_", " "),
        )
    axes[0].set_ylabel(
        "Normal - tangential metric change / square metric trace"
    )
    axes[1].legend(frameon=False, fontsize=7)
    figure.suptitle("QLAK-CA1-51 center-occlusion pilot (descriptive)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("animal_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--folds", type=int, choices=(2, 4), default=4)
    argument = parser.parse_args()

    segments = introduced_boundaries(BLOCKED)
    queries = segment_boundary_queries(
        segments,
        DISTANCES_CM,
        tangential_fractions=np.array([0.4, 0.6]),
    )
    with Mat73Animal(argument.animal_file) as animal:
        sequence_cells = {
            name: animal.common_registered_cells(*sessions)
            for name, sessions in SEQUENCES.items()
        }
        stable = animal.common_registered_cells(0, 1, 10, 11, 20)
        stable_cells = {name: stable for name in SEQUENCES}
        result = {
            "status": "descriptive_one_animal_pilot_not_inference",
            "animal": argument.animal_file.stem.replace(".complete", ""),
            "condition": {
                "environment": "o",
                "blocked_partitions": list(BLOCKED),
                "boundary_segments": len(segments),
                "query_distances_cm": _finite_list(DISTANCES_CM),
                "tangential_fractions": [0.4, 0.6],
            },
            "estimator": {
                "folds": argument.folds,
                "fold_type": "contiguous equal-duration temporal blocks",
                "response": "raw binary rise-extracted event indicator",
                "population_normalization": "1 / common-cell count",
                "reference": "mean of bracketing square-session metrics",
                "bandwidths_cm": list(BANDWIDTHS_CM),
                "kernel": "tricube",
                "minimum_effective_samples": 40.0,
                "minimum_design_eigenratio": 0.01,
                "common_support": "center partition removed in all sessions",
            },
            "populations": {
                "sequence_common_cells": _analyze_population(
                    animal,
                    sequence_cells,
                    queries,
                    n_fold=argument.folds,
                ),
                "stable_five_session_core": _analyze_population(
                    animal,
                    stable_cells,
                    queries,
                    n_fold=argument.folds,
                ),
            },
        }

    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    _plot_summary(result, argument.figure)
    compact = {}
    for population_name, population in result["populations"].items():
        compact[population_name] = {}
        for sequence_name, sequence in population.items():
            compact[population_name][sequence_name] = {}
            for bandwidth in BANDWIDTHS_CM:
                entry = sequence["weight_modes"]["unweighted"][
                    f"{bandwidth:g}_cm"
                ]
                profile = entry[
                    "condition_vs_squares_pooled_reference_scale"
                ]
                compact[population_name][sequence_name][
                    f"{bandwidth:g}_cm"
                ] = {
                    "n_cells": sequence["n_cells"],
                    "near_anisotropy": profile["anisotropy"][0],
                    "near_normal_magnification": (
                        profile["normal_magnification"][0]
                    ),
                    "near_square_null": (
                        entry[
                            "square_to_square_null_pooled_reference_scale"
                        ]["anisotropy"][0]
                    ),
                    "sign_criterion": entry["near_sign_criterion"],
                }
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
