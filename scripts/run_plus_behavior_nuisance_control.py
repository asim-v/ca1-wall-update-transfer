"""Run the post-outcome speed/heading-adjusted ``+``-boundary control.

The frozen animal files, cells, square--plus--square sessions, query grid,
whole-block folds, guards, occupancy weights, bandwidths, and common-support
operator are reused exactly. Only the local regression is extended: speed and
the first two heading harmonics are partialled out jointly with the intercept
before the spatial dx/dy slopes are estimated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ca1_geometry.arena import (
    introduced_boundaries,
    positions_on_accessible_support,
    segment_boundary_queries,
)
from ca1_geometry.behavior_control import (
    KINEMATIC_COVARIATE_NAMES,
    estimate_behavior_adjusted_session_metric,
    kinematic_covariates,
)
from ca1_geometry.io import Mat73Animal
from ca1_geometry.local_linear import LocalMapConfig
from ca1_geometry.metrics import anisotropy_profile
from ca1_geometry.pilot import (
    occupancy_balance_weights,
    query_balanced_block_folds,
    residual_directional_reliability,
)


DISTANCES_CM = np.array([2.5, 7.5, 12.5])
DISTANCE_EDGES_CM = np.array([0.0, 5.0, 10.0, 15.0])
TANGENTIAL_FRACTIONS = np.array([0.4, 0.5, 0.6])
BANDWIDTHS_CM = (7.5, 10.0)
PRIMARY_BANDWIDTH_CM = 10.0
N_FOLD = 4
BLOCK_FRAMES = 60 * 30
GUARD_FRAMES = 1 * 30
FRAME_RATE_HZ = 30.0
VELOCITY_HALF_WINDOW_FRAMES = 5


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _minimum_at_valid(
    value: np.ndarray, valid: np.ndarray
) -> float | None:
    selected = value[:, valid]
    return _finite(float(np.min(selected))) if selected.size else None


def _rank_range_at_valid(
    value: np.ndarray, valid: np.ndarray
) -> dict[str, int] | None:
    selected = value[:, valid]
    if not selected.size:
        return None
    return {
        "minimum": int(np.min(selected)),
        "maximum": int(np.max(selected)),
    }


def _kinematic_summary(speed: np.ndarray) -> dict[str, float]:
    return {
        "mean_speed_cm_s": float(np.mean(speed)),
        "median_speed_cm_s": float(np.median(speed)),
        "q05_speed_cm_s": float(np.quantile(speed, 0.05)),
        "q95_speed_cm_s": float(np.quantile(speed, 0.95)),
        "maximum_speed_cm_s": float(np.max(speed)),
    }


def _source_sequence_result(
    source: dict[str, Any], sequence_name: str, bandwidth: float
) -> dict[str, Any]:
    return source["sequences"][sequence_name]["weight_modes"][
        "occupancy_balanced"
    ][f"{bandwidth:g}_cm"]


def _analyze_animal(
    animal_path: Path,
    source_path: Path,
) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    animal_name = animal_path.stem.replace(".complete", "")
    if source["animal"] != animal_name:
        raise ValueError(
            f"animal mismatch: {animal_name!r} versus {source['animal']!r}"
        )
    if source["condition_environment"] != "+":
        raise ValueError(f"{source_path} is not a plus-condition result")

    blocked = tuple(int(item) for item in source["blocked_partitions"])
    queries = segment_boundary_queries(
        introduced_boundaries(blocked),
        DISTANCES_CM,
        tangential_fractions=TANGENTIAL_FRACTIONS,
    )
    stable_sessions = [
        int(value) - 1 for value in source["stable_sessions_one_based"]
    ]
    sequence_session = {
        name: tuple(int(value) - 1 for value in sequence["sessions_one_based"])
        for name, sequence in source["sequences"].items()
    }
    unique_session = sorted(
        {session for triple in sequence_session.values() for session in triple}
    )

    with Mat73Animal(animal_path) as animal:
        stable_cells = animal.common_registered_cells(*stable_sessions)
        if stable_cells.size != int(source["stable_cell_count"]):
            raise ValueError(
                f"{animal_name}: stable-cell count differs from frozen source"
            )
        position = {
            session: animal.position(session) for session in unique_session
        }
        response = {
            session: animal.trace(session, stable_cells)
            for session in unique_session
        }
        kinematic = {
            session: kinematic_covariates(
                position[session],
                frame_rate_hz=FRAME_RATE_HZ,
                half_window_frames=VELOCITY_HALF_WINDOW_FRAMES,
            )
            for session in unique_session
        }
        keep = {
            session: positions_on_accessible_support(
                position[session], blocked
            )
            for session in unique_session
        }
        fold = {
            session: query_balanced_block_folds(
                position[session],
                keep[session],
                queries.position,
                bandwidth=PRIMARY_BANDWIDTH_CM,
                n_fold=N_FOLD,
                block_frames=BLOCK_FRAMES,
                guard_frames=GUARD_FRAMES,
            )
            for session in unique_session
        }

        sequence_result: dict[str, Any] = {}
        for sequence_name, sessions in sequence_session.items():
            positions = [position[session] for session in sessions]
            keeps = [keep[session] for session in sessions]
            weights = occupancy_balance_weights(positions, keeps)
            by_bandwidth: dict[str, Any] = {}
            for bandwidth in BANDWIDTHS_CM:
                config = LocalMapConfig(
                    bandwidth=bandwidth,
                    min_effective_samples=40.0,
                    min_design_eigenratio=0.01,
                )
                estimate = [
                    estimate_behavior_adjusted_session_metric(
                        position[session],
                        response[session],
                        kinematic[session].design,
                        queries.position,
                        common_blocked=blocked,
                        config=config,
                        n_fold=N_FOLD,
                        sample_weight=weights[index],
                        fold_assignment=fold[session],
                    )
                    for index, session in enumerate(sessions)
                ]
                square_pre, condition, square_post = estimate
                reference = 0.5 * (
                    square_pre.metric + square_post.metric
                )
                pooled_reference = 0.5 * (
                    square_pre.pooled + square_post.pooled
                )
                valid = (
                    square_pre.valid
                    & condition.valid
                    & square_post.valid
                )
                null_valid = square_pre.valid & square_post.valid
                target = anisotropy_profile(
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
                reliability = residual_directional_reliability(
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
                source_result = _source_sequence_result(
                    source, sequence_name, bandwidth
                )
                by_bandwidth[f"{bandwidth:g}_cm"] = {
                    "target": _profile_dict(target),
                    "square_pseudo_wall_null": _profile_dict(square_null),
                    "target_residual_reliability": _reliability_dict(
                        reliability
                    ),
                    "near_target_residual_reliability": _reliability_dict(
                        near_reliability
                    ),
                    "valid_queries": int(valid.sum()),
                    "near_valid_queries": int(near.sum()),
                    "near_valid_segments": int(
                        np.unique(queries.segment_index[near]).size
                    ),
                    "frozen_unadjusted_comparator": {
                        "near_anisotropy": source_result["target"][
                            "anisotropy"
                        ][0],
                        "near_normal_magnification": source_result["target"][
                            "normal_magnification"
                        ][0],
                        "near_tangential_change": source_result["target"][
                            "tangential_change"
                        ][0],
                        "near_square_null": source_result[
                            "square_pseudo_wall_null"
                        ]["anisotropy"][0],
                        "valid_queries": int(
                            source_result["valid_queries"]
                        ),
                        "near_valid_queries": int(
                            source_result["near_valid_queries"]
                        ),
                    },
                    "minimum_original_design_eigenratio": {
                        "square_pre": _minimum_at_valid(
                            square_pre.original_design_eigenratio,
                            square_pre.valid,
                        ),
                        "condition": _minimum_at_valid(
                            condition.original_design_eigenratio,
                            condition.valid,
                        ),
                        "square_post": _minimum_at_valid(
                            square_post.original_design_eigenratio,
                            square_post.valid,
                        ),
                    },
                    "minimum_adjusted_spatial_eigenratio": {
                        "square_pre": _minimum_at_valid(
                            square_pre.adjusted_spatial_eigenratio,
                            square_pre.valid,
                        ),
                        "condition": _minimum_at_valid(
                            condition.adjusted_spatial_eigenratio,
                            condition.valid,
                        ),
                        "square_post": _minimum_at_valid(
                            square_post.adjusted_spatial_eigenratio,
                            square_post.valid,
                        ),
                    },
                    "nuisance_design_rank_including_intercept": {
                        "square_pre": _rank_range_at_valid(
                            square_pre.nuisance_rank,
                            square_pre.valid,
                        ),
                        "condition": _rank_range_at_valid(
                            condition.nuisance_rank,
                            condition.valid,
                        ),
                        "square_post": _rank_range_at_valid(
                            square_post.nuisance_rank,
                            square_post.valid,
                        ),
                    },
                    "frames_per_fold": {
                        label: [
                            int(item) for item in value.frames_per_fold
                        ]
                        for label, value in zip(
                            ("square_pre", "condition", "square_post"),
                            estimate,
                            strict=True,
                        )
                    },
                }
            sequence_result[sequence_name] = {
                "sessions_one_based": [
                    session + 1 for session in sessions
                ],
                "kinematics": {
                    label: _kinematic_summary(
                        kinematic[session].speed_cm_s
                    )
                    for label, session in zip(
                        ("square_pre", "condition", "square_post"),
                        sessions,
                        strict=True,
                    )
                },
                "bandwidths": by_bandwidth,
            }

    adjusted_primary = [
        value["bandwidths"]["10_cm"]["target"]["anisotropy"][0]
        for value in sequence_result.values()
    ]
    unadjusted_primary = [
        value["bandwidths"]["10_cm"]["frozen_unadjusted_comparator"][
            "near_anisotropy"
        ]
        for value in sequence_result.values()
    ]
    return {
        "animal": animal_name,
        "source_frozen_result": str(source_path),
        "source_frozen_result_sha256": _sha256(source_path),
        "stable_cell_count": int(source["stable_cell_count"]),
        "stable_sessions_one_based": source["stable_sessions_one_based"],
        "blocked_partitions": list(blocked),
        "sequences": sequence_result,
        "animal_level_primary_mean": {
            "unadjusted_near_anisotropy": float(
                np.mean(unadjusted_primary)
            ),
            "behavior_adjusted_near_anisotropy": float(
                np.mean(adjusted_primary)
            ),
            "adjusted_minus_unadjusted": float(
                np.mean(adjusted_primary) - np.mean(unadjusted_primary)
            ),
        },
    }


def _save_figure(result: dict[str, Any], output: Path) -> None:
    animals = list(result["animals"])
    unadjusted = np.asarray(
        [
            result["animals"][animal]["animal_level_primary_mean"][
                "unadjusted_near_anisotropy"
            ]
            for animal in animals
        ],
        dtype=np.float64,
    )
    adjusted = np.asarray(
        [
            result["animals"][animal]["animal_level_primary_mean"][
                "behavior_adjusted_near_anisotropy"
            ]
            for animal in animals
        ],
        dtype=np.float64,
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    color = plt.cm.tab10(np.arange(len(animals)))
    for index, animal in enumerate(animals):
        axes[0].plot(
            [0, 1],
            [unadjusted[index], adjusted[index]],
            marker="o",
            color=color[index],
            label=animal,
        )
        sequence = result["animals"][animal]["sequences"].values()
        axes[1].plot(
            [1, 2, 3],
            [
                value["bandwidths"]["10_cm"]["target"]["anisotropy"][0]
                for value in sequence
            ],
            marker="o",
            color=color[index],
            label=animal,
        )
    for axis in axes:
        axis.axhline(0.0, color="0.5", linewidth=1)
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Frozen\nunadjusted", "Speed/heading\nadjusted"],
        ylabel="Mean near-wall normal − tangent contrast",
        title="Animal means across 3 exposures",
    )
    axes[1].set(
        xticks=[1, 2, 3],
        xlabel="Exposure sequence",
        ylabel="Adjusted near-wall contrast",
        title="Exploratory adjusted trajectories",
    )
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Post-outcome behavior nuisance control")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--animal-files", type=Path, nargs="+", required=True
    )
    parser.add_argument(
        "--source-results", type=Path, nargs="+", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    argument = parser.parse_args()
    if len(argument.animal_files) != len(argument.source_results):
        raise ValueError(
            "--animal-files and --source-results must have equal length"
        )

    animal_result = {
        path.stem.replace(".complete", ""): _analyze_animal(path, source)
        for path, source in zip(
            argument.animal_files,
            argument.source_results,
            strict=True,
        )
    }
    animal_unadjusted = np.asarray(
        [
            value["animal_level_primary_mean"][
                "unadjusted_near_anisotropy"
            ]
            for value in animal_result.values()
        ],
        dtype=np.float64,
    )
    animal_adjusted = np.asarray(
        [
            value["animal_level_primary_mean"][
                "behavior_adjusted_near_anisotropy"
            ]
            for value in animal_result.values()
        ],
        dtype=np.float64,
    )
    primary_sequence = [
        sequence["bandwidths"]["10_cm"]
        for animal in animal_result.values()
        for sequence in animal["sequences"].values()
    ]
    result = {
        "status": (
            "explicitly_exploratory_post_outcome_behavior_nuisance_control"
        ),
        "interpretation_limit": (
            "Behavior covariates can remain strongly position-correlated; "
            "this conditional association does not identify a causal neural "
            "effect of boundaries independently of behavior."
        ),
        "limitations": [
            "This control was specified and run after inspecting the frozen "
            "plus-condition outcomes.",
            "Speed and allocentric heading can remain strongly correlated "
            "with position even after local residualization.",
            "The nuisance model is additive and includes only speed and the "
            "first two heading harmonics; it omits interactions, acceleration, "
            "turning, and unmeasured behavior.",
            "Velocity is derived from position alone; tracking-confidence "
            "values are unavailable and speeds are not outcome-adaptively "
            "trimmed or winsorized.",
        ],
        "frozen_elements": [
            "eligible animals",
            "stable registered cells",
            "square-plus-square sessions",
            "boundary queries and normal/tangent directions",
            "condition-common support",
            "query-balanced whole temporal folds and one-second guards",
            "coarse occupancy weights",
            "10 cm primary and 7.5 cm sensitivity bandwidths",
            "cross-fold metric and bracketing-square reference",
        ],
        "changed_element": (
            "local spatial dx/dy slopes are estimated jointly after "
            "partialling out the behavior nuisance design"
        ),
        "kinematic_definition": {
            "sampling_rate_hz": FRAME_RATE_HZ,
            "velocity": (
                "position[t+5]-position[t-5] over 10/30 s; available "
                "asymmetric interval for the first/last five frames"
            ),
            "velocity_window_seconds": (
                2 * VELOCITY_HALF_WINDOW_FRAMES / FRAME_RATE_HZ
            ),
            "nuisance_columns": list(KINEMATIC_COVARIATE_NAMES),
            "heading_at_exact_zero_speed": (
                "all heading harmonic columns set to zero"
            ),
            "standardization": (
                "each column centered and scaled over the complete session; "
                "numerically constant columns set to zero"
            ),
        },
        "regression": {
            "method": (
                "weighted Frisch-Waugh-Lovell residualization of dx/dy and "
                "neural responses against intercept plus nuisance columns"
            ),
            "kernel": "same frozen tricube spatial kernel",
            "adjusted_spatial_eigenratio_threshold": 0.01,
            "ridge_relative": 1e-8,
        },
        "cohort_descriptive_summary": {
            "n_animals": len(animal_result),
            "n_repeated_exposures": len(primary_sequence),
            "animal_mean_unadjusted_near_anisotropy": (
                animal_unadjusted.tolist()
            ),
            "animal_mean_behavior_adjusted_near_anisotropy": (
                animal_adjusted.tolist()
            ),
            "animal_mean_adjusted_minus_unadjusted": (
                animal_adjusted - animal_unadjusted
            ).tolist(),
            "mean_of_animal_means_unadjusted": float(
                np.mean(animal_unadjusted)
            ),
            "mean_of_animal_means_behavior_adjusted": float(
                np.mean(animal_adjusted)
            ),
            "median_of_animal_means_unadjusted": float(
                np.median(animal_unadjusted)
            ),
            "median_of_animal_means_behavior_adjusted": float(
                np.median(animal_adjusted)
            ),
            "positive_adjusted_near_anisotropy_sequences": int(
                sum(
                    value["target"]["anisotropy"][0] > 0
                    for value in primary_sequence
                )
            ),
            "positive_adjusted_near_normal_magnification_sequences": int(
                sum(
                    value["target"]["normal_magnification"][0] > 0
                    for value in primary_sequence
                )
            ),
            "sequences_with_unchanged_valid_query_count": int(
                sum(
                    value["valid_queries"]
                    == value["frozen_unadjusted_comparator"][
                        "valid_queries"
                    ]
                    for value in primary_sequence
                )
            ),
            "sequences_with_unchanged_near_query_count": int(
                sum(
                    value["near_valid_queries"]
                    == value["frozen_unadjusted_comparator"][
                        "near_valid_queries"
                    ]
                    for value in primary_sequence
                )
            ),
        },
        "animals": animal_result,
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    _save_figure(result, argument.figure)
    print(
        json.dumps(
            {
                animal: value["animal_level_primary_mean"]
                for animal, value in animal_result.items()
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
