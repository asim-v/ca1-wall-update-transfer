"""Run a post-outcome exploratory all-exposure common-support analysis.

This script deliberately does not replace or modify the frozen primary
condition analysis. It refits that 10 cm occupancy-balanced estimator, then
restricts every longitudinal sequence to the identical query intersection
that is valid in all nine sequence-session fits.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import zlib

import matplotlib.pyplot as plt
import numpy as np

from ca1_geometry.arena import (
    BoundaryQueries,
    introduced_boundaries,
    positions_on_accessible_support,
    segment_boundary_queries,
)
from ca1_geometry.exploratory import exact_segment_label_spin
from ca1_geometry.io import Mat73Animal
from ca1_geometry.local_linear import LocalMapConfig
from ca1_geometry.metrics import (
    anisotropy_components,
    anisotropy_profile,
    cross_metric,
    pooled_metric,
)
from ca1_geometry.pilot import (
    DirectionalReliability,
    SessionMetric,
    estimate_session_metric,
    occupancy_balance_weights,
    query_balanced_block_folds,
    residual_directional_reliability,
)


DISTANCES_CM = np.array([2.5, 7.5, 12.5])
DISTANCE_EDGES_CM = np.array([0.0, 5.0, 10.0, 15.0])
TANGENTIAL_FRACTIONS = np.array([0.4, 0.5, 0.6])
BANDWIDTH_CM = 10.0
N_FOLD = 4
BLOCK_FRAMES = 60 * 30
GUARD_FRAMES = 1 * 30
DEFAULT_SUBSET_NEURONS = 47
DEFAULT_SUBSET_DRAWS = 200
DEFAULT_SEED = 20_260_729


@dataclass(frozen=True)
class SequenceFit:
    """In-memory fits needed for common-support postprocessing."""

    sessions: tuple[int, int, int]
    estimates: tuple[SessionMetric, SessionMetric, SessionMetric]
    mean_event_rate_hz: tuple[float, float, float]
    native_valid: np.ndarray


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


def _reliability_dict(value: DirectionalReliability) -> dict[str, Any]:
    return {
        "normal": _finite(value.normal),
        "tangent": _finite(value.tangent),
        "contrast": _finite(value.contrast),
        "n_query": int(value.n_query),
    }


def _difference(last: float | None, first: float | None) -> float | None:
    if last is None or first is None:
        return None
    return float(last - first)


def _list_difference(
    last: list[float | None], first: list[float | None]
) -> list[float | None]:
    return [
        _difference(last_value, first_value)
        for last_value, first_value in zip(last, first, strict=True)
    ]


def _distribution(value: list[float]) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "n_draws": int(array.size),
            "n_finite": 0,
            "mean": None,
            "sd": None,
            "q025": None,
            "median": None,
            "q975": None,
            "minimum": None,
            "maximum": None,
            "fraction_positive": None,
        }
    return {
        "n_draws": int(array.size),
        "n_finite": int(finite.size),
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite)),
        "q025": float(np.quantile(finite, 0.025)),
        "median": float(np.median(finite)),
        "q975": float(np.quantile(finite, 0.975)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "fraction_positive": float(np.mean(finite > 0)),
    }


def _discover_plus_sequences(
    animal: Mat73Animal,
) -> tuple[list[tuple[int, int, int]], tuple[int, ...]]:
    square_sessions = [
        session
        for session in range(animal.n_sessions)
        if animal.environment(session) == "square"
    ]
    sequences = []
    for square_pre, square_post in zip(
        square_sessions[:-1], square_sessions[1:], strict=True
    ):
        matches = [
            session
            for session in range(square_pre + 1, square_post)
            if animal.environment(session) == "+"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one '+' session between squares "
                f"{square_pre + 1} and {square_post + 1}; found {len(matches)}"
            )
        sequences.append((square_pre, matches[0], square_post))
    if len(sequences) != 3:
        raise ValueError(
            f"expected three square-'+'-square sequences; found {len(sequences)}"
        )
    blocked_values = {
        animal.blocked(sessions[1]) for sessions in sequences
    }
    if len(blocked_values) != 1:
        raise ValueError("the '+' blocked partition mask changed over exposure")
    return sequences, blocked_values.pop()


def _fit_sequence(
    animal: Mat73Animal,
    sessions: tuple[int, int, int],
    cells: np.ndarray,
    queries: BoundaryQueries,
    blocked: tuple[int, ...],
) -> SequenceFit:
    position = [animal.position(session) for session in sessions]
    keep = [
        positions_on_accessible_support(value, blocked)
        for value in position
    ]
    weight = occupancy_balance_weights(position, keep)
    fold = [
        query_balanced_block_folds(
            position[index],
            keep[index],
            queries.position,
            bandwidth=BANDWIDTH_CM,
            n_fold=N_FOLD,
            block_frames=BLOCK_FRAMES,
            guard_frames=GUARD_FRAMES,
        )
        for index in range(3)
    ]
    config = LocalMapConfig(
        bandwidth=BANDWIDTH_CM,
        min_effective_samples=40.0,
        min_design_eigenratio=0.01,
    )
    estimates: list[SessionMetric] = []
    rates: list[float] = []
    for index, session in enumerate(sessions):
        response = animal.trace(session, cells)
        rates.append(float(np.mean(response) * 30.0))
        estimates.append(
            estimate_session_metric(
                position[index],
                response,
                queries.position,
                common_blocked=blocked,
                config=config,
                n_fold=N_FOLD,
                sample_weight=weight[index],
                fold_assignment=fold[index],
            )
        )
    estimate_tuple = (estimates[0], estimates[1], estimates[2])
    return SequenceFit(
        sessions=sessions,
        estimates=estimate_tuple,
        mean_event_rate_hz=(rates[0], rates[1], rates[2]),
        native_valid=np.logical_and.reduce(
            [estimate.valid for estimate in estimate_tuple]
        ),
    )


def _summarize_sequence(
    fit: SequenceFit,
    common_valid: np.ndarray,
    queries: BoundaryQueries,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    square_pre, condition, square_post = fit.estimates
    reference = 0.5 * (square_pre.metric + square_post.metric)
    pooled_reference = 0.5 * (square_pre.pooled + square_post.pooled)
    target = anisotropy_profile(
        condition.metric,
        reference,
        queries.normal,
        queries.distance,
        DISTANCE_EDGES_CM,
        tangent=queries.tangent,
        valid=common_valid,
        denominator_metric=pooled_reference,
    )
    square_null = anisotropy_profile(
        square_post.metric,
        square_pre.metric,
        queries.normal,
        queries.distance,
        DISTANCE_EDGES_CM,
        tangent=queries.tangent,
        valid=common_valid,
        denominator_metric=pooled_reference,
    )
    near = common_valid & (queries.distance < 5.0)
    reliability = residual_directional_reliability(
        condition.jacobians,
        [square_pre.jacobians, square_post.jacobians],
        queries.normal,
        queries.tangent,
        valid=common_valid,
    )
    near_reliability = residual_directional_reliability(
        condition.jacobians,
        [square_pre.jacobians, square_post.jacobians],
        queries.normal,
        queries.tangent,
        valid=near,
    )
    component = anisotropy_components(
        condition.metric,
        reference,
        queries.normal,
        tangent=queries.tangent,
    )
    denominator = np.trace(pooled_reference, axis1=1, axis2=2)
    output = {
        "sessions_one_based": [session + 1 for session in fit.sessions],
        "mean_event_rate_hz": list(fit.mean_event_rate_hz),
        "native_valid_queries_before_intersection": int(
            fit.native_valid.sum()
        ),
        "target": _profile_dict(target),
        "square_pseudo_wall_null": _profile_dict(square_null),
        "target_residual_reliability": _reliability_dict(reliability),
        "near_target_residual_reliability": _reliability_dict(
            near_reliability
        ),
        "frames_per_fold": {
            name: [int(item) for item in estimate.frames_per_fold]
            for name, estimate in zip(
                ("square_pre", "condition", "square_post"),
                fit.estimates,
                strict=True,
            )
        },
    }
    return output, component["contrast"], denominator


def _endpoint_change(
    first: dict[str, Any],
    last: dict[str, Any],
) -> dict[str, Any]:
    target_first = first["target"]
    target_last = last["target"]
    reliability_first = first["target_residual_reliability"]
    reliability_last = last["target_residual_reliability"]
    near_reliability_first = first["near_target_residual_reliability"]
    near_reliability_last = last["near_target_residual_reliability"]
    return {
        "definition": "sequence_3_minus_sequence_1",
        "target": {
            key: _list_difference(target_last[key], target_first[key])
            for key in (
                "anisotropy",
                "normal_magnification",
                "tangential_change",
            )
        },
        "target_residual_reliability": {
            key: _difference(reliability_last[key], reliability_first[key])
            for key in ("normal", "tangent", "contrast")
        },
        "near_target_residual_reliability": {
            key: _difference(
                near_reliability_last[key], near_reliability_first[key]
            )
            for key in ("normal", "tangent", "contrast")
        },
    }


def _equal_neuron_sensitivity(
    fits: list[SequenceFit],
    common_valid: np.ndarray,
    queries: BoundaryQueries,
    stable_cells: np.ndarray,
    *,
    subset_neurons: int,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if stable_cells.size < subset_neurons:
        return {
            "status": "skipped_insufficient_stable_cells",
            "available_stable_cells": int(stable_cells.size),
            "requested_subset_neurons": int(subset_neurons),
        }
    rng = np.random.default_rng(seed)
    draw_values = [
        {
            "near_anisotropy": [],
            "near_normal_magnification": [],
            "contrast_reliability": [],
            "near_contrast_reliability": [],
        }
        for _ in fits
    ]
    endpoint_values = {
        "near_anisotropy": [],
        "near_normal_magnification": [],
        "contrast_reliability": [],
        "near_contrast_reliability": [],
    }
    unique_subsets: set[bytes] = set()
    first_subset: np.ndarray | None = None
    near = common_valid & (queries.distance < 5.0)

    for _ in range(draws):
        subset = np.sort(
            rng.choice(
                stable_cells.size,
                size=subset_neurons,
                replace=False,
            )
        )
        unique_subsets.add(subset.tobytes())
        if first_subset is None:
            first_subset = subset.copy()
        current: list[dict[str, float]] = []
        for sequence_index, fit in enumerate(fits):
            fold_jacobian = [
                np.take(estimate.jacobians, subset, axis=2)
                for estimate in fit.estimates
            ]
            pooled_jacobian = [
                np.take(estimate.pooled_jacobian, subset, axis=1)
                for estimate in fit.estimates
            ]
            metric = [cross_metric(value) for value in fold_jacobian]
            pooled = [pooled_metric(value) for value in pooled_jacobian]
            reference = 0.5 * (metric[0] + metric[2])
            pooled_reference = 0.5 * (pooled[0] + pooled[2])
            profile = anisotropy_profile(
                metric[1],
                reference,
                queries.normal,
                queries.distance,
                DISTANCE_EDGES_CM,
                tangent=queries.tangent,
                valid=common_valid,
                denominator_metric=pooled_reference,
            )
            reliability = residual_directional_reliability(
                fold_jacobian[1],
                [fold_jacobian[0], fold_jacobian[2]],
                queries.normal,
                queries.tangent,
                valid=common_valid,
            )
            near_reliability = residual_directional_reliability(
                fold_jacobian[1],
                [fold_jacobian[0], fold_jacobian[2]],
                queries.normal,
                queries.tangent,
                valid=near,
            )
            value = {
                "near_anisotropy": float(profile.anisotropy[0]),
                "near_normal_magnification": float(
                    profile.normal_magnification[0]
                ),
                "contrast_reliability": float(reliability.contrast),
                "near_contrast_reliability": float(
                    near_reliability.contrast
                ),
            }
            current.append(value)
            for key, item in value.items():
                draw_values[sequence_index][key].append(item)
        for key in endpoint_values:
            endpoint_values[key].append(current[2][key] - current[0][key])

    if first_subset is None:
        raise RuntimeError("no equal-neuron subsets were generated")
    return {
        "status": "post_outcome_exploratory_equal_neuron_count_sensitivity",
        "definition": (
            "each seeded subset is fixed across all seven sessions and all "
            "three exposures; tensors are recomputed from stored fit "
            "Jacobians without refitting"
        ),
        "subset_neurons": int(subset_neurons),
        "draws": int(draws),
        "seed": int(seed),
        "unique_subset_count": len(unique_subsets),
        "first_subset_longitudinal_cell_indices_one_based": [
            int(item + 1) for item in stable_cells[first_subset]
        ],
        "sequences": {
            f"sequence_{index + 1}": {
                key: _distribution(value)
                for key, value in sequence.items()
            }
            for index, sequence in enumerate(draw_values)
        },
        "endpoint_change_sequence_3_minus_sequence_1": {
            key: _distribution(value)
            for key, value in endpoint_values.items()
        },
    }


def _common_support_dict(
    common_valid: np.ndarray,
    queries: BoundaryQueries,
    segment_count: int,
) -> dict[str, Any]:
    near = common_valid & (queries.distance < 5.0)
    near_segment = np.unique(queries.segment_index[near])
    by_distance = []
    for distance in DISTANCES_CM:
        selected = common_valid & np.isclose(queries.distance, distance)
        by_distance.append(
            {
                "distance_cm": float(distance),
                "valid_queries": int(selected.sum()),
                "valid_segments": int(
                    np.unique(queries.segment_index[selected]).size
                ),
            }
        )
    return {
        "definition": (
            "logical intersection of valid query masks from all three "
            "sequence triplets; reused unchanged for every sequence endpoint"
        ),
        "valid_queries": int(common_valid.sum()),
        "query_total": int(common_valid.size),
        "near_valid_queries": int(near.sum()),
        "near_query_total": int(np.count_nonzero(queries.distance < 5.0)),
        "near_valid_segments": int(near_segment.size),
        "segment_total": int(segment_count),
        "near_segment_indices_zero_based": [
            int(item) for item in near_segment
        ],
        "by_distance": by_distance,
    }


def _fit_animal(
    path: Path,
    *,
    subset_neurons: int,
    subset_draws: int,
    seed: int,
) -> dict[str, Any]:
    with Mat73Animal(path) as animal:
        sequences, blocked = _discover_plus_sequences(animal)
        segments = introduced_boundaries(blocked)
        queries = segment_boundary_queries(
            segments,
            DISTANCES_CM,
            tangential_fractions=TANGENTIAL_FRACTIONS,
        )
        stable_sessions = sorted(
            {session for sequence in sequences for session in sequence}
        )
        if len(stable_sessions) != 7:
            raise ValueError(
                "overlapping longitudinal triplets must contain seven sessions"
            )
        stable_cells = animal.common_registered_cells(*stable_sessions)
        fits = [
            _fit_sequence(
                animal,
                sequence,
                stable_cells,
                queries,
                blocked,
            )
            for sequence in sequences
        ]

    common_valid = np.logical_and.reduce(
        [fit.native_valid for fit in fits]
    )
    near = common_valid & (queries.distance < 5.0)
    sequence_result: dict[str, Any] = {}
    contrast: list[np.ndarray] = []
    denominator: list[np.ndarray] = []
    for index, fit in enumerate(fits):
        summary, query_contrast, query_denominator = _summarize_sequence(
            fit, common_valid, queries
        )
        sequence_result[f"sequence_{index + 1}"] = summary
        contrast.append(query_contrast)
        denominator.append(query_denominator)

    label_spin = exact_segment_label_spin(
        np.stack(contrast),
        np.stack(denominator),
        queries.segment_index,
        valid=near,
        n_segments=len(segments),
    )
    animal_name = path.stem.replace(".complete", "")
    animal_seed = seed + zlib.crc32(animal_name.encode("utf-8"))
    sensitivity = _equal_neuron_sensitivity(
        fits,
        common_valid,
        queries,
        stable_cells,
        subset_neurons=subset_neurons,
        draws=subset_draws,
        seed=animal_seed,
    )
    boundary = [
        {
            "segment_index_zero_based": index,
            "start_cm": list(segment.start),
            "end_cm": list(segment.end),
            "normal": list(segment.normal),
            "blocked_partition": segment.blocked_partition,
            "accessible_partition": segment.accessible_partition,
        }
        for index, segment in enumerate(segments)
    ]
    sequences_as_list = list(sequence_result.values())
    return {
        "animal": animal_name,
        "input_file": str(path),
        "blocked_partitions": list(blocked),
        "boundary_segments": boundary,
        "stable_sessions_one_based": [
            session + 1 for session in stable_sessions
        ],
        "stable_cell_count": int(stable_cells.size),
        "common_query_intersection": _common_support_dict(
            common_valid, queries, len(segments)
        ),
        "sequences": sequence_result,
        "endpoint_change": _endpoint_change(
            sequences_as_list[0], sequences_as_list[2]
        ),
        "exact_boundary_segment_label_spin": label_spin,
        "equal_neuron_count_sensitivity": sensitivity,
    }


def _cohort_label_spin(animals: list[dict[str, Any]]) -> dict[str, Any]:
    signatures = [
        [
            (
                tuple(segment["start_cm"]),
                tuple(segment["end_cm"]),
                tuple(segment["normal"]),
            )
            for segment in animal["boundary_segments"]
        ]
        for animal in animals
    ]
    if any(value != signatures[0] for value in signatures[1:]):
        raise ValueError(
            "cohort label spins require identical boundary segment geometry"
        )
    statistic = np.asarray(
        [
            animal["exact_boundary_segment_label_spin"][
                "exact_statistics"
            ]
            for animal in animals
        ],
        dtype=np.float64,
    )
    animal_mean = np.mean(statistic, axis=0)
    observed = float(
        np.mean(
            [
                animal["exact_boundary_segment_label_spin"][
                    "observed_statistic"
                ]
                for animal in animals
            ]
        )
    )
    tolerance = 16.0 * np.finfo(float).eps * max(1.0, abs(observed))
    return {
        "status": (
            "exploratory_label_calibration_not_a_randomized_experiment_p_value"
        ),
        "definition": (
            "animal mean of exposure-pooled near-wall contrasts under the "
            "same one-segment label spin in every animal and exposure"
        ),
        "n_animals": len(animals),
        "n_segments": len(signatures[0]),
        "n_exact_labelings": int(animal_mean.size),
        "observed_animal_mean": observed,
        "two_sided_tail_fraction": float(
            np.mean(np.abs(animal_mean) >= abs(observed) - tolerance)
        ),
        "positive_tail_fraction": float(
            np.mean(animal_mean >= observed - tolerance)
        ),
        "spin_distribution": {
            "minimum": float(np.min(animal_mean)),
            "q025": float(np.quantile(animal_mean, 0.025)),
            "median": float(np.median(animal_mean)),
            "q975": float(np.quantile(animal_mean, 0.975)),
            "maximum": float(np.max(animal_mean)),
        },
        "exact_animal_mean_statistics": animal_mean.tolist(),
    }


def _save_figure(result: dict[str, Any], output: Path) -> None:
    animals = result["animals"]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(animals)))
    x = np.arange(1, 4)
    for color, animal in zip(colors, animals, strict=True):
        sequence = list(animal["sequences"].values())
        label = animal["animal"]
        axes[0].plot(
            x,
            [item["target"]["anisotropy"][0] for item in sequence],
            marker="o",
            color=color,
            label=label,
        )
        axes[1].plot(
            x,
            [
                item["target_residual_reliability"]["contrast"]
                for item in sequence
            ],
            marker="o",
            color=color,
        )

    axes[0].axhline(0.0, color="0.5", linewidth=1)
    axes[0].set(
        xlabel="Exposure sequence",
        ylabel="Normal-minus-tangent / square scale",
        title="Near-wall anisotropy",
        xticks=x,
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0.0, color="0.5", linewidth=1)
    axes[1].axhline(0.3, color="0.6", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="Exposure sequence",
        ylabel="Fold-pair correlation",
        title="Residual contrast reliability",
        xticks=x,
        ylim=(-1.0, 1.0),
    )

    labels = [animal["animal"] for animal in animals] + ["animal mean"]
    diagnostic = [
        animal["exact_boundary_segment_label_spin"] for animal in animals
    ] + [result["cohort_exact_boundary_segment_label_spin"]]
    observed = [
        value.get("observed_statistic", value.get("observed_animal_mean"))
        for value in diagnostic
    ]
    lower = [value["spin_distribution"]["q025"] for value in diagnostic]
    upper = [value["spin_distribution"]["q975"] for value in diagnostic]
    y = np.arange(len(labels))
    axes[2].hlines(y, lower, upper, color="0.55", linewidth=3)
    axes[2].scatter(observed, y, color="0.15", zorder=3)
    axes[2].axvline(0.0, color="0.5", linewidth=1)
    axes[2].set(
        xlabel="Exposure-pooled near anisotropy (95% spin interval)",
        title="Segment-label calibration",
        yticks=y,
        yticklabels=labels,
    )
    figure.suptitle(
        "Post-outcome exploratory common-query support for the '+' condition"
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("animal_files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument(
        "--subset-neurons",
        type=int,
        default=DEFAULT_SUBSET_NEURONS,
    )
    parser.add_argument(
        "--subset-draws",
        type=int,
        default=DEFAULT_SUBSET_DRAWS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    argument = parser.parse_args()
    if argument.subset_neurons < 1 or argument.subset_draws < 1:
        parser.error("subset-neurons and subset-draws must be positive")
    if len({path.resolve() for path in argument.animal_files}) != len(
        argument.animal_files
    ):
        parser.error("animal files must be unique")

    animals = [
        _fit_animal(
            path,
            subset_neurons=argument.subset_neurons,
            subset_draws=argument.subset_draws,
            seed=argument.seed,
        )
        for path in argument.animal_files
    ]
    result = {
        "status": (
            "post_outcome_exploratory_common_spatial_support_not_confirmatory"
        ),
        "condition": "+",
        "interpretation_limit": (
            "the common-support, label-spin, and equal-neuron analyses were "
            "specified after inspecting frozen condition outcomes"
        ),
        "estimator": {
            "bandwidth_cm": BANDWIDTH_CM,
            "weighting": "coarse occupancy balanced within each triplet",
            "folds": N_FOLD,
            "fold_scheme": (
                "query-design-balanced whole 60-second temporal blocks using "
                "positions only, with one-second edge guards"
            ),
            "stable_population": (
                "one seven-session registered-cell core per animal"
            ),
            "query_distances_cm": _finite_list(DISTANCES_CM),
            "tangential_fractions": _finite_list(TANGENTIAL_FRACTIONS),
            "normalizer": "pooled bracketing-square tensor, scale only",
        },
        "animals": animals,
        "cohort_exact_boundary_segment_label_spin": _cohort_label_spin(
            animals
        ),
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    figure = (
        argument.figure
        if argument.figure is not None
        else argument.output.with_suffix(".png")
    )
    _save_figure(result, figure)

    compact = {
        animal["animal"]: {
            "stable_cells": animal["stable_cell_count"],
            "common_near_queries": animal[
                "common_query_intersection"
            ]["near_valid_queries"],
            "common_near_segments": animal[
                "common_query_intersection"
            ]["near_valid_segments"],
            "near_anisotropy": [
                value["target"]["anisotropy"][0]
                for value in animal["sequences"].values()
            ],
            "contrast_reliability": [
                value["target_residual_reliability"]["contrast"]
                for value in animal["sequences"].values()
            ],
            "endpoint_near_anisotropy_change": animal["endpoint_change"][
                "target"
            ]["anisotropy"][0],
            "label_spin_two_sided_tail_fraction": animal[
                "exact_boundary_segment_label_spin"
            ]["two_sided_tail_fraction"],
        }
        for animal in animals
    }
    compact["cohort_label_spin"] = {
        key: result["cohort_exact_boundary_segment_label_spin"][key]
        for key in (
            "observed_animal_mean",
            "two_sided_tail_fraction",
            "positive_tail_fraction",
        )
    }
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
