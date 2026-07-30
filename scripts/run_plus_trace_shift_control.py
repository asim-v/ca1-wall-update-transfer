"""Run a diagnostic circular trace-position shift control for the plus arena.

This is a single-animal, post-selection diagnostic and not cohort-level
inference. It intentionally leaves the frozen primary runner unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

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
    SessionMetric,
    estimate_session_metric,
    occupancy_balance_weights,
    query_balanced_block_folds,
)
from ca1_geometry.shift_control import (
    circular_shift_population,
    generate_population_shift_lags,
    plus_one_pvalues,
    unique_sequence_sessions,
)


DISTANCES_CM = np.array([2.5, 7.5, 12.5])
DISTANCE_EDGES_CM = np.array([0.0, 5.0, 10.0, 15.0])
TANGENTIAL_FRACTIONS = np.array([0.4, 0.5, 0.6])
BANDWIDTH_CM = 10.0
N_FOLD = 4
BLOCK_FRAMES = 60 * 30
GUARD_FRAMES = 1 * 30
FRAME_RATE_HZ = 30


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _discover_plus_sequences(
    animal: Mat73Animal,
) -> tuple[list[tuple[int, int, int]], tuple[int, ...]]:
    square_sessions = [
        session
        for session in range(animal.n_sessions)
        if animal.environment(session) == "square"
    ]
    sequence: list[tuple[int, int, int]] = []
    for square_pre, square_post in zip(
        square_sessions[:-1], square_sessions[1:], strict=True
    ):
        plus = [
            session
            for session in range(square_pre + 1, square_post)
            if animal.environment(session) == "+"
        ]
        if len(plus) != 1:
            raise ValueError(
                f"expected one '+' session between squares "
                f"{square_pre + 1} and {square_post + 1}; found {len(plus)}"
            )
        sequence.append((square_pre, plus[0], square_post))
    if not sequence:
        raise ValueError("no complete square-plus-square sequence found")

    blocked = {animal.blocked(value[1]) for value in sequence}
    if len(blocked) != 1:
        raise ValueError("plus sessions have inconsistent blocked partitions")
    return sequence, blocked.pop()


def _select_sequences(
    all_sequences: list[tuple[int, int, int]],
    requested: list[int] | None,
) -> list[tuple[int, tuple[int, int, int]]]:
    if requested is None:
        return list(enumerate(all_sequences, start=1))
    if len(set(requested)) != len(requested):
        raise ValueError("--sequences must not contain duplicates")
    invalid = [
        value
        for value in requested
        if not 1 <= value <= len(all_sequences)
    ]
    if invalid:
        raise ValueError(
            f"sequence indices out of range 1..{len(all_sequences)}: "
            f"{invalid}"
        )
    return [(value, all_sequences[value - 1]) for value in requested]


def _estimate(
    position: np.ndarray,
    response: np.ndarray,
    query: np.ndarray,
    *,
    blocked: tuple[int, ...],
    weight: np.ndarray,
    fold: np.ndarray,
) -> SessionMetric:
    return estimate_session_metric(
        position,
        response,
        query,
        common_blocked=blocked,
        config=LocalMapConfig(
            bandwidth=BANDWIDTH_CM,
            min_effective_samples=40.0,
            min_design_eigenratio=0.01,
        ),
        n_fold=N_FOLD,
        sample_weight=weight,
        fold_assignment=fold,
    )


def _near_statistic(
    estimate: tuple[SessionMetric, SessionMetric, SessionMetric],
    queries: Any,
) -> dict[str, float | int | None]:
    square_pre, condition, square_post = estimate
    reference = 0.5 * (square_pre.metric + square_post.metric)
    pooled_reference = 0.5 * (
        square_pre.pooled + square_post.pooled
    )
    valid = square_pre.valid & condition.valid & square_post.valid
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
    return {
        "near_anisotropy": _finite(profile.anisotropy[0]),
        "near_normal_magnification": _finite(
            profile.normal_magnification[0]
        ),
        "near_tangential_change": _finite(
            profile.tangential_change[0]
        ),
        "near_valid_queries": int(profile.n_query[0]),
        "all_valid_queries": int(valid.sum()),
    }


def _sequence_weights(
    selected: list[tuple[int, tuple[int, int, int]]],
    position: dict[int, np.ndarray],
    keep: dict[int, np.ndarray],
) -> dict[int, dict[int, np.ndarray]]:
    result: dict[int, dict[int, np.ndarray]] = {}
    for label, sequence in selected:
        weights = occupancy_balance_weights(
            [position[session] for session in sequence],
            [keep[session] for session in sequence],
        )
        result[label] = {
            session: weight
            for session, weight in zip(sequence, weights, strict=True)
        }
    return result


def _analyze_responses(
    selected: list[tuple[int, tuple[int, int, int]]],
    position: dict[int, np.ndarray],
    response: dict[int, np.ndarray],
    fold: dict[int, np.ndarray],
    weights: dict[int, dict[int, np.ndarray]],
    queries: Any,
    blocked: tuple[int, ...],
) -> dict[int, dict[str, float | int | None]]:
    output: dict[int, dict[str, float | int | None]] = {}
    for label, sequence in selected:
        estimate = tuple(
            _estimate(
                position[session],
                response[session],
                queries.position,
                blocked=blocked,
                weight=weights[label][session],
                fold=fold[session],
            )
            for session in sequence
        )
        output[label] = _near_statistic(estimate, queries)
    return output


def _fixed_sequence_mean(values: list[float | None]) -> float:
    array = np.asarray(
        [np.nan if value is None else value for value in values],
        dtype=np.float64,
    )
    if not np.isfinite(array).all():
        return float("nan")
    return float(array.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("animal_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--n-shuffle",
        type=int,
        default=19,
        help="number of Monte Carlo circular shifts (default: 19)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260729,
        help="deterministic random seed",
    )
    parser.add_argument(
        "--min-shift-seconds",
        type=int,
        default=30,
        help="minimum circular distance from the unshifted trace",
    )
    parser.add_argument(
        "--sequences",
        type=int,
        nargs="+",
        help="optional one-based exposure sequence indices",
    )
    argument = parser.parse_args()
    if argument.n_shuffle < 1:
        parser.error("--n-shuffle must be positive")
    if argument.min_shift_seconds < 1:
        parser.error("--min-shift-seconds must be positive")

    with Mat73Animal(argument.animal_file) as animal:
        all_sequences, blocked = _discover_plus_sequences(animal)
        selected = _select_sequences(all_sequences, argument.sequences)

        # Keep the frozen cell set independent of an optional sequence subset.
        all_stable_sessions = unique_sequence_sessions(all_sequences)
        stable_cells = animal.common_registered_cells(*all_stable_sessions)
        selected_sessions = unique_sequence_sessions(
            [sequence for _, sequence in selected]
        )
        position = {
            session: animal.position(session)
            for session in all_stable_sessions
        }
        response = {
            session: animal.trace(session, stable_cells)
            for session in selected_sessions
        }
        nonfinite = {
            session: int(np.count_nonzero(~np.isfinite(value)))
            for session, value in response.items()
            if not np.isfinite(value).all()
        }
        if nonfinite:
            raise ValueError(
                "stable-cell traces contain non-finite values; shifting "
                f"would change the fixed frame mask: {nonfinite}"
            )

        queries = segment_boundary_queries(
            introduced_boundaries(blocked),
            DISTANCES_CM,
            tangential_fractions=TANGENTIAL_FRACTIONS,
        )
        keep = {
            session: positions_on_accessible_support(
                position[session], blocked
            )
            for session in selected_sessions
        }
        fold = {
            session: query_balanced_block_folds(
                position[session],
                keep[session],
                queries.position,
                bandwidth=BANDWIDTH_CM,
                n_fold=N_FOLD,
                block_frames=BLOCK_FRAMES,
                guard_frames=GUARD_FRAMES,
            )
            for session in selected_sessions
        }
        weights = _sequence_weights(selected, position, keep)
        lags = generate_population_shift_lags(
            {
                session: position[session].shape[0]
                for session in all_stable_sessions
            },
            n_shuffle=argument.n_shuffle,
            seed=argument.seed,
            minimum_lag_frames=(
                argument.min_shift_seconds * FRAME_RATE_HZ
            ),
        )

        observed = _analyze_responses(
            selected,
            position,
            response,
            fold,
            weights,
            queries,
            blocked,
        )
        surrogate_by_sequence: dict[int, list[float | None]] = {
            label: [] for label, _ in selected
        }
        surrogate_mean: list[float | None] = []
        for shuffle in range(argument.n_shuffle):
            shifted = {
                session: circular_shift_population(
                    value, int(lags[session][shuffle])
                )
                for session, value in response.items()
            }
            shuffled = _analyze_responses(
                selected,
                position,
                shifted,
                fold,
                weights,
                queries,
                blocked,
            )
            values = []
            for label, _ in selected:
                value = shuffled[label]["near_anisotropy"]
                surrogate_by_sequence[label].append(value)
                values.append(value)
            surrogate_mean.append(_finite(_fixed_sequence_mean(values)))
            print(
                f"completed shift {shuffle + 1}/{argument.n_shuffle}",
                file=sys.stderr,
                flush=True,
            )

        observed_values = [
            observed[label]["near_anisotropy"]
            for label, _ in selected
        ]
        observed_mean = _fixed_sequence_mean(observed_values)
        result = {
            "status": (
                "diagnostic_trace_position_shift_control_"
                "not_cohort_inference"
            ),
            "diagnostic_only": True,
            "post_selection_note": (
                "The plus condition was selected using behavior-only support "
                "after the frozen center-condition analysis; Monte Carlo "
                "p-values are descriptive diagnostics, not confirmatory "
                "cohort inference."
            ),
            "animal": argument.animal_file.stem.replace(".complete", ""),
            "condition_environment": "+",
            "selected_sequences_one_based": [
                label for label, _ in selected
            ],
            "sequence_sessions_one_based": {
                f"sequence_{label}": [
                    session + 1 for session in sequence
                ]
                for label, sequence in selected
            },
            "stable_sessions_one_based": [
                session + 1 for session in all_stable_sessions
            ],
            "stable_cell_count": int(stable_cells.size),
            "stable_cell_indices_zero_based": [
                int(value) for value in stable_cells
            ],
            "frozen_analysis": {
                "bandwidth_cm": BANDWIDTH_CM,
                "weighting": "coarse occupancy balanced within sequence",
                "folds": N_FOLD,
                "fold_scheme": (
                    "query-design-balanced whole temporal blocks using "
                    "positions only"
                ),
                "block_seconds": BLOCK_FRAMES / FRAME_RATE_HZ,
                "guard_seconds_each_edge": (
                    GUARD_FRAMES / FRAME_RATE_HZ
                ),
                "query_distances_cm": [
                    float(value) for value in DISTANCES_CM
                ],
                "tangential_fractions": [
                    float(value) for value in TANGENTIAL_FRACTIONS
                ],
                "query_count": int(queries.position.shape[0]),
                "common_blocked_partitions": list(blocked),
                "cells_queries_folds_weights_fixed_across_shifts": True,
            },
            "shift_control": {
                "n_shuffle": argument.n_shuffle,
                "seed": argument.seed,
                "frame_rate_hz": FRAME_RATE_HZ,
                "minimum_lag_seconds": argument.min_shift_seconds,
                "population_shared_lag_within_session": True,
                "shared_session_lag_reused_across_sequences": True,
                "lag_sampling": (
                    "uniform integer circular lag with at least the stated "
                    "distance from zero in either direction"
                ),
                "session_lags_frames": {
                    str(session + 1): [
                        int(value) for value in session_lags
                    ]
                    for session, session_lags in lags.items()
                },
            },
            "observed": {
                "sequence": {
                    f"sequence_{label}": observed[label]
                    for label, _ in selected
                },
                "unweighted_mean_near_anisotropy": _finite(observed_mean),
            },
            "surrogate_near_anisotropy": {
                "sequence": {
                    f"sequence_{label}": surrogate_by_sequence[label]
                    for label, _ in selected
                },
                "unweighted_complete_sequence_mean": surrogate_mean,
            },
            "plus_one_p_values": {
                "sequence": {
                    f"sequence_{label}": plus_one_pvalues(
                        float(
                            np.nan
                            if observed[label]["near_anisotropy"] is None
                            else observed[label]["near_anisotropy"]
                        ),
                        surrogate_by_sequence[label],
                    )
                    for label, _ in selected
                },
                "unweighted_complete_sequence_mean": plus_one_pvalues(
                    observed_mean,
                    surrogate_mean,
                ),
            },
        }

    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "observed_mean_near_anisotropy": _finite(observed_mean),
                "mean_plus_one_p_values": result["plus_one_p_values"][
                    "unweighted_complete_sequence_mean"
                ],
                "output": str(argument.output),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
