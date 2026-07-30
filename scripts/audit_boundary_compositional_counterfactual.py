"""Audit and narrowly test additive wall-profile counterfactuals.

The released deformation series does not contain globally isolated one-wall
environments.  This script therefore separates two questions that can easily
be conflated:

1. Is a held-out whole-shape response identifiable from an additive wall
   design?
2. Are there any *local* corners for which the observed shapes supply all
   four states of a two-wall factorial (OO, WO, OW, WW)?

The first question is answered from geometry alone.  For the second, the
script constructs an interaction-free WW predictor from OO, WO, and OW neural
profiles in one exposure cycle, withholds every WW neural rate from fitting,
and scores the prediction on WW in the following cycle against a distinct
square baseline.  The local test is a deliberately limited falsification
check; it is not a general whole-shape decoder.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ca1_geometry.boundary_fragments import (  # noqa: E402
    common_support_bins,
    globally_demeaned_local_cell_rate,
    local_cell_rate,
    spearman_correlation,
)
from ca1_geometry.io import Mat73Animal  # noqa: E402
from ca1_geometry.seams import (  # noqa: E402
    OrientedSeam,
    SeamState,
    internal_seams,
    seam_frame,
    seam_state,
    seam_strip_bins,
)


RateFunction = Callable[
    [np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, int], ...]],
    np.ndarray,
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "boundary_compositional_counterfactual.json"
        ),
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-bins", type=int, default=6)
    parser.add_argument("--minimum-cells", type=int, default=20)
    return parser.parse_args()


def _raw_local(
    rate: np.ndarray,
    occupancy: np.ndarray,
    cells: np.ndarray,
    bins: tuple[tuple[int, int], ...],
) -> np.ndarray:
    del occupancy
    return local_cell_rate(rate, cells, bins)


RATE_MODES: dict[str, RateFunction] = {
    "raw_local_rate": _raw_local,
    "global_rate_demeaned": globally_demeaned_local_cell_rate,
}


@dataclass(frozen=True)
class FactorialQuery:
    """A local orthogonal-wall pair with complete OO/WO/OW/WW coverage."""

    seam_a: OrientedSeam
    seam_b: OrientedSeam
    bins: tuple[tuple[int, int], ...]
    state_environments: dict[str, tuple[str, ...]]

    @property
    def identifier(self) -> str:
        return (
            f"{self.seam_a.source}_to_{self.seam_a.target}"
            f"__{self.seam_b.source}_to_{self.seam_b.target}"
        )


def _ordered_environment_names(
    blocked_by_environment: dict[str, tuple[int, ...]],
) -> list[str]:
    if "square" not in blocked_by_environment:
        raise ValueError("geometry must include square")
    return [
        "square",
        *sorted(
            name
            for name in blocked_by_environment
            if name != "square"
        ),
    ]


def _state_letter(state: SeamState) -> str | None:
    if state is SeamState.OPEN:
        return "O"
    if state is SeamState.WALL:
        return "W"
    return None


def factorial_queries(
    blocked_by_environment: dict[str, tuple[int, ...]],
) -> list[FactorialQuery]:
    """Enumerate orthogonal same-target seam pairs with full 2x2 coverage."""

    seams = internal_seams()
    result: list[FactorialQuery] = []
    for index, seam_a in enumerate(seams):
        for seam_b in seams[index + 1 :]:
            if seam_a.target != seam_b.target:
                continue
            normal_a = seam_frame(seam_a)[2]
            normal_b = seam_frame(seam_b)[2]
            if not np.isclose(float(normal_a @ normal_b), 0.0):
                continue
            bins = tuple(
                sorted(
                    set(seam_strip_bins(seam_a))
                    & set(seam_strip_bins(seam_b))
                )
            )
            if not bins:
                continue

            states: dict[str, list[str]] = defaultdict(list)
            for environment in _ordered_environment_names(
                blocked_by_environment
            ):
                blocked = blocked_by_environment[environment]
                first = _state_letter(seam_state(blocked, seam_a))
                second = _state_letter(seam_state(blocked, seam_b))
                if first is not None and second is not None:
                    states[first + second].append(environment)
            if not all(state in states for state in ("OO", "WO", "OW", "WW")):
                continue
            result.append(
                FactorialQuery(
                    seam_a=seam_a,
                    seam_b=seam_b,
                    bins=bins,
                    state_environments={
                        state: tuple(states[state])
                        for state in ("OO", "WO", "OW", "WW")
                    },
                )
            )
    return result


def _incidence_matrix(
    blocked_by_environment: dict[str, tuple[int, ...]],
    *,
    oriented: bool,
) -> tuple[list[str], list[OrientedSeam], np.ndarray]:
    environments = _ordered_environment_names(blocked_by_environment)
    seams = internal_seams(both_orientations=oriented)
    if oriented:
        matrix = np.asarray(
            [
                [
                    seam_state(blocked_by_environment[name], seam)
                    is SeamState.WALL
                    for seam in seams
                ]
                for name in environments
            ],
            dtype=np.float64,
        )
    else:
        matrix = np.asarray(
            [
                [
                    (
                        (seam.source in blocked_by_environment[name])
                        != (seam.target in blocked_by_environment[name])
                    )
                    for seam in seams
                ]
                for name in environments
            ],
            dtype=np.float64,
        )
    return environments, seams, matrix


def _matrix_rank(matrix: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=np.float64)))


def _design_diagnostic(
    blocked_by_environment: dict[str, tuple[int, ...]],
    *,
    oriented: bool,
) -> dict[str, Any]:
    environments, seams, matrix = _incidence_matrix(
        blocked_by_environment,
        oriented=oriented,
    )
    design = np.column_stack((np.ones(matrix.shape[0]), matrix))
    leave_one_out = {}
    for index, environment in enumerate(environments):
        if environment == "square":
            continue
        training = np.delete(design, index, axis=0)
        augmented = np.vstack((training, design[index]))
        _, singular_value, right_vector = np.linalg.svd(
            training,
            full_matrices=True,
        )
        tolerance = (
            max(training.shape)
            * singular_value[0]
            * np.finfo(np.float64).eps
        )
        training_rank = int(np.count_nonzero(singular_value > tolerance))
        row_basis = right_vector[:training_rank]
        projection = (design[index] @ row_basis.T) @ row_basis
        rowspace_residual = design[index] - projection
        residual_norm = float(np.linalg.norm(rowspace_residual))
        leave_one_out[environment] = {
            "training_rank": training_rank,
            "rank_after_adding_target": _matrix_rank(augmented),
            "target_row_in_training_row_span": (
                training_rank == _matrix_rank(augmented)
            ),
            "target_sensitivity_to_training_nullspace_l2": residual_norm,
            "training_equivalent_models_can_change_target_arbitrarily": (
                residual_norm > 1e-10
            ),
        }

    patterns: dict[tuple[int, ...], list[list[int]]] = defaultdict(list)
    for index, seam in enumerate(seams):
        pattern = tuple(int(value) for value in matrix[:, index])
        patterns[pattern].append([seam.source, seam.target])
    duplicate_patterns = [
        {
            "seams": group,
            "environment_pattern": list(pattern),
        }
        for pattern, group in patterns.items()
        if len(group) > 1
    ]
    never_present = [
        [seam.source, seam.target]
        for index, seam in enumerate(seams)
        if not np.any(matrix[:, index])
    ]
    return {
        "representation": (
            "oriented source-blocked-to-target-accessible walls"
            if oriented
            else "unoriented physical boundaries"
        ),
        "environments": environments,
        "seams": [[seam.source, seam.target] for seam in seams],
        "incidence_rows": {
            environment: [
                int(value)
                for value in matrix[index]
            ]
            for index, environment in enumerate(environments)
        },
        "observations": int(design.shape[0]),
        "wall_parameters": int(matrix.shape[1]),
        "parameters_with_intercept": int(design.shape[1]),
        "rank_without_intercept": _matrix_rank(matrix),
        "rank_with_intercept": _matrix_rank(design),
        "nullity_with_intercept": (
            int(design.shape[1]) - _matrix_rank(design)
        ),
        "unique_column_patterns": len(patterns),
        "duplicate_column_patterns": duplicate_patterns,
        "never_present_seams": never_present,
        "leave_one_deformed_shape_out": leave_one_out,
        "all_deformed_targets_estimable_from_other_shapes": all(
            item["target_row_in_training_row_span"]
            for item in leave_one_out.values()
        ),
    }


def geometry_audit(
    blocked_by_environment: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    """Return exact design-rank and local-factorial diagnostics."""

    oriented = _design_diagnostic(
        blocked_by_environment,
        oriented=True,
    )
    physical = _design_diagnostic(
        blocked_by_environment,
        oriented=False,
    )
    environments = physical["environments"]
    boundary_counts = {
        environment: int(
            sum(physical["incidence_rows"][environment])
        )
        for environment in environments
    }
    queries = factorial_queries(blocked_by_environment)
    return {
        "unique_environment_geometries": len(environments),
        "boundary_counts": boundary_counts,
        "globally_isolated_one_boundary_environments": [
            environment
            for environment, count in boundary_counts.items()
            if count == 1
        ],
        "minimum_nonzero_boundary_count": min(
            count for count in boundary_counts.values() if count > 0
        ),
        "oriented_wall_design": oriented,
        "physical_boundary_design": physical,
        "complete_local_two_wall_factorials": [
            _query_to_dict(query) for query in queries
        ],
        "complete_local_factorial_count": len(queries),
        "distinct_ww_target_environments": sorted(
            {
                environment
                for query in queries
                for environment in query.state_environments["WW"]
            }
        ),
    }


def _query_to_dict(query: FactorialQuery) -> dict[str, Any]:
    return {
        "query": query.identifier,
        "seam_a": [query.seam_a.source, query.seam_a.target],
        "seam_b": [query.seam_b.source, query.seam_b.target],
        "common_target_partition": query.seam_a.target,
        "overlap_bins": [list(value) for value in query.bins],
        "state_environments": {
            state: list(environments)
            for state, environments in query.state_environments.items()
        },
    }


def _geometry_from_sessions(
    environment: list[str],
    blocked: list[tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for name, value in zip(environment, blocked, strict=True):
        value = tuple(int(item) for item in value)
        if name in result and result[name] != value:
            raise ValueError(
                f"environment {name!r} maps to multiple blocked sets"
            )
        result[name] = value
    return result


def _cycle_session(
    environment: list[str],
    *,
    cycle: int,
    name: str,
) -> int:
    start = cycle * 10
    stop = min(start + 10, len(environment))
    matches = [
        session
        for session in range(start, stop)
        if environment[session] == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"cycle {cycle + 1} has {len(matches)} sessions named {name!r}"
        )
    return matches[0]


def _finite_rmse(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    keep = np.isfinite(first) & np.isfinite(second)
    if not np.any(keep):
        return float("nan")
    return float(np.sqrt(np.mean((first[keep] - second[keep]) ** 2)))


def _normalized_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    keep = np.isfinite(prediction) & np.isfinite(target)
    if not np.any(keep):
        return float("nan")
    denominator = float(np.sqrt(np.mean(target[keep] ** 2)))
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return _finite_rmse(prediction[keep], target[keep]) / denominator


def _score_predictors(
    *,
    additive: np.ndarray,
    wall_a_only: np.ndarray,
    wall_b_only: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any] | None:
    predictors = {
        "interaction_free_additive": additive,
        "wall_a_only": wall_a_only,
        "wall_b_only": wall_b_only,
    }
    score = {
        name: {
            "spearman_r": spearman_correlation(value, target),
            "normalized_rmse": _normalized_rmse(value, target),
        }
        for name, value in predictors.items()
    }
    if any(
        not np.isfinite(metric)
        for value in score.values()
        for metric in value.values()
    ):
        return None
    single_r = [
        score["wall_a_only"]["spearman_r"],
        score["wall_b_only"]["spearman_r"],
    ]
    single_rmse = [
        score["wall_a_only"]["normalized_rmse"],
        score["wall_b_only"]["normalized_rmse"],
    ]
    return {
        "predictors": score,
        "additive_minus_best_single_spearman_r": (
            score["interaction_free_additive"]["spearman_r"]
            - max(single_r)
        ),
        "additive_minus_mean_single_spearman_r": (
            score["interaction_free_additive"]["spearman_r"]
            - float(np.mean(single_r))
        ),
        "additive_minus_best_single_normalized_rmse": (
            score["interaction_free_additive"]["normalized_rmse"]
            - min(single_rmse)
        ),
    }


def _factorial_record(
    *,
    query: FactorialQuery,
    ww_environment: str,
    training_cycle: int,
    environment: list[str],
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    train_square = _cycle_session(
        environment,
        cycle=training_cycle,
        name="square",
    )
    test_square = _cycle_session(
        environment,
        cycle=training_cycle + 2,
        name="square",
    )
    target_test = _cycle_session(
        environment,
        cycle=training_cycle + 1,
        name=ww_environment,
    )
    state_sessions = {
        state: [
            _cycle_session(
                environment,
                cycle=training_cycle,
                name=name,
            )
            for name in query.state_environments[state]
        ]
        for state in ("OO", "WO", "OW")
    }
    required = sorted(
        {
            train_square,
            test_square,
            target_test,
            *(
                session
                for sessions in state_sessions.values()
                for session in sessions
            ),
        }
    )
    cells = np.flatnonzero(
        np.logical_and.reduce([registered[item] for item in required])
    )
    if cells.size < minimum_cells:
        return None, {
            "query": query.identifier,
            "target_environment": ww_environment,
            "training_exposure": training_cycle + 1,
            "test_exposure": training_cycle + 2,
            "reason": "insufficient_common_registered_cells",
            "observed_cells": int(cells.size),
            "required_cells": minimum_cells,
        }
    support = common_support_bins(
        [occupancy[item] for item in required],
        query.bins,
        minimum_seconds=minimum_seconds,
    )
    if len(support) < minimum_bins:
        return None, {
            "query": query.identifier,
            "target_environment": ww_environment,
            "training_exposure": training_cycle + 1,
            "test_exposure": training_cycle + 2,
            "reason": "insufficient_common_occupancy_bins",
            "observed_bins": len(support),
            "required_bins": minimum_bins,
            "candidate_bins": len(query.bins),
            "cells": int(cells.size),
        }

    scores: dict[str, Any] = {}
    for mode, rate_function in RATE_MODES.items():
        def value(session: int) -> np.ndarray:
            return rate_function(
                rate[session],
                occupancy[session],
                cells,
                support,
            )

        square_train = value(train_square)
        condition = {
            state: np.mean(
                np.stack([value(session) for session in sessions]),
                axis=0,
            )
            for state, sessions in state_sessions.items()
        }
        wall_a = condition["WO"] - square_train
        wall_b = condition["OW"] - square_train
        additive = (
            condition["WO"]
            + condition["OW"]
            - condition["OO"]
            - square_train
        )
        target = value(target_test) - value(test_square)
        score = _score_predictors(
            additive=additive,
            wall_a_only=wall_a,
            wall_b_only=wall_b,
            target=target,
        )
        if score is None:
            return None, {
                "query": query.identifier,
                "target_environment": ww_environment,
                "training_exposure": training_cycle + 1,
                "test_exposure": training_cycle + 2,
                "reason": "nonfinite_predictor_score",
                "mode": mode,
                "cells": int(cells.size),
                "common_bins": len(support),
            }
        scores[mode] = score

    return (
        {
            "query": query.identifier,
            "seam_a": [query.seam_a.source, query.seam_a.target],
            "seam_b": [query.seam_b.source, query.seam_b.target],
            "target_environment": ww_environment,
            "training_exposure": training_cycle + 1,
            "test_exposure": training_cycle + 2,
            "training_square_session": train_square + 1,
            "test_square_session": test_square + 1,
            "test_target_session": target_test + 1,
            "target_training_session_excluded": (
                _cycle_session(
                    environment,
                    cycle=training_cycle,
                    name=ww_environment,
                )
                + 1
            ),
            "training_state_environments": {
                state: list(query.state_environments[state])
                for state in ("OO", "WO", "OW")
            },
            "training_state_sessions": {
                state: [session + 1 for session in sessions]
                for state, sessions in state_sessions.items()
            },
            "cells": int(cells.size),
            "candidate_bins": len(query.bins),
            "common_bins": len(support),
            "scores": scores,
        },
        None,
    )


def _summarize_records(
    records: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    if not records:
        return {
            "records": 0,
            "mean_additive_spearman_r": None,
            "mean_best_single_spearman_r": None,
            "mean_additive_minus_best_single_spearman_r": None,
            "positive_additive_minus_best_single_records": 0,
            "mean_additive_normalized_rmse": None,
            "mean_best_single_normalized_rmse": None,
        }
    additive_r = np.asarray(
        [
            record["scores"][mode]["predictors"][
                "interaction_free_additive"
            ]["spearman_r"]
            for record in records
        ]
    )
    best_single_r = np.asarray(
        [
            max(
                record["scores"][mode]["predictors"]["wall_a_only"][
                    "spearman_r"
                ],
                record["scores"][mode]["predictors"]["wall_b_only"][
                    "spearman_r"
                ],
            )
            for record in records
        ]
    )
    additive_rmse = np.asarray(
        [
            record["scores"][mode]["predictors"][
                "interaction_free_additive"
            ]["normalized_rmse"]
            for record in records
        ]
    )
    best_single_rmse = np.asarray(
        [
            min(
                record["scores"][mode]["predictors"]["wall_a_only"][
                    "normalized_rmse"
                ],
                record["scores"][mode]["predictors"]["wall_b_only"][
                    "normalized_rmse"
                ],
            )
            for record in records
        ]
    )
    difference = additive_r - best_single_r
    return {
        "records": len(records),
        "mean_additive_spearman_r": float(np.mean(additive_r)),
        "mean_best_single_spearman_r": float(np.mean(best_single_r)),
        "mean_additive_minus_best_single_spearman_r": float(
            np.mean(difference)
        ),
        "positive_additive_minus_best_single_records": int(
            np.count_nonzero(difference > 0)
        ),
        "mean_additive_normalized_rmse": float(np.mean(additive_rmse)),
        "mean_best_single_normalized_rmse": float(
            np.mean(best_single_rmse)
        ),
    }


def analyze_animal(
    path: Path,
    *,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    with Mat73Animal(path) as animal:
        environment = [
            animal.environment(session)
            for session in range(animal.n_sessions)
        ]
        blocked = [
            animal.blocked(session)
            for session in range(animal.n_sessions)
        ]
        geometry = _geometry_from_sessions(environment, blocked)
        queries = factorial_queries(geometry)
        rate = {
            session: animal.stored_rate_maps(
                session,
                smoothed=False,
            )
            for session in range(animal.n_sessions)
        }
        occupancy = {
            session: animal.sampling_map(session)
            for session in range(animal.n_sessions)
        }
        registered = {
            session: animal.registered_cells(session)
            for session in range(animal.n_sessions)
        }
        repetitions = (animal.n_sessions - 1) // 10

    records = []
    exclusions = []
    for training_cycle in range(repetitions - 1):
        for query in queries:
            for ww_environment in query.state_environments["WW"]:
                record, exclusion = _factorial_record(
                    query=query,
                    ww_environment=ww_environment,
                    training_cycle=training_cycle,
                    environment=environment,
                    rate=rate,
                    occupancy=occupancy,
                    registered=registered,
                    minimum_seconds=minimum_seconds,
                    minimum_bins=minimum_bins,
                    minimum_cells=minimum_cells,
                )
                if record is not None:
                    records.append(record)
                if exclusion is not None:
                    exclusions.append(exclusion)

    summaries = {
        mode: _summarize_records(records, mode=mode)
        for mode in RATE_MODES
    }
    exclusion_counts = {
        reason: sum(
            item["reason"] == reason
            for item in exclusions
        )
        for reason in sorted({item["reason"] for item in exclusions})
    }
    return (
        {
            "animal": path.name.removesuffix(".complete.mat"),
            "exposure_cycles": repetitions,
            "attempted_records": len(records) + len(exclusions),
            "eligible_records": len(records),
            "exclusion_counts": exclusion_counts,
            "exclusions": exclusions,
            "summaries": summaries,
            "records": records,
        },
        geometry,
    )


def cohort_summary(
    animals: list[dict[str, Any]],
) -> dict[str, Any]:
    output = {}
    for mode in RATE_MODES:
        eligible = [
            animal
            for animal in animals
            if animal["summaries"][mode]["records"] > 0
        ]
        differences = np.asarray(
            [
                animal["summaries"][mode][
                    "mean_additive_minus_best_single_spearman_r"
                ]
                for animal in eligible
            ],
            dtype=np.float64,
        )
        additive = np.asarray(
            [
                animal["summaries"][mode]["mean_additive_spearman_r"]
                for animal in eligible
            ],
            dtype=np.float64,
        )
        best = np.asarray(
            [
                animal["summaries"][mode][
                    "mean_best_single_spearman_r"
                ]
                for animal in eligible
            ],
            dtype=np.float64,
        )
        additive_rmse = np.asarray(
            [
                animal["summaries"][mode][
                    "mean_additive_normalized_rmse"
                ]
                for animal in eligible
            ],
            dtype=np.float64,
        )
        best_single_rmse = np.asarray(
            [
                animal["summaries"][mode][
                    "mean_best_single_normalized_rmse"
                ]
                for animal in eligible
            ],
            dtype=np.float64,
        )
        output[mode] = {
            "animals": len(eligible),
            "records": int(
                sum(animal["summaries"][mode]["records"] for animal in eligible)
            ),
            "animal_mean_additive_spearman_r": (
                float(np.mean(additive)) if eligible else None
            ),
            "animal_mean_best_single_spearman_r": (
                float(np.mean(best)) if eligible else None
            ),
            "animal_mean_additive_minus_best_single_spearman_r": (
                float(np.mean(differences)) if eligible else None
            ),
            "positive_animals": (
                int(np.count_nonzero(differences > 0)) if eligible else 0
            ),
            "animal_mean_additive_normalized_rmse": (
                float(np.mean(additive_rmse)) if eligible else None
            ),
            "animal_mean_best_single_normalized_rmse": (
                float(np.mean(best_single_rmse)) if eligible else None
            ),
            "animal_mean_additive_minus_best_single_normalized_rmse": (
                float(np.mean(additive_rmse - best_single_rmse))
                if eligible
                else None
            ),
            "animals_with_lower_additive_normalized_rmse": (
                int(np.count_nonzero(additive_rmse < best_single_rmse))
                if eligible
                else 0
            ),
            "animal_values": {
                animal["animal"]: animal["summaries"][mode][
                    "mean_additive_minus_best_single_spearman_r"
                ]
                for animal in eligible
            },
            "animal_normalized_rmse_values": {
                animal["animal"]: (
                    animal["summaries"][mode][
                        "mean_additive_normalized_rmse"
                    ]
                    - animal["summaries"][mode][
                        "mean_best_single_normalized_rmse"
                    ]
                )
                for animal in eligible
            },
        }
    return output


def main() -> None:
    argument = parse_arguments()
    paths = sorted(
        argument.data_dir.glob("QLAK-CA1-*.complete.mat")
    )
    if not paths:
        raise FileNotFoundError(
            f"no complete animal files found beneath {argument.data_dir}"
        )

    animals = []
    reference_geometry: dict[str, tuple[int, ...]] | None = None
    for path in paths:
        animal, geometry = analyze_animal(
            path,
            minimum_seconds=argument.minimum_seconds,
            minimum_bins=argument.minimum_bins,
            minimum_cells=argument.minimum_cells,
        )
        if reference_geometry is None:
            reference_geometry = geometry
        elif geometry != reference_geometry:
            raise ValueError(
                f"{path.name} uses a different environment-to-geometry map"
            )
        animals.append(animal)
        print(
            animal["animal"],
            {
                mode: value[
                    "mean_additive_minus_best_single_spearman_r"
                ]
                for mode, value in animal["summaries"].items()
            },
        )
    assert reference_geometry is not None
    audit = geometry_audit(reference_geometry)
    distinct_targets = len(audit["distinct_ww_target_environments"])
    report = {
        "status": "stopped_underidentified_compositional_branch",
        "question": (
            "Can profiles from locally isolated wall states predict a "
            "held-out two-wall CA1 response without using target neural "
            "rates?"
        ),
        "decision": {
            "general_whole_shape_test_identified": False,
            "limited_local_falsification_test_feasible": True,
            "candidate_general_compositional_contribution": False,
            "reason": (
                "Every held-out deformation row lies outside the row span "
                "of the remaining additive wall design. Only two complete "
                "local 2x2 wall pairs exist, and both use the same WW target "
                "environment, so the neural check cannot establish a "
                "portable compositional primitive."
            ),
        },
        "design": {
            "target_neural_rates_excluded_from_predictor_fitting": True,
            "target_training_ww_session_excluded": True,
            "target_geometry_labels_used_to_define_queries": True,
            "target_occupancy_and_registration_used_for_common_support": True,
            "training_and_test_exposures_nonoverlapping": True,
            "training_baseline": "square at start of training cycle",
            "test_baseline": "square after the test cycle",
            "training_and_test_baselines_nonoverlapping": True,
            "interaction_free_predictor": (
                "(mean WO + mean OW - mean OO) - training square"
            ),
            "test_target": "WW in next cycle - post-test square",
            "best_single_comparator": (
                "larger target-scored correlation of WO-square and "
                "OW-square; an oracle benchmark, not a fitted predictor"
            ),
            "normalized_rmse": (
                "RMSE divided by the RMS of the held-out test residual; "
                "no target-fitted intercept or scale"
            ),
            "predicted_object": (
                "registered-cell rate-change vector in the 3x3-bin "
                "intersection of two exact-location wall strips"
            ),
            "rate_modes": list(RATE_MODES),
            "inferential_unit": "animal",
        },
        "settings": {
            "minimum_seconds_per_common_bin": argument.minimum_seconds,
            "minimum_common_bins": argument.minimum_bins,
            "minimum_common_cells": argument.minimum_cells,
        },
        "geometry_identifiability": audit,
        "replication_scope": {
            "complete_local_factorials": audit[
                "complete_local_factorial_count"
            ],
            "distinct_ww_target_environments": distinct_targets,
            "independent_global_shape_replication": False,
        },
        "cohort": cohort_summary(animals),
        "animals": animals,
        "strongest_current_dataset_alternative": (
            "Treat the two bit-donut corner queries as a narrow "
            "falsification of interaction-free addition, while retaining "
            "target-rate-heldout exact-location single-fragment recurrence "
            "as the identified reusable-profile analysis. A general "
            "compositional test requires new geometries that independently "
            "factor wall A and wall B across multiple WW shapes."
        ),
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["cohort"], indent=2))


if __name__ == "__main__":
    main()
