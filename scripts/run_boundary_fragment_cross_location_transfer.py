"""Strict cross-location transfer test for wall-conditioned CA1 profiles.

For a wall at target oriented seam ``t`` in exposure cycle ``k + 1``, all
neural rates from the matching target geometry in training cycle ``k`` are
withheld.  At each different physical seam ``s``, the remaining training
geometries provide wall and open profiles at that same source location.

The primary source vector is the source wall-minus-open profile.  It predicts
the held-out target wall residual, computed only from the target test session
and its later square.  Thus predictor and evaluation neural sessions are
disjoint.  Source wall/open subtraction removes the source-place profile;
local square subtraction, an exact-target-location open predictor, and a
same-location exact wall-minus-open benchmark provide place controls.

The primary geometric subset uses source seams exactly one 25 cm grid step
away with the same signed wall normal as the target.  Opposite and orthogonal
normals at the same 25 cm distance, the source open profile, and the
exact-location wall effect are controls or benchmarks.  Source selection uses
labels, occupancy, and registration but never target neural values.  Reused
target sessions and source pairs are not independent; results are aggregated
first within target query and then within animal, without population
inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, NamedTuple

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


def _raw_local_rate(
    rate: np.ndarray,
    occupancy: np.ndarray,
    cells: np.ndarray,
    bins: tuple[tuple[int, int], ...],
) -> np.ndarray:
    del occupancy
    return local_cell_rate(rate, cells, bins)


RATE_MODES: dict[str, RateFunction] = {
    "raw_local_rate": _raw_local_rate,
    "global_rate_demeaned": globally_demeaned_local_cell_rate,
}
PRIMARY_DISTANCE_CM = 25.0


class TemplateSessions(NamedTuple):
    target_wall: tuple[int, ...]
    target_open: tuple[int, ...]
    source_wall: tuple[int, ...]
    source_open: tuple[int, ...]


PRIMARY_KEYS = (
    "one_grid_step_same_direction_source_effect_r_to_target_residual",
    "one_grid_step_same_direction_wall_minus_open_r",
    "one_grid_step_same_direction_wall_minus_exact_open_r",
    (
        "one_grid_step_same_direction_minus_exact_distance_nonsame_"
        "source_effect_r"
    ),
    "all_same_minus_opposite_source_effect_r",
    "all_same_minus_orthogonal_source_effect_r",
    (
        "one_grid_step_same_direction_minus_exact_location_"
        "effect_r_to_target_residual"
    ),
)


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
            / "boundary_fragment_cross_location_transfer.json"
        ),
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-bins", type=int, default=6)
    parser.add_argument("--minimum-cells", type=int, default=20)
    return parser.parse_args()


def _direction(seam: OrientedSeam) -> int:
    return seam.target - seam.source


def _orientation_relation(
    target: OrientedSeam,
    source: OrientedSeam,
) -> str:
    target_direction = _direction(target)
    source_direction = _direction(source)
    if source_direction == target_direction:
        return "same_signed_normal"
    if source_direction == -target_direction:
        return "opposite_normal"
    return "orthogonal_axis"


def _midpoint_distance_cm(
    first: OrientedSeam,
    second: OrientedSeam,
) -> float:
    first_start, first_end, _ = seam_frame(first)
    second_start, second_end, _ = seam_frame(second)
    first_midpoint = (first_start + first_end) / 2.0
    second_midpoint = (second_start + second_end) / 2.0
    return float(np.linalg.norm(first_midpoint - second_midpoint))


def _template_sessions(
    *,
    training_start: int,
    target_offset: int,
    target_seam: OrientedSeam,
    source_seam: OrientedSeam,
    blocked: list[tuple[int, ...]],
) -> TemplateSessions | None:
    """Return training templates while excluding the target geometry."""

    if source_seam.unordered == target_seam.unordered:
        return None
    target_training_session = training_start + target_offset
    candidates = tuple(
        training_start + offset
        for offset in range(1, 10)
        if offset != target_offset
    )

    def sessions(seam: OrientedSeam, state: SeamState) -> tuple[int, ...]:
        return tuple(
            session
            for session in candidates
            if seam_state(blocked[session], seam) is state
        )

    result = TemplateSessions(
        target_wall=sessions(target_seam, SeamState.WALL),
        target_open=sessions(target_seam, SeamState.OPEN),
        source_wall=sessions(source_seam, SeamState.WALL),
        source_open=sessions(source_seam, SeamState.OPEN),
    )
    if target_training_session in {
        session
        for values in result
        for session in values
    }:
        raise AssertionError("target training geometry entered a template")
    if any(not values for values in result):
        return None
    return result


def _mean_residual(
    sessions: tuple[int, ...],
    *,
    baseline: int,
    seam_bins: tuple[tuple[int, int], ...],
    cells: np.ndarray,
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    rate_function: RateFunction,
) -> np.ndarray:
    baseline_value = rate_function(
        rate[baseline],
        occupancy[baseline],
        cells,
        seam_bins,
    )
    value = [
        rate_function(
            rate[session],
            occupancy[session],
            cells,
            seam_bins,
        )
        - baseline_value
        for session in sessions
    ]
    return np.mean(np.stack(value), axis=0)


def _mode_metrics(
    *,
    source_wall: np.ndarray,
    source_open: np.ndarray,
    target_wall: np.ndarray,
    target_open: np.ndarray,
    exact_wall: np.ndarray,
) -> dict[str, float] | None:
    source_effect = source_wall - source_open
    target_effect = target_wall - target_open
    exact_effect = exact_wall - target_open
    correlation = {
        "source_effect_r_to_target_residual": spearman_correlation(
            source_effect,
            target_wall,
        ),
        "exact_effect_r_to_target_residual": spearman_correlation(
            exact_effect,
            target_wall,
        ),
        "shared_training_place_adjusted_effect_r_sensitivity": (
            spearman_correlation(
                source_effect,
                target_effect,
            )
        ),
        "shared_training_exact_place_adjusted_effect_r_sensitivity": (
            spearman_correlation(
                exact_effect,
                target_effect,
            )
        ),
        "source_effect_r_to_target_open_profile": spearman_correlation(
            source_effect,
            target_open,
        ),
        "source_wall_r_to_target_residual": spearman_correlation(
            source_wall,
            target_wall,
        ),
        "source_open_r_to_target_residual": spearman_correlation(
            source_open,
            target_wall,
        ),
        "target_exact_open_r_to_target_residual": spearman_correlation(
            target_open,
            target_wall,
        ),
        "source_open_r_to_target_open_profile": spearman_correlation(
            source_open,
            target_open,
        ),
    }
    if not all(np.isfinite(value) for value in correlation.values()):
        return None
    correlation["source_wall_minus_open_r_to_target_residual"] = (
        correlation["source_wall_r_to_target_residual"]
        - correlation["source_open_r_to_target_residual"]
    )
    correlation["source_wall_minus_target_exact_open_r"] = (
        correlation["source_wall_r_to_target_residual"]
        - correlation["target_exact_open_r_to_target_residual"]
    )
    correlation[
        "source_effect_minus_exact_effect_r_to_target_residual"
    ] = (
        correlation["source_effect_r_to_target_residual"]
        - correlation["exact_effect_r_to_target_residual"]
    )
    correlation["source_effect_specificity_over_target_open_r"] = (
        correlation["source_effect_r_to_target_residual"]
        - correlation["source_effect_r_to_target_open_profile"]
    )
    return correlation


def _query_common_cells(
    *,
    training_start: int,
    test_start: int,
    target_offset: int,
    registered: dict[int, np.ndarray],
) -> np.ndarray:
    """Return one cell set shared by every source comparison in a query."""

    target_training_session = training_start + target_offset
    required = (
        training_start,
        test_start + 10,
        test_start + target_offset,
        *(
            training_start + offset
            for offset in range(1, 10)
            if offset != target_offset
        ),
    )
    if target_training_session in required:
        raise AssertionError(
            "target training geometry entered the query cell mask"
        )
    return np.flatnonzero(
        np.logical_and.reduce(
            [registered[session] for session in required]
        )
    )


def _source_record(
    *,
    training_start: int,
    test_start: int,
    target_offset: int,
    target_seam: OrientedSeam,
    source_seam: OrientedSeam,
    blocked: list[tuple[int, ...]],
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    query_cells: np.ndarray,
    strips: dict[OrientedSeam, tuple[tuple[int, int], ...]],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
    rate_modes: dict[str, RateFunction] | None = None,
) -> dict[str, Any] | None:
    templates = _template_sessions(
        training_start=training_start,
        target_offset=target_offset,
        target_seam=target_seam,
        source_seam=source_seam,
        blocked=blocked,
    )
    if templates is None:
        return None
    target_test = test_start + target_offset
    test_baseline = test_start + 10
    training_baseline = training_start
    required = (
        training_baseline,
        test_baseline,
        target_test,
        *templates.target_wall,
        *templates.target_open,
        *templates.source_wall,
        *templates.source_open,
    )
    required = tuple(dict.fromkeys(required))
    cells = np.asarray(query_cells, dtype=np.int64)
    if any(
        not np.all(registered[session][cells])
        for session in required
    ):
        raise AssertionError(
            "query-level common cells do not cover a source template"
        )
    if cells.size < minimum_cells:
        return None

    target_support_sessions = (
        training_baseline,
        test_baseline,
        target_test,
        *templates.target_wall,
        *templates.target_open,
    )
    source_support_sessions = (
        training_baseline,
        *templates.source_wall,
        *templates.source_open,
    )
    target_support = common_support_bins(
        [
            occupancy[session]
            for session in target_support_sessions
        ],
        strips[target_seam],
        minimum_seconds=minimum_seconds,
    )
    source_support = common_support_bins(
        [
            occupancy[session]
            for session in source_support_sessions
        ],
        strips[source_seam],
        minimum_seconds=minimum_seconds,
    )
    if (
        len(target_support) < minimum_bins
        or len(source_support) < minimum_bins
    ):
        return None

    active_rate_modes = RATE_MODES if rate_modes is None else rate_modes
    by_mode: dict[str, dict[str, float]] = {}
    for mode, rate_function in active_rate_modes.items():
        source_wall = _mean_residual(
            templates.source_wall,
            baseline=training_baseline,
            seam_bins=source_support,
            cells=cells,
            rate=rate,
            occupancy=occupancy,
            rate_function=rate_function,
        )
        source_open = _mean_residual(
            templates.source_open,
            baseline=training_baseline,
            seam_bins=source_support,
            cells=cells,
            rate=rate,
            occupancy=occupancy,
            rate_function=rate_function,
        )
        target_open = _mean_residual(
            templates.target_open,
            baseline=training_baseline,
            seam_bins=target_support,
            cells=cells,
            rate=rate,
            occupancy=occupancy,
            rate_function=rate_function,
        )
        exact_wall = _mean_residual(
            templates.target_wall,
            baseline=training_baseline,
            seam_bins=target_support,
            cells=cells,
            rate=rate,
            occupancy=occupancy,
            rate_function=rate_function,
        )
        target_wall = (
            rate_function(
                rate[target_test],
                occupancy[target_test],
                cells,
                target_support,
            )
            - rate_function(
                rate[test_baseline],
                occupancy[test_baseline],
                cells,
                target_support,
            )
        )
        metrics = _mode_metrics(
            source_wall=source_wall,
            source_open=source_open,
            target_wall=target_wall,
            target_open=target_open,
            exact_wall=exact_wall,
        )
        if metrics is None:
            return None
        by_mode[mode] = metrics

    return {
        "source_seam": [source_seam.source, source_seam.target],
        "orientation_relation": _orientation_relation(
            target_seam,
            source_seam,
        ),
        "midpoint_distance_cm": _midpoint_distance_cm(
            target_seam,
            source_seam,
        ),
        "cells": int(cells.size),
        "target_bins": int(len(target_support)),
        "source_bins": int(len(source_support)),
        "target_wall_training_sessions": [
            session + 1 for session in templates.target_wall
        ],
        "target_open_training_sessions": [
            session + 1 for session in templates.target_open
        ],
        "source_wall_training_sessions": [
            session + 1 for session in templates.source_wall
        ],
        "source_open_training_sessions": [
            session + 1 for session in templates.source_open
        ],
        "metrics": by_mode,
    }


def _record_summary(
    records: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any] | None:
    if not records:
        return None
    metric_names = tuple(records[0]["metrics"][mode])
    output: dict[str, Any] = {
        "source_pairs": len(records),
        "median_cells": float(
            np.median([record["cells"] for record in records])
        ),
        "median_target_bins": float(
            np.median([record["target_bins"] for record in records])
        ),
        "median_source_bins": float(
            np.median([record["source_bins"] for record in records])
        ),
        "distance_cm_min_mean_max": [
            float(np.min(
                [record["midpoint_distance_cm"] for record in records]
            )),
            float(np.mean(
                [record["midpoint_distance_cm"] for record in records]
            )),
            float(np.max(
                [record["midpoint_distance_cm"] for record in records]
            )),
        ],
    }
    output.update(
        {
            metric: float(
                np.mean(
                    [
                        record["metrics"][mode][metric]
                        for record in records
                    ]
                )
            )
            for metric in metric_names
        }
    )
    return output


def _difference(
    first: dict[str, Any] | None,
    second: dict[str, Any] | None,
    metric: str,
) -> float | None:
    if first is None or second is None:
        return None
    return float(first[metric] - second[metric])


def _query_summary(
    source_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    same = [
        record
        for record in source_records
        if record["orientation_relation"] == "same_signed_normal"
    ]
    if not same:
        return None
    nearest_distance = min(
        record["midpoint_distance_cm"] for record in same
    )
    nearest_same = [
        record
        for record in same
        if np.isclose(
            record["midpoint_distance_cm"],
            nearest_distance,
        )
    ]
    one_grid_step_same = [
        record
        for record in same
        if np.isclose(
            record["midpoint_distance_cm"],
            PRIMARY_DISTANCE_CM,
        )
    ]
    opposite = [
        record
        for record in source_records
        if record["orientation_relation"] == "opposite_normal"
    ]
    orthogonal = [
        record
        for record in source_records
        if record["orientation_relation"] == "orthogonal_axis"
    ]
    distance_matched_opposite = [
        record
        for record in opposite
        if np.isclose(
            record["midpoint_distance_cm"],
            nearest_distance,
        )
    ]
    distance_matched_orthogonal = [
        record
        for record in orthogonal
        if np.isclose(
            record["midpoint_distance_cm"],
            nearest_distance,
        )
    ]
    distance_matched_nonsame = (
        distance_matched_opposite + distance_matched_orthogonal
    )
    one_grid_step_opposite = [
        record
        for record in opposite
        if np.isclose(
            record["midpoint_distance_cm"],
            PRIMARY_DISTANCE_CM,
        )
    ]
    one_grid_step_orthogonal = [
        record
        for record in orthogonal
        if np.isclose(
            record["midpoint_distance_cm"],
            PRIMARY_DISTANCE_CM,
        )
    ]
    one_grid_step_nonsame = (
        one_grid_step_opposite + one_grid_step_orthogonal
    )

    modes: dict[str, Any] = {}
    for mode in RATE_MODES:
        nearest = _record_summary(nearest_same, mode=mode)
        one_grid_step = _record_summary(
            one_grid_step_same,
            mode=mode,
        )
        all_same = _record_summary(same, mode=mode)
        opposite_summary = _record_summary(opposite, mode=mode)
        orthogonal_summary = _record_summary(orthogonal, mode=mode)
        matched_opposite = _record_summary(
            distance_matched_opposite,
            mode=mode,
        )
        matched_orthogonal = _record_summary(
            distance_matched_orthogonal,
            mode=mode,
        )
        matched_nonsame = _record_summary(
            distance_matched_nonsame,
            mode=mode,
        )
        exact_distance_nonsame = _record_summary(
            one_grid_step_nonsame,
            mode=mode,
        )
        if nearest is None or all_same is None:
            raise AssertionError("same-direction summaries disappeared")
        contrasts = {
            (
                "one_grid_step_same_direction_source_effect_r_to_"
                "target_residual"
            ): (
                one_grid_step["source_effect_r_to_target_residual"]
                if one_grid_step is not None
                else None
            ),
            "one_grid_step_same_direction_wall_minus_open_r": (
                one_grid_step[
                    "source_wall_minus_open_r_to_target_residual"
                ]
                if one_grid_step is not None
                else None
            ),
            "one_grid_step_same_direction_wall_minus_exact_open_r": (
                one_grid_step[
                    "source_wall_minus_target_exact_open_r"
                ]
                if one_grid_step is not None
                else None
            ),
            (
                "one_grid_step_same_direction_minus_exact_distance_"
                "nonsame_source_effect_r"
            ): _difference(
                one_grid_step,
                exact_distance_nonsame,
                "source_effect_r_to_target_residual",
            ),
            "all_same_minus_opposite_source_effect_r": _difference(
                all_same,
                opposite_summary,
                "source_effect_r_to_target_residual",
            ),
            "all_same_minus_orthogonal_source_effect_r": _difference(
                all_same,
                orthogonal_summary,
                "source_effect_r_to_target_residual",
            ),
            (
                "one_grid_step_same_direction_minus_exact_location_"
                "effect_r_to_target_residual"
            ): (
                one_grid_step[
                    "source_effect_minus_exact_effect_r_to_target_residual"
                ]
                if one_grid_step is not None
                else None
            ),
        }
        modes[mode] = {
            "nearest_same_direction_distance_cm": nearest_distance,
            "groups": {
                "nearest_same_direction": nearest,
                "one_grid_step_same_direction": one_grid_step,
                "all_same_direction": all_same,
                "all_opposite_direction": opposite_summary,
                "all_orthogonal_axis": orthogonal_summary,
                "distance_matched_opposite": matched_opposite,
                "distance_matched_orthogonal": matched_orthogonal,
                "distance_matched_nonsame": matched_nonsame,
                "one_grid_step_exact_distance_nonsame": (
                    exact_distance_nonsame
                ),
            },
            "contrasts": contrasts,
        }
    return {
        "eligible_source_pairs": len(source_records),
        "modes": modes,
    }


def _animal_mode_summary(
    queries: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "eligible_target_queries": len(queries),
    }
    for key in PRIMARY_KEYS:
        values = [
            query["modes"][mode]["contrasts"][key]
            for query in queries
            if query["modes"][mode]["contrasts"][key] is not None
        ]
        result[key] = {
            "target_queries": len(values),
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "positive_target_queries": int(
                np.count_nonzero(np.asarray(values) > 0)
            ),
        }
        grouped_means: dict[str, float | None] = {}
        grouping = {
            "prediction_record_equal": lambda query: (
                query["training_exposure"],
                query["test_exposure"],
                query["target_environment"],
                tuple(query["target_seam"]),
            ),
            "exposure_pair_equal": lambda query: (
                query["training_exposure"],
                query["test_exposure"],
            ),
            "target_environment_equal": lambda query: (
                query["target_environment"],
            ),
            "target_oriented_seam_equal": lambda query: (
                tuple(query["target_seam"]),
            ),
        }
        for label, group_key in grouping.items():
            groups: dict[tuple[Any, ...], list[float]] = {}
            for query in queries:
                value = query["modes"][mode]["contrasts"][key]
                if value is None:
                    continue
                groups.setdefault(group_key(query), []).append(value)
            grouped_means[label] = (
                float(
                    np.mean(
                        [
                            np.mean(group_values)
                            for group_values in groups.values()
                        ]
                    )
                )
                if groups
                else None
            )
        result[key]["aggregation_sensitivity_means"] = grouped_means
    distance = [
        query["modes"][mode][
            "nearest_same_direction_distance_cm"
        ]
        for query in queries
    ]
    result["nearest_same_direction_distance_cm_min_median_max"] = [
        float(np.min(distance)),
        float(np.median(distance)),
        float(np.max(distance)),
    ]
    result["total_eligible_source_pairs"] = int(
        sum(query["eligible_source_pairs"] for query in queries)
    )
    return result


def _compact_query_output(
    query: dict[str, Any],
) -> dict[str, Any]:
    """Drop redundant group metrics after animal summaries are computed."""

    compact = {
        key: value
        for key, value in query.items()
        if key != "modes"
    }
    support_fields = (
        "source_pairs",
        "median_cells",
        "median_target_bins",
        "median_source_bins",
        "distance_cm_min_mean_max",
        "source_effect_r_to_target_residual",
        "exact_effect_r_to_target_residual",
        "source_wall_r_to_target_residual",
        "source_open_r_to_target_residual",
        "target_exact_open_r_to_target_residual",
        "shared_training_place_adjusted_effect_r_sensitivity",
    )
    compact["modes"] = {}
    for mode, mode_value in query["modes"].items():
        compact_groups = {}
        for label, group in mode_value["groups"].items():
            compact_groups[label] = (
                {
                    field: group[field]
                    for field in support_fields
                }
                if group is not None
                else None
            )
        compact["modes"][mode] = {
            "nearest_same_direction_distance_cm": mode_value[
                "nearest_same_direction_distance_cm"
            ],
            "groups": compact_groups,
            "contrasts": mode_value["contrasts"],
        }
    return compact


def analyze_animal(
    path: Path,
    *,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> dict[str, Any]:
    seams = internal_seams()
    strips = {seam: seam_strip_bins(seam) for seam in seams}
    with Mat73Animal(path) as animal:
        environment = [
            animal.environment(session)
            for session in range(animal.n_sessions)
        ]
        blocked = [
            animal.blocked(session)
            for session in range(animal.n_sessions)
        ]
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

    queries: list[dict[str, Any]] = []
    for training_exposure in range(repetitions - 1):
        training_start = training_exposure * 10
        test_start = (training_exposure + 1) * 10
        for target_offset in range(1, 10):
            target_test = test_start + target_offset
            query_cells = _query_common_cells(
                training_start=training_start,
                test_start=test_start,
                target_offset=target_offset,
                registered=registered,
            )
            if query_cells.size < minimum_cells:
                continue
            for target_seam in seams:
                if (
                    seam_state(blocked[target_test], target_seam)
                    is not SeamState.WALL
                ):
                    continue
                source_records = []
                for source_seam in seams:
                    if source_seam.unordered == target_seam.unordered:
                        continue
                    record = _source_record(
                        training_start=training_start,
                        test_start=test_start,
                        target_offset=target_offset,
                        target_seam=target_seam,
                        source_seam=source_seam,
                        blocked=blocked,
                        rate=rate,
                        occupancy=occupancy,
                        registered=registered,
                        query_cells=query_cells,
                        strips=strips,
                        minimum_seconds=minimum_seconds,
                        minimum_bins=minimum_bins,
                        minimum_cells=minimum_cells,
                    )
                    if record is not None:
                        source_records.append(record)
                summary = _query_summary(source_records)
                if summary is None:
                    continue
                queries.append(
                    {
                        "training_exposure": training_exposure + 1,
                        "test_exposure": training_exposure + 2,
                        "target_environment": environment[target_test],
                        "withheld_training_session": (
                            training_start + target_offset + 1
                        ),
                        "target_test_session": target_test + 1,
                        "target_test_square_session": test_start + 11,
                        "target_seam": [
                            target_seam.source,
                            target_seam.target,
                        ],
                        "query_common_cells": int(query_cells.size),
                        **summary,
                    }
                )

    if not queries:
        raise ValueError(f"{path.name} has no eligible transfer queries")
    mode_summary = {
        mode: _animal_mode_summary(queries, mode=mode)
        for mode in RATE_MODES
    }
    return {
        "animal": path.name.removesuffix(".complete.mat"),
        "repetitions": repetitions,
        "modes": mode_summary,
        "queries": [
            _compact_query_output(query)
            for query in queries
        ],
    }


def _cohort_summary(
    animals: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in RATE_MODES:
        mode_result: dict[str, Any] = {}
        for key in PRIMARY_KEYS:
            values = {
                animal["animal"]: animal["modes"][mode][key]["mean"]
                for animal in animals
                if animal["modes"][mode][key]["mean"] is not None
            }
            array = np.asarray(list(values.values()), dtype=np.float64)
            mode_result[key] = {
                "animals": len(values),
                "positive_animals": int(
                    np.count_nonzero(array > 0)
                ),
                "animal_mean": (
                    float(np.mean(array)) if array.size else None
                ),
                "animal_median": (
                    float(np.median(array)) if array.size else None
                ),
                "animal_values": values,
            }
            aggregation_paths = {}
            for path_name in (
                "prediction_record_equal",
                "exposure_pair_equal",
                "target_environment_equal",
                "target_oriented_seam_equal",
            ):
                path_values = {
                    animal["animal"]: animal["modes"][mode][key][
                        "aggregation_sensitivity_means"
                    ][path_name]
                    for animal in animals
                    if animal["modes"][mode][key][
                        "aggregation_sensitivity_means"
                    ][path_name]
                    is not None
                }
                path_array = np.asarray(
                    list(path_values.values()),
                    dtype=np.float64,
                )
                aggregation_paths[path_name] = {
                    "animals": len(path_values),
                    "positive_animals": int(
                        np.count_nonzero(path_array > 0)
                    ),
                    "animal_mean": (
                        float(np.mean(path_array))
                        if path_array.size
                        else None
                    ),
                    "animal_values": path_values,
                }
            mode_result[key]["aggregation_paths"] = aggregation_paths
        mode_result["eligible_target_queries_by_animal"] = {
            animal["animal"]: animal["modes"][mode][
                "eligible_target_queries"
            ]
            for animal in animals
        }
        result[mode] = mode_result
    return result


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
    for path in paths:
        result = analyze_animal(
            path,
            minimum_seconds=argument.minimum_seconds,
            minimum_bins=argument.minimum_bins,
            minimum_cells=argument.minimum_cells,
        )
        animals.append(result)
        print(
            result["animal"],
            {
                mode: round(
                    value[
                        "one_grid_step_same_direction_source_effect_r_to_"
                        "target_residual"
                    ]["mean"],
                    4,
                )
                for mode, value in result["modes"].items()
            },
        )

    report = {
        "status": (
            "exploratory_target_rate_heldout_cross_location_transfer"
        ),
        "question": (
            "Does a wall-minus-open registered-cell profile learned at other "
            "physical seam locations predict a neural-rate-held-out target "
            "wall residual?"
        ),
        "design": {
            "target_geometry_training_neural_rates_excluded": True,
            "target_test_neural_rates_used_only_for_evaluation": True,
            "target_wall_label_occupancy_and_registration_test_aware": True,
            "predicted_object": (
                "one local square-residual strip rate per registered cell; "
                "not a full spatial map"
            ),
            "source_effect": (
                "mean source-location wall residual minus mean same-source-"
                "location open residual in the training exposure"
            ),
            "target_evaluation_vector": (
                "target-test wall local rate minus the later test-square "
                "local rate; no training neural vector enters this outcome"
            ),
            "predictor_evaluation_neural_sessions_disjoint": True,
            "square_baselines": (
                "training templates use the square before the training "
                "cycle; target test uses the square after the test cycle"
            ),
            "primary_source_selection": (
                "eligible different physical seams exactly one 25 cm grid "
                "step away with the same signed wall normal; selection uses "
                "geometry/support only"
            ),
            "place_controls": [
                "wall and open templates are contrasted at the same source",
                "an exact-location open profile is a competing predictor",
                "independent local square residuals are used at both places",
                "exact seam-midpoint distances are recorded and matched",
            ],
            "orientation_controls": [
                "opposite wall normal",
                "orthogonal wall axis",
                "non-same normals at the exact 25 cm source distance",
            ],
            "benchmark": (
                "exact-target-location wall-minus-open effect learned from "
                "other training geometries on identical cells/support"
            ),
            "shared_training_place_adjusted_sensitivity_not_primary": (
                "a source effect versus target-wall-minus-trained-open "
                "correlation is stored only as sensitivity because the "
                "source and open-place templates can reuse training sessions"
            ),
            "aggregation": (
                "equal-weight source seams within target query, then equal-"
                "weight target queries within animal, then descriptive "
                "equal-weight animal summary"
            ),
            "cell_support": (
                "one target-query-level global registered-cell intersection "
                "over the target test, later square, training square, and all "
                "eight non-target training sessions is forced identically "
                "across every source seam and orientation control"
            ),
            "inferential_unit": "animal",
            "population_inference_performed": False,
            "dependence_caveat": (
                "target queries, source pairs, seams, and cells are reused "
                "and are estimator internals rather than independent units"
            ),
        },
        "settings": {
            "minimum_seconds_per_common_bin": argument.minimum_seconds,
            "minimum_common_bins_per_location": argument.minimum_bins,
            "minimum_common_cells": argument.minimum_cells,
            "primary_source_midpoint_distance_cm": PRIMARY_DISTANCE_CM,
            "rate_modes": list(RATE_MODES),
        },
        "cohort_descriptive": _cohort_summary(animals),
        "animals": animals,
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["cohort_descriptive"], indent=2))


if __name__ == "__main__":
    main()
