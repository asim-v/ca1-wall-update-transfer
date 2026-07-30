"""Deterministic stored-map controls for the reusable-wall screen.

The controls in this file retain the discovery screen's fixed global-shape
pair, registered-cell, common-support, and animal-level aggregation.  They
change one feature at a time: distance from the seam, square subtraction,
spatial-bin weighting, or exact physical seam identity.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ca1_geometry.boundary_fragments import (  # noqa: E402
    common_support_bins,
    independent_square_residual_correlation,
    local_cell_rate,
    spearman_correlation,
)
from ca1_geometry.io import Mat73Animal  # noqa: E402
from ca1_geometry.seams import (  # noqa: E402
    OrientedSeam,
    SeamState,
    internal_seams,
    seam_state,
    seam_strip_bins,
)


RateEstimator = Callable[
    [np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, int], ...]],
    np.ndarray,
]


CONTROL_SPECS: dict[str, dict[str, Any]] = {
    "near_2p5_7p5_cm": {
        "depths_cm": (2.5, 7.5),
        "minimum_bins": 4,
        "estimator": "equal_bin",
        "square_subtraction": True,
        "comparison": "shared_wall_minus_same_seam_wall_open",
    },
    "far_17p5_22p5_cm": {
        "depths_cm": (17.5, 22.5),
        "minimum_bins": 4,
        "estimator": "equal_bin",
        "square_subtraction": True,
        "comparison": "shared_wall_minus_same_seam_wall_open",
    },
    "raw_target_no_square": {
        "depths_cm": (2.5, 7.5, 12.5),
        "minimum_bins": 6,
        "estimator": "equal_bin",
        "square_subtraction": False,
        "comparison": "shared_wall_minus_same_seam_wall_open",
    },
    "occupancy_weighted": {
        "depths_cm": (2.5, 7.5, 12.5),
        "minimum_bins": 6,
        "estimator": "within_strip_occupancy_weighted",
        "square_subtraction": True,
        "comparison": "shared_wall_minus_same_seam_wall_open",
    },
    "exact_vs_same_orientation_different_seam": {
        "depths_cm": (2.5, 7.5, 12.5),
        "minimum_bins": 6,
        "estimator": "equal_bin",
        "square_subtraction": True,
        "comparison": (
            "exact_wall_wall_minus_same_orientation_different_seam_wall_wall"
        ),
    },
}


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
            / "boundary_fragment_controls.json"
        ),
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-cells", type=int, default=20)
    parser.add_argument(
        "--minimum-environment-pairs",
        type=int,
        default=12,
        help="Sequence-level coverage gate; theoretical maximum is 20.",
    )
    return parser.parse_args()


def _equal_bin_rate(
    rate: np.ndarray,
    occupancy: np.ndarray,
    cells: np.ndarray,
    bins: tuple[tuple[int, int], ...],
) -> np.ndarray:
    del occupancy
    return local_cell_rate(rate, cells, bins)


def _within_strip_occupancy_weighted_rate(
    rate: np.ndarray,
    occupancy: np.ndarray,
    cells: np.ndarray,
    bins: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Average local rates using each session's measured dwell in the strip."""

    dwell = np.asarray(
        [occupancy[y_bin, x_bin] for y_bin, x_bin in bins],
        dtype=np.float64,
    )
    keep = np.isfinite(dwell) & (dwell > 0)
    if not np.any(keep):
        return np.full(cells.size, np.nan)
    value = np.stack(
        [rate[cells, y_bin, x_bin] for y_bin, x_bin in bins],
        axis=1,
    )
    valid = keep[None, :] & np.isfinite(value)
    numerator = np.sum(
        np.where(valid, value * dwell[None, :], 0.0),
        axis=1,
    )
    denominator = np.sum(
        np.where(valid, dwell[None, :], 0.0),
        axis=1,
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(cells.size, np.nan),
        where=denominator > 0,
    )


ESTIMATORS: dict[str, RateEstimator] = {
    "equal_bin": _equal_bin_rate,
    "within_strip_occupancy_weighted": (
        _within_strip_occupancy_weighted_rate
    ),
}


def _session_cache(
    animal: Mat73Animal,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    list[str],
    list[tuple[int, ...]],
]:
    rate = {
        session: animal.stored_rate_maps(session, smoothed=False)
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
    environment = [
        animal.environment(session) for session in range(animal.n_sessions)
    ]
    blocked = [
        animal.blocked(session) for session in range(animal.n_sessions)
    ]
    return rate, occupancy, registered, environment, blocked


def _common_cells(
    sessions: tuple[int, int, int, int],
    registered: dict[int, np.ndarray],
) -> np.ndarray:
    return np.flatnonzero(
        np.logical_and.reduce(
            [registered[session] for session in sessions]
        )
    )


def _same_seam_correlation(
    *,
    sessions: tuple[int, int, int, int],
    seam: OrientedSeam,
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    strip: dict[OrientedSeam, tuple[tuple[int, int], ...]],
    estimator: RateEstimator,
    subtract_square: bool,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> dict[str, Any] | None:
    pre_square, first_target, second_target, post_square = sessions
    support_sessions = (
        sessions
        if subtract_square
        else (first_target, second_target)
    )
    cells = np.flatnonzero(
        np.logical_and.reduce(
            [registered[session] for session in support_sessions]
        )
    )
    if cells.size < minimum_cells:
        return None
    support = common_support_bins(
        [occupancy[session] for session in support_sessions],
        strip[seam],
        minimum_seconds=minimum_seconds,
    )
    if len(support) < minimum_bins:
        return None
    value_sessions = (
        sessions
        if subtract_square
        else (first_target, second_target)
    )
    value = {
        session: estimator(
            rate[session],
            occupancy[session],
            cells,
            support,
        )
        for session in value_sessions
    }
    if subtract_square:
        result = independent_square_residual_correlation(
            value[first_target],
            value[second_target],
            value[pre_square],
            value[post_square],
        )
        correlation = result.mean
    else:
        correlation = spearman_correlation(
            value[first_target],
            value[second_target],
        )
    if not np.isfinite(correlation):
        return None
    return {
        "correlation": float(correlation),
        "cells": int(cells.size),
        "bins": int(len(support)),
    }


def _different_seam_correlation(
    *,
    sessions: tuple[int, int, int, int],
    first_seam: OrientedSeam,
    second_seam: OrientedSeam,
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    strip: dict[OrientedSeam, tuple[tuple[int, int], ...]],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> dict[str, Any] | None:
    """Correlate edits at two distinct but identically directed wall seams."""

    pre_square, first_target, second_target, post_square = sessions
    cells = _common_cells(sessions, registered)
    if cells.size < minimum_cells:
        return None
    first_support = common_support_bins(
        [occupancy[session] for session in sessions],
        strip[first_seam],
        minimum_seconds=minimum_seconds,
    )
    second_support = common_support_bins(
        [occupancy[session] for session in sessions],
        strip[second_seam],
        minimum_seconds=minimum_seconds,
    )
    if (
        len(first_support) < minimum_bins
        or len(second_support) < minimum_bins
    ):
        return None
    first_value = {
        session: local_cell_rate(rate[session], cells, first_support)
        for session in (pre_square, first_target, post_square)
    }
    second_value = {
        session: local_cell_rate(rate[session], cells, second_support)
        for session in (pre_square, second_target, post_square)
    }
    first_assignment = spearman_correlation(
        first_value[first_target] - first_value[pre_square],
        second_value[second_target] - second_value[post_square],
    )
    second_assignment = spearman_correlation(
        first_value[first_target] - first_value[post_square],
        second_value[second_target] - second_value[pre_square],
    )
    correlation = float(np.nanmean([first_assignment, second_assignment]))
    if not np.isfinite(correlation):
        return None
    return {
        "correlation": correlation,
        "cells": int(cells.size),
        "first_bins": int(len(first_support)),
        "second_bins": int(len(second_support)),
    }


def _seam_direction(seam: OrientedSeam) -> int:
    """Encode the oriented wall normal as -3, -1, +1, or +3."""

    return seam.target - seam.source


def _same_direction_different_pairs(
    first_walls: list[OrientedSeam],
    second_walls: list[OrientedSeam],
) -> list[tuple[OrientedSeam, OrientedSeam]]:
    """Return cross-location wall pairs with the same signed grid normal."""

    return [
        (first_seam, second_seam)
        for first_seam, second_seam in itertools.product(
            first_walls,
            second_walls,
        )
        if first_seam != second_seam
        and _seam_direction(first_seam) == _seam_direction(second_seam)
    ]


def _standard_pair_summary(
    *,
    sessions: tuple[int, int, int, int],
    blocked: list[tuple[int, ...]],
    seams: list[OrientedSeam],
    strip: dict[OrientedSeam, tuple[tuple[int, int], ...]],
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    estimator: RateEstimator,
    subtract_square: bool,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> dict[str, Any] | None:
    _, first_target, second_target, _ = sessions
    shared: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    for seam in seams:
        first_state = seam_state(blocked[first_target], seam)
        second_state = seam_state(blocked[second_target], seam)
        if (
            first_state is SeamState.WALL
            and second_state is SeamState.WALL
        ):
            destination = shared
        elif {first_state, second_state} == {
            SeamState.WALL,
            SeamState.OPEN,
        }:
            destination = changed
        else:
            continue
        result = _same_seam_correlation(
            sessions=sessions,
            seam=seam,
            rate=rate,
            occupancy=occupancy,
            registered=registered,
            strip=strip,
            estimator=estimator,
            subtract_square=subtract_square,
            minimum_seconds=minimum_seconds,
            minimum_bins=minimum_bins,
            minimum_cells=minimum_cells,
        )
        if result is not None:
            destination.append(result)
    if not shared or not changed:
        return None
    shared_value = np.asarray([item["correlation"] for item in shared])
    changed_value = np.asarray([item["correlation"] for item in changed])
    return {
        "shared_wall_mean": float(np.mean(shared_value)),
        "changed_wall_mean": float(np.mean(changed_value)),
        "contrast": float(
            np.mean(shared_value) - np.mean(changed_value)
        ),
        "shared_wall_seams": int(shared_value.size),
        "changed_wall_seams": int(changed_value.size),
        "median_cells": float(
            np.median([item["cells"] for item in shared + changed])
        ),
        "median_bins": float(
            np.median([item["bins"] for item in shared + changed])
        ),
    }


def _different_seam_pair_summary(
    *,
    sessions: tuple[int, int, int, int],
    blocked: list[tuple[int, ...]],
    seams: list[OrientedSeam],
    strip: dict[OrientedSeam, tuple[tuple[int, int], ...]],
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> dict[str, Any] | None:
    """Contrast exact shared seams with parallel walls at other locations."""

    _, first_target, second_target, _ = sessions
    first_walls = [
        seam
        for seam in seams
        if seam_state(blocked[first_target], seam) is SeamState.WALL
    ]
    second_walls = [
        seam
        for seam in seams
        if seam_state(blocked[second_target], seam) is SeamState.WALL
    ]
    shared_seams = [
        seam
        for seam in first_walls
        if seam_state(blocked[second_target], seam) is SeamState.WALL
    ]
    exact: dict[OrientedSeam, dict[str, Any]] = {}
    for seam in shared_seams:
        result = _same_seam_correlation(
            sessions=sessions,
            seam=seam,
            rate=rate,
            occupancy=occupancy,
            registered=registered,
            strip=strip,
            estimator=_equal_bin_rate,
            subtract_square=True,
            minimum_seconds=minimum_seconds,
            minimum_bins=minimum_bins,
            minimum_cells=minimum_cells,
        )
        if result is not None:
            exact[seam] = result

    mismatch: list[dict[str, Any]] = []
    for first_seam, second_seam in _same_direction_different_pairs(
        first_walls,
        second_walls,
    ):
        result = _different_seam_correlation(
            sessions=sessions,
            first_seam=first_seam,
            second_seam=second_seam,
            rate=rate,
            occupancy=occupancy,
            registered=registered,
            strip=strip,
            minimum_seconds=minimum_seconds,
            minimum_bins=minimum_bins,
            minimum_cells=minimum_cells,
        )
        if result is not None:
            mismatch.append(result)
    if not mismatch or not exact:
        return None
    exact_value = np.asarray(
        [item["correlation"] for item in exact.values()]
    )
    mismatch_value = np.asarray(
        [item["correlation"] for item in mismatch]
    )
    return {
        "exact_seam_mean": float(np.mean(exact_value)),
        "different_seam_mean": float(np.mean(mismatch_value)),
        "contrast": float(
            np.mean(exact_value) - np.mean(mismatch_value)
        ),
        "exact_shared_seams": int(exact_value.size),
        "ordered_different_seam_comparisons": int(mismatch_value.size),
        "median_cells": float(
            np.median([item["cells"] for item in mismatch])
        ),
    }


def _sequence_control(
    *,
    name: str,
    spec: dict[str, Any],
    exposure: int,
    pre_square: int,
    post_square: int,
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    environment: list[str],
    blocked: list[tuple[int, ...]],
    seams: list[OrientedSeam],
    minimum_seconds: float,
    minimum_cells: int,
    minimum_environment_pairs: int,
) -> dict[str, Any]:
    strip = {
        seam: seam_strip_bins(seam, depths_cm=spec["depths_cm"])
        for seam in seams
    }
    pair_records: list[dict[str, Any]] = []
    targets = range(pre_square + 1, post_square)
    for first_target, second_target in itertools.combinations(targets, 2):
        sessions = (
            pre_square,
            first_target,
            second_target,
            post_square,
        )
        if name == "exact_vs_same_orientation_different_seam":
            summary = _different_seam_pair_summary(
                sessions=sessions,
                blocked=blocked,
                seams=seams,
                strip=strip,
                rate=rate,
                occupancy=occupancy,
                registered=registered,
                minimum_seconds=minimum_seconds,
                minimum_bins=spec["minimum_bins"],
                minimum_cells=minimum_cells,
            )
        else:
            summary = _standard_pair_summary(
                sessions=sessions,
                blocked=blocked,
                seams=seams,
                strip=strip,
                rate=rate,
                occupancy=occupancy,
                registered=registered,
                estimator=ESTIMATORS[spec["estimator"]],
                subtract_square=spec["square_subtraction"],
                minimum_seconds=minimum_seconds,
                minimum_bins=spec["minimum_bins"],
                minimum_cells=minimum_cells,
            )
        if summary is not None:
            pair_records.append(
                {
                    "first_environment": environment[first_target],
                    "second_environment": environment[second_target],
                    **summary,
                }
            )
    value = (
        float(np.mean([item["contrast"] for item in pair_records]))
        if pair_records
        else None
    )
    return {
        "exposure": exposure,
        "pre_square_session": pre_square + 1,
        "post_square_session": post_square + 1,
        "comparable_environment_pairs": len(pair_records),
        "coverage_eligible": (
            len(pair_records) >= minimum_environment_pairs
        ),
        "contrast": value,
        "environment_pairs": pair_records,
    }


def analyze_animal(
    path: Path,
    *,
    minimum_seconds: float,
    minimum_cells: int,
    minimum_environment_pairs: int,
) -> dict[str, Any]:
    seams = internal_seams()
    with Mat73Animal(path) as animal:
        rate, occupancy, registered, environment, blocked = _session_cache(
            animal
        )
        repetitions = (animal.n_sessions - 1) // 10
        controls: dict[str, dict[str, Any]] = {}
        for name, spec in CONTROL_SPECS.items():
            sequences = [
                _sequence_control(
                    name=name,
                    spec=spec,
                    exposure=repetition + 1,
                    pre_square=repetition * 10,
                    post_square=repetition * 10 + 10,
                    rate=rate,
                    occupancy=occupancy,
                    registered=registered,
                    environment=environment,
                    blocked=blocked,
                    seams=seams,
                    minimum_seconds=minimum_seconds,
                    minimum_cells=minimum_cells,
                    minimum_environment_pairs=minimum_environment_pairs,
                )
                for repetition in range(repetitions)
            ]
            finite = [
                item["contrast"]
                for item in sequences
                if item["contrast"] is not None
            ]
            eligible = [
                item["contrast"]
                for item in sequences
                if item["coverage_eligible"]
                and item["contrast"] is not None
            ]
            controls[name] = {
                "all_sequence_mean": (
                    float(np.mean(finite)) if finite else None
                ),
                "eligible_sequence_mean": (
                    float(np.mean(eligible)) if eligible else None
                ),
                "finite_sequences": len(finite),
                "coverage_eligible_sequences": len(eligible),
                "total_comparable_environment_pairs": int(
                    sum(
                        item["comparable_environment_pairs"]
                        for item in sequences
                    )
                ),
                "sequences": sequences,
            }
    return {
        "animal": path.name.removesuffix(".complete.mat"),
        "sessions": len(environment),
        "controls": controls,
    }


def _cohort_summary(animals: list[dict[str, Any]]) -> dict[str, Any]:
    cohort: dict[str, Any] = {}
    for name in CONTROL_SPECS:
        all_value = [
            animal["controls"][name]["all_sequence_mean"]
            for animal in animals
            if animal["controls"][name]["all_sequence_mean"] is not None
        ]
        eligible_value = [
            animal["controls"][name]["eligible_sequence_mean"]
            for animal in animals
            if animal["controls"][name]["eligible_sequence_mean"] is not None
        ]
        cohort[name] = {
            "animals_with_finite_sequences": len(all_value),
            "positive_all_sequence_animals": int(
                np.count_nonzero(np.asarray(all_value) > 0)
            ),
            "all_sequence_animal_mean": (
                float(np.mean(all_value)) if all_value else None
            ),
            "coverage_eligible_animals": len(eligible_value),
            "positive_coverage_eligible_animals": int(
                np.count_nonzero(np.asarray(eligible_value) > 0)
            ),
            "eligible_animal_mean": (
                float(np.mean(eligible_value))
                if eligible_value
                else None
            ),
            "total_comparable_environment_pairs": int(
                sum(
                    animal["controls"][name][
                        "total_comparable_environment_pairs"
                    ]
                    for animal in animals
                )
            ),
        }
    return cohort


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    argument = parse_arguments()
    paths = sorted(
        argument.data_dir.glob("QLAK-CA1-*.complete.mat")
    )
    if not paths:
        raise FileNotFoundError(
            f"no complete animal files found beneath {argument.data_dir}"
        )
    animals = [
        analyze_animal(
            path,
            minimum_seconds=argument.minimum_seconds,
            minimum_cells=argument.minimum_cells,
            minimum_environment_pairs=argument.minimum_environment_pairs,
        )
        for path in paths
    ]
    report = {
        "status": "descriptive_stress_tests_not_confirmatory_inference",
        "question": (
            "Does the exact-location wall-conditioned profile advantage "
            "survive local-distance, baseline, weighting, and "
            "physical-location controls?"
        ),
        "shared_design": {
            "rate_maps": "released_unsmoothed_5_cm_event_probability",
            "candidate_seam_length_cm": 25.0,
            "minimum_seconds_per_common_bin": argument.minimum_seconds,
            "minimum_common_cells": argument.minimum_cells,
            "minimum_environment_pairs_per_sequence": (
                argument.minimum_environment_pairs
            ),
            "correlation": "spearman",
            "baseline_when_used": (
                "mean of opposite independent pre/post-square assignments"
            ),
            "pair_aggregation": (
                "mean seam correlation within label, label contrast within "
                "a fixed pair of global environments, then equal-weight mean "
                "over comparable environment pairs within exposure"
            ),
            "different_seam_matching": (
                "distinct physical seams that are walls in their respective "
                "target sessions and have the same signed grid-normal "
                "direction; orthogonal and opposite-facing walls are excluded"
            ),
            "animal_aggregation": (
                "equal-weight mean over finite exposure-level contrasts; "
                "coverage-gated means are reported separately"
            ),
            "inferential_unit": (
                "animal; seam and environment-pair records are descriptive"
            ),
        },
        "control_specs": CONTROL_SPECS,
        "cohort": _cohort_summary(animals),
        "animals": animals,
    }
    report = _json_safe(report)
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    for animal in report["animals"]:
        values = {
            name: item["all_sequence_mean"]
            for name, item in animal["controls"].items()
        }
        print(animal["animal"], json.dumps(values, sort_keys=True))
    print(json.dumps(report["cohort"], indent=2))


if __name__ == "__main__":
    main()
