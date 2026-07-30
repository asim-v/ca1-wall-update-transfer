"""Mirror-lag open control for one-step cross-location wall transfer.

For each preselected source/target pair in the cross-location analysis, this
audit reflects the wall target across the source seam.  The reflected control
must be a valid oriented seam with the same signed normal, lie one 25 cm grid
step from the source along the same translation axis, and be OPEN in the same
held-out target session in which the target seam is a WALL.

The source wall-minus-open predictor is evaluated against local residuals at
both positions in that one held-out session.  Target and mirror-control strips
use translated, one-to-one matched bins that pass the same occupancy rule at
both locations in the target session and its later square.  One registered
cell set is shared by the source predictor and both evaluation vectors.

This is a post-outcome spatial falsification.  It asks whether the reported
one-step transfer is more wall-specific than generic, layout-matched nearby
population continuity.  It cannot randomize absolute position, geometry,
behavior, or acquisition order.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_boundary_fragment_cross_location_transfer as transfer  # noqa: E402
from ca1_geometry.boundary_fragments import (  # noqa: E402
    common_support_bins,
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


METRICS = (
    "source_effect_r_to_wall_target",
    "source_effect_r_to_mirror_open",
    "wall_minus_mirror_open_correlation_advantage",
    "source_effect_r_to_within_session_wall_minus_mirror_open",
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
            / "boundary_fragment_cross_location_mirror_open.json"
        ),
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-bins", type=int, default=6)
    parser.add_argument("--minimum-cells", type=int, default=20)
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Print label-only coverage without loading neural maps.",
    )
    return parser.parse_args()


def _midpoint(seam: OrientedSeam) -> np.ndarray:
    start, end, _normal = seam_frame(seam)
    return (start + end) / 2.0


def _translation_axis(
    source: OrientedSeam,
    target: OrientedSeam,
) -> str:
    """Classify an exact one-step translation in the source seam frame."""

    if (
        transfer._orientation_relation(target, source)
        != "same_signed_normal"
    ):
        raise ValueError("source and target must have the same signed normal")
    displacement = _midpoint(target) - _midpoint(source)
    if not np.isclose(np.linalg.norm(displacement), 25.0):
        raise ValueError("source and target must be exactly one grid step")
    _start, _end, normal = seam_frame(source)
    tangent = np.asarray([-normal[1], normal[0]])
    normal_cm = float(abs(displacement @ normal))
    tangent_cm = float(abs(displacement @ tangent))
    if np.isclose(normal_cm, 25.0) and np.isclose(tangent_cm, 0.0):
        return "normal"
    if np.isclose(normal_cm, 0.0) and np.isclose(tangent_cm, 25.0):
        return "tangential"
    raise AssertionError("one-step displacement is not axis aligned")


def _mirror_control_seam(
    source: OrientedSeam,
    target: OrientedSeam,
    *,
    seams: tuple[OrientedSeam, ...],
) -> OrientedSeam | None:
    """Reflect ``target`` across ``source`` while preserving orientation."""

    if source == target:
        return None
    if (
        transfer._orientation_relation(target, source)
        != "same_signed_normal"
        or not np.isclose(
            transfer._midpoint_distance_cm(source, target),
            transfer.PRIMARY_DISTANCE_CM,
        )
    ):
        return None
    reflected_midpoint = 2.0 * _midpoint(source) - _midpoint(target)
    matches = [
        seam
        for seam in seams
        if transfer._orientation_relation(source, seam)
        == "same_signed_normal"
        and np.allclose(_midpoint(seam), reflected_midpoint)
    ]
    if len(matches) > 1:
        raise AssertionError("mirror midpoint has multiple oriented seams")
    if not matches:
        return None
    control = matches[0]
    if control in (source, target):
        raise AssertionError("mirror control did not change physical seam")
    if not np.isclose(
        transfer._midpoint_distance_cm(source, control),
        transfer.PRIMARY_DISTANCE_CM,
    ):
        raise AssertionError("mirror control does not preserve source lag")
    if _translation_axis(source, control) != _translation_axis(
        source,
        target,
    ):
        raise AssertionError("mirror control changed translation axis")
    return control


def _strip_minimum_distance_cm(
    first: OrientedSeam,
    second: OrientedSeam,
) -> float:
    return float(
        min(
            np.linalg.norm(np.subtract(a, b)) * 5.0
            for a in seam_strip_bins(first)
            for b in seam_strip_bins(second)
        )
    )


def _translated_bin_pairs(
    target: OrientedSeam,
    control: OrientedSeam,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Pair corresponding strip bins under the exact seam translation."""

    if transfer._orientation_relation(target, control) != (
        "same_signed_normal"
    ):
        raise ValueError("target and control must share signed orientation")
    displacement = (_midpoint(control) - _midpoint(target)) / 5.0
    x_shift = int(round(float(displacement[0])))
    row_shift = int(round(float(displacement[1])))
    if not np.allclose(displacement, [x_shift, row_shift]):
        raise ValueError("seam translation is not aligned to 5 cm bins")
    control_bins = set(seam_strip_bins(control))
    pairs = []
    for row, column in seam_strip_bins(target):
        translated = (row + row_shift, column + x_shift)
        if translated not in control_bins:
            raise AssertionError("translated strip bin is absent")
        pairs.append(((row, column), translated))
    if len(pairs) != len(control_bins):
        raise AssertionError("strip translation is not one-to-one")
    return tuple(pairs)


def _matched_evaluation_bins(
    *,
    occupancy: dict[int, np.ndarray],
    sessions: tuple[int, ...],
    target: OrientedSeam,
    control: OrientedSeam,
    minimum_seconds: float,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    """Keep paired relative bins supported at both evaluation locations."""

    pairs = _translated_bin_pairs(target, control)
    target_bins: list[tuple[int, int]] = []
    control_bins: list[tuple[int, int]] = []
    for target_bin, control_bin in pairs:
        values = []
        for session in sessions:
            values.extend(
                (
                    occupancy[session][target_bin],
                    occupancy[session][control_bin],
                )
            )
        array = np.asarray(values, dtype=np.float64)
        if (
            np.isfinite(array).all()
            and float(np.min(array)) >= minimum_seconds
        ):
            target_bins.append(target_bin)
            control_bins.append(control_bin)
    return tuple(target_bins), tuple(control_bins)


def _finite_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float | None:
    value = spearman_correlation(first, second)
    return float(value) if np.isfinite(value) else None


def _mode_metrics(
    *,
    source_effect: np.ndarray,
    wall_target: np.ndarray,
    mirror_open: np.ndarray,
) -> dict[str, float] | None:
    wall_r = _finite_correlation(source_effect, wall_target)
    open_r = _finite_correlation(source_effect, mirror_open)
    difference_r = _finite_correlation(
        source_effect,
        wall_target - mirror_open,
    )
    if wall_r is None or open_r is None or difference_r is None:
        return None
    return {
        "source_effect_r_to_wall_target": wall_r,
        "source_effect_r_to_mirror_open": open_r,
        "wall_minus_mirror_open_correlation_advantage": (
            wall_r - open_r
        ),
        "source_effect_r_to_within_session_wall_minus_mirror_open": (
            difference_r
        ),
    }


def _paired_record(
    *,
    training_start: int,
    test_start: int,
    target_offset: int,
    target_seam: OrientedSeam,
    source_seam: OrientedSeam,
    control_seam: OrientedSeam,
    blocked: list[tuple[int, ...]],
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    strips: dict[OrientedSeam, tuple[tuple[int, int], ...]],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Compute one source/wall/mirrored-open comparison."""

    target_test = test_start + target_offset
    test_baseline = test_start + 10
    training_baseline = training_start
    if seam_state(blocked[target_test], target_seam) is not SeamState.WALL:
        raise AssertionError("target seam is not a wall")
    if seam_state(blocked[target_test], control_seam) is not SeamState.OPEN:
        raise AssertionError("mirror control seam is not open")
    templates = transfer._template_sessions(
        training_start=training_start,
        target_offset=target_offset,
        target_seam=target_seam,
        source_seam=source_seam,
        blocked=blocked,
    )
    if templates is None:
        return None, "missing_training_template"
    cells = transfer._query_common_cells(
        training_start=training_start,
        test_start=test_start,
        target_offset=target_offset,
        registered=registered,
    )
    if cells.size < minimum_cells:
        return None, "insufficient_common_cells"
    source_sessions = (
        training_baseline,
        *templates.source_wall,
        *templates.source_open,
    )
    source_bins = common_support_bins(
        [occupancy[session] for session in source_sessions],
        strips[source_seam],
        minimum_seconds=minimum_seconds,
    )
    if len(source_bins) < minimum_bins:
        return None, "insufficient_source_bins"
    target_bins, control_bins = _matched_evaluation_bins(
        occupancy=occupancy,
        sessions=(target_test, test_baseline),
        target=target_seam,
        control=control_seam,
        minimum_seconds=minimum_seconds,
    )
    if len(target_bins) < minimum_bins:
        return None, "insufficient_paired_evaluation_bins"
    if len(target_bins) != len(control_bins):
        raise AssertionError("target and control support are not paired")

    modes: dict[str, dict[str, float]] = {}
    for mode, rate_function in transfer.RATE_MODES.items():
        source_wall = transfer._mean_residual(
            templates.source_wall,
            baseline=training_baseline,
            seam_bins=source_bins,
            cells=cells,
            rate=rate,
            occupancy=occupancy,
            rate_function=rate_function,
        )
        source_open = transfer._mean_residual(
            templates.source_open,
            baseline=training_baseline,
            seam_bins=source_bins,
            cells=cells,
            rate=rate,
            occupancy=occupancy,
            rate_function=rate_function,
        )
        source_effect = source_wall - source_open
        wall_target = (
            rate_function(
                rate[target_test],
                occupancy[target_test],
                cells,
                target_bins,
            )
            - rate_function(
                rate[test_baseline],
                occupancy[test_baseline],
                cells,
                target_bins,
            )
        )
        mirror_open = (
            rate_function(
                rate[target_test],
                occupancy[target_test],
                cells,
                control_bins,
            )
            - rate_function(
                rate[test_baseline],
                occupancy[test_baseline],
                cells,
                control_bins,
            )
        )
        metrics = _mode_metrics(
            source_effect=source_effect,
            wall_target=wall_target,
            mirror_open=mirror_open,
        )
        if metrics is None:
            return None, f"nonfinite_{mode}_metric"
        modes[mode] = metrics

    source_target_axis = _translation_axis(source_seam, target_seam)
    return (
        {
            "source_seam": [source_seam.source, source_seam.target],
            "target_wall_seam": [
                target_seam.source,
                target_seam.target,
            ],
            "mirror_open_seam": [
                control_seam.source,
                control_seam.target,
            ],
            "translation_axis": source_target_axis,
            "source_to_wall_midpoint_distance_cm": (
                transfer._midpoint_distance_cm(
                    source_seam,
                    target_seam,
                )
            ),
            "source_to_open_midpoint_distance_cm": (
                transfer._midpoint_distance_cm(
                    source_seam,
                    control_seam,
                )
            ),
            "source_to_wall_strip_minimum_bin_center_distance_cm": (
                _strip_minimum_distance_cm(source_seam, target_seam)
            ),
            "source_to_open_strip_minimum_bin_center_distance_cm": (
                _strip_minimum_distance_cm(source_seam, control_seam)
            ),
            "target_to_control_midpoint_distance_cm": (
                transfer._midpoint_distance_cm(
                    target_seam,
                    control_seam,
                )
            ),
            "target_wall_state": "wall",
            "mirror_control_state": "open",
            "same_held_out_target_session": True,
            "identical_registered_cells": True,
            "paired_relative_evaluation_bins": True,
            "cells": int(cells.size),
            "source_bins": len(source_bins),
            "paired_evaluation_bins": len(target_bins),
            "source_wall_training_sessions": [
                session + 1 for session in templates.source_wall
            ],
            "source_open_training_sessions": [
                session + 1 for session in templates.source_open
            ],
            "modes": modes,
        },
        None,
    )


def _geometry_candidates(
    *,
    environment: list[str],
    blocked: list[tuple[int, ...]],
    repetitions: int,
    seams: tuple[OrientedSeam, ...],
) -> dict[str, Any]:
    """Count label-only mirror-control coverage without reading neural maps."""

    del environment
    counts: Counter[str] = Counter()
    queries: set[tuple[int, int, OrientedSeam]] = set()
    target_environments: Counter[str] = Counter()
    for training_exposure in range(repetitions - 1):
        training_start = training_exposure * 10
        test_start = (training_exposure + 1) * 10
        for target_offset in range(1, 10):
            target_test = test_start + target_offset
            for target_seam in seams:
                if (
                    seam_state(blocked[target_test], target_seam)
                    is not SeamState.WALL
                ):
                    continue
                for source_seam in seams:
                    if (
                        transfer._orientation_relation(
                            target_seam,
                            source_seam,
                        )
                        != "same_signed_normal"
                        or not np.isclose(
                            transfer._midpoint_distance_cm(
                                target_seam,
                                source_seam,
                            ),
                            transfer.PRIMARY_DISTANCE_CM,
                        )
                    ):
                        continue
                    counts["primary_geometry_pairs"] += 1
                    templates = transfer._template_sessions(
                        training_start=training_start,
                        target_offset=target_offset,
                        target_seam=target_seam,
                        source_seam=source_seam,
                        blocked=blocked,
                    )
                    if templates is None:
                        counts["missing_training_template"] += 1
                        continue
                    counts["template_eligible_primary_pairs"] += 1
                    control = _mirror_control_seam(
                        source_seam,
                        target_seam,
                        seams=seams,
                    )
                    if control is None:
                        counts["no_internal_mirror"] += 1
                        continue
                    counts["internal_mirror_pairs"] += 1
                    if (
                        seam_state(blocked[target_test], control)
                        is not SeamState.OPEN
                    ):
                        counts["mirror_not_open"] += 1
                        continue
                    counts["mirror_open_pairs"] += 1
                    query = (
                        training_exposure,
                        target_offset,
                        target_seam,
                    )
                    queries.add(query)
                    target_environments[
                        str(target_test)
                    ] += 1
    return {
        **dict(counts),
        "mirror_open_target_queries": len(queries),
        "target_session_ids_with_pair_counts": dict(
            sorted(target_environments.items())
        ),
    }


def geometry_census(path: Path) -> dict[str, Any]:
    seams = tuple(internal_seams())
    with Mat73Animal(path) as animal:
        environment = [
            animal.environment(session)
            for session in range(animal.n_sessions)
        ]
        blocked = [
            animal.blocked(session)
            for session in range(animal.n_sessions)
        ]
        repetitions = (animal.n_sessions - 1) // 10
    return {
        "animal": path.name.removesuffix(".complete.mat"),
        **_geometry_candidates(
            environment=environment,
            blocked=blocked,
            repetitions=repetitions,
            seams=seams,
        ),
    }


def _query_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_pairs": len(records),
        "translation_axes": dict(
            Counter(record["translation_axis"] for record in records)
        ),
        "modes": {
            mode: {
                metric: float(
                    np.mean(
                        [
                            record["modes"][mode][metric]
                            for record in records
                        ]
                    )
                )
                for metric in METRICS
            }
            for mode in transfer.RATE_MODES
        },
    }


def _animal_summary(
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "eligible_target_queries": len(queries),
        "eligible_source_wall_open_mirror_triplets": int(
            sum(query["summary"]["source_pairs"] for query in queries)
        ),
        "translation_axes": dict(
            sum(
                (
                    Counter(query["summary"]["translation_axes"])
                    for query in queries
                ),
                Counter(),
            )
        ),
        "cells_min_median_max": None,
        "paired_bins_min_median_max": None,
        "modes": {},
    }
    records = [
        record
        for query in queries
        for record in query["records"]
    ]
    if records:
        cells = np.asarray([record["cells"] for record in records])
        bins = np.asarray(
            [record["paired_evaluation_bins"] for record in records]
        )
        output["cells_min_median_max"] = [
            int(np.min(cells)),
            float(np.median(cells)),
            int(np.max(cells)),
        ]
        output["paired_bins_min_median_max"] = [
            int(np.min(bins)),
            float(np.median(bins)),
            int(np.max(bins)),
        ]
    for mode in transfer.RATE_MODES:
        output["modes"][mode] = {}
        for metric in METRICS:
            values = [
                query["summary"]["modes"][mode][metric]
                for query in queries
            ]
            output["modes"][mode][metric] = {
                "target_queries": len(values),
                "query_mean": (
                    float(np.mean(values)) if values else None
                ),
                "positive_target_queries": int(
                    np.count_nonzero(np.asarray(values) > 0)
                ),
            }
    return output


def analyze_animal(
    path: Path,
    *,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> dict[str, Any]:
    seams = tuple(internal_seams())
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
    unavailable: Counter[str] = Counter()
    geometry_pairs = 0
    for training_exposure in range(repetitions - 1):
        training_start = training_exposure * 10
        test_start = (training_exposure + 1) * 10
        for target_offset in range(1, 10):
            target_test = test_start + target_offset
            for target_seam in seams:
                if (
                    seam_state(blocked[target_test], target_seam)
                    is not SeamState.WALL
                ):
                    continue
                records = []
                for source_seam in seams:
                    if (
                        transfer._orientation_relation(
                            target_seam,
                            source_seam,
                        )
                        != "same_signed_normal"
                        or not np.isclose(
                            transfer._midpoint_distance_cm(
                                target_seam,
                                source_seam,
                            ),
                            transfer.PRIMARY_DISTANCE_CM,
                        )
                    ):
                        continue
                    control_seam = _mirror_control_seam(
                        source_seam,
                        target_seam,
                        seams=seams,
                    )
                    if control_seam is None:
                        unavailable["no_internal_mirror"] += 1
                        continue
                    if (
                        seam_state(blocked[target_test], control_seam)
                        is not SeamState.OPEN
                    ):
                        unavailable["mirror_not_open"] += 1
                        continue
                    geometry_pairs += 1
                    record, reason = _paired_record(
                        training_start=training_start,
                        test_start=test_start,
                        target_offset=target_offset,
                        target_seam=target_seam,
                        source_seam=source_seam,
                        control_seam=control_seam,
                        blocked=blocked,
                        rate=rate,
                        occupancy=occupancy,
                        registered=registered,
                        strips=strips,
                        minimum_seconds=minimum_seconds,
                        minimum_bins=minimum_bins,
                        minimum_cells=minimum_cells,
                    )
                    if record is None:
                        assert reason is not None
                        unavailable[reason] += 1
                    else:
                        records.append(record)
                if not records:
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
                        "target_wall_seam": [
                            target_seam.source,
                            target_seam.target,
                        ],
                        "summary": _query_summary(records),
                        "records": records,
                    }
                )
    return {
        "animal": path.name.removesuffix(".complete.mat"),
        "repetitions": repetitions,
        "geometry_mirror_open_pairs_before_neural_support": geometry_pairs,
        "unavailable_reasons": dict(unavailable),
        "summary": _animal_summary(queries),
        "queries": queries,
    }


def _cohort_summary(
    animals: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "animals": len(animals),
        "animals_with_eligible_queries": int(
            sum(
                animal["summary"]["eligible_target_queries"] > 0
                for animal in animals
            )
        ),
        "eligible_target_queries": int(
            sum(
                animal["summary"]["eligible_target_queries"]
                for animal in animals
            )
        ),
        "eligible_source_wall_open_mirror_triplets": int(
            sum(
                animal["summary"][
                    "eligible_source_wall_open_mirror_triplets"
                ]
                for animal in animals
            )
        ),
        "modes": {},
    }
    for mode in transfer.RATE_MODES:
        output["modes"][mode] = {}
        for metric in METRICS:
            values_by_animal = {
                animal["animal"]: animal["summary"]["modes"][mode][
                    metric
                ]["query_mean"]
                for animal in animals
                if animal["summary"]["modes"][mode][metric][
                    "query_mean"
                ]
                is not None
            }
            values = np.asarray(
                list(values_by_animal.values()),
                dtype=np.float64,
            )
            output["modes"][mode][metric] = {
                "animals": int(values.size),
                "animal_mean": (
                    float(np.mean(values)) if values.size else None
                ),
                "animal_median": (
                    float(np.median(values)) if values.size else None
                ),
                "positive_animals": int(
                    np.count_nonzero(values > 0)
                ),
                "values_by_animal": values_by_animal,
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
    if argument.geometry_only:
        census = [geometry_census(path) for path in paths]
        print(
            json.dumps(
                {
                    "animals": census,
                    "animals_with_mirror_open_queries": int(
                        sum(
                            item["mirror_open_target_queries"] > 0
                            for item in census
                        )
                    ),
                    "mirror_open_target_queries": int(
                        sum(
                            item["mirror_open_target_queries"]
                            for item in census
                        )
                    ),
                    "mirror_open_pairs": int(
                        sum(
                            item.get("mirror_open_pairs", 0)
                            for item in census
                        )
                    ),
                },
                indent=2,
            )
        )
        return

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
            result["summary"]["eligible_target_queries"],
            result["summary"][
                "eligible_source_wall_open_mirror_triplets"
            ],
        )
    report = {
        "status": (
            "post_outcome_layout_matched_same_session_mirror_open_"
            "falsification"
        ),
        "question": (
            "Does a one-step source wall-minus-open vector predict a "
            "held-out wall strip better than an exact mirror-lag open strip "
            "in the same target session?"
        ),
        "design": {
            "source_predictor": (
                "training-exposure source wall residual minus source open "
                "residual, with target-geometry neural rates excluded"
            ),
            "wall_evaluation": (
                "held-out target-session wall strip minus the later-square "
                "strip at the same coordinates"
            ),
            "open_control_evaluation": (
                "OPEN seam in the same held-out target session, obtained by "
                "reflecting the wall target across the source seam"
            ),
            "source_distance_matched_cm": 25.0,
            "same_signed_seam_normal": True,
            "same_translation_axis": True,
            "opposite_displacement_sign": True,
            "same_target_session_and_global_geometry": True,
            "same_target_square_baseline": True,
            "paired_relative_target_control_bins": True,
            "same_registered_cells_for_all_vectors": True,
            "target_neural_rates_excluded_from_source_training": True,
            "source_and_target_neural_sessions_disjoint": True,
            "primary_metric": (
                "source-effect r to wall target minus source-effect r to "
                "mirror-lag open control"
            ),
            "secondary_metric": (
                "source-effect r to the within-session wall-minus-mirror-"
                "open vector"
            ),
            "rate_modes": list(transfer.RATE_MODES),
            "aggregation": (
                "equal source pairs within target query, equal queries "
                "within animal, descriptive equal-weight animals"
            ),
            "population_inference_performed": False,
        },
        "settings": {
            "minimum_seconds_per_common_bin": argument.minimum_seconds,
            "minimum_common_bins": argument.minimum_bins,
            "minimum_common_cells": argument.minimum_cells,
            "strip_depths_cm": [2.5, 7.5, 12.5],
            "geometry_feasibility_gate": (
                "at least 5 animals and at least 20 label-eligible target "
                "queries before neural support"
            ),
        },
        "cohort_descriptive": _cohort_summary(animals),
        "animals": animals,
        "limitations": [
            (
                "The wall and mirror-open locations have different absolute "
                "coordinates and can differ in nonfocal local context."
            ),
            (
                "A within-session open seam matches global geometry and "
                "session state but cannot match the animal's local behavior "
                "at two distinct positions."
            ),
            (
                "Source wall/open training pools are observational geometry "
                "groups, not a randomized focal-wall intervention."
            ),
            (
                "Queries, source pairs, cells, and sessions are dependent; "
                "animals are the biological units and no population p-value "
                "is reported."
            ),
        ],
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            report["cohort_descriptive"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
