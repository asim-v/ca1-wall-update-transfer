"""Test exact-location CA1 wall-conditioned profiles with target rates held out.

The profile is learned from one exposure cycle using every geometry except the
target geometry's neural rates. It is tested on a local registered-cell
rate-change vector from that geometry in the next cycle. Target wall labels,
occupancy support, and the registration mask are known when constructing the
test vector; this is not blind prediction of a full spatial map or global
shape.
Training and test residuals use non-overlapping outer square baselines.
"""

from __future__ import annotations

import argparse
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
            / "boundary_component_validation.json"
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


def _mean_vector(
    session: list[int],
    *,
    baseline: int,
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    cells: np.ndarray,
    bins: tuple[tuple[int, int], ...],
    rate_function: RateFunction,
) -> np.ndarray:
    baseline_value = rate_function(
        rate[baseline],
        occupancy[baseline],
        cells,
        bins,
    )
    residual = [
        rate_function(rate[item], occupancy[item], cells, bins)
        - baseline_value
        for item in session
    ]
    return np.mean(np.stack(residual), axis=0)


def _target_vector(
    session: int,
    *,
    baseline: int,
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    cells: np.ndarray,
    bins: tuple[tuple[int, int], ...],
    rate_function: RateFunction,
) -> np.ndarray:
    return (
        rate_function(rate[session], occupancy[session], cells, bins)
        - rate_function(
            rate[baseline],
            occupancy[baseline],
            cells,
            bins,
        )
    )


def _orientation(seam: OrientedSeam) -> str:
    return "vertical_wall" if abs(seam.source - seam.target) == 1 else (
        "horizontal_wall"
    )


def nonfocal_target_context(
    blocked_partitions: tuple[int, ...],
    seam: OrientedSeam,
) -> tuple[tuple[int, bool], ...]:
    """Encode target-neighbor states other than the focal source.

    The target partition and physical seam remain fixed.  Consequently, the
    arena's outer boundary is fixed too; only internal grid neighbors need to
    be represented explicitly.
    """

    target_row, target_column = divmod(seam.target, 3)
    blocked_set = set(blocked_partitions)
    neighbors: list[int] = []
    for delta_row, delta_column in (
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1),
    ):
        row = target_row + delta_row
        column = target_column + delta_column
        if not (0 <= row < 3 and 0 <= column < 3):
            continue
        neighbor = 3 * row + column
        if neighbor != seam.source:
            neighbors.append(neighbor)
    return tuple(
        (neighbor, neighbor in blocked_set)
        for neighbor in sorted(neighbors)
    )


def _record(
    *,
    target_offset: int,
    training_exposure: int,
    environment: list[str],
    blocked: list[tuple[int, ...]],
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    seam: OrientedSeam,
    strip: tuple[tuple[int, int], ...],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
    match_nonfocal_context: bool = False,
    match_global_counterfactual: bool = False,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, ...]]] | None:
    train_start = training_exposure * 10
    test_start = (training_exposure + 1) * 10
    train_baseline = train_start
    test_baseline = (training_exposure + 2) * 10
    target_train = train_start + target_offset
    target_test = test_start + target_offset
    target_context = nonfocal_target_context(
        blocked[target_test],
        seam,
    )

    matching_wall = [
        train_start + offset
        for offset in range(1, 10)
        if offset != target_offset
        and seam_state(
            blocked[train_start + offset],
            seam,
        )
        is SeamState.WALL
        and (
            not match_nonfocal_context
            or nonfocal_target_context(
                blocked[train_start + offset],
                seam,
            )
            == target_context
        )
    ]
    matching_open = [
        train_start + offset
        for offset in range(1, 10)
        if offset != target_offset
        and seam_state(
            blocked[train_start + offset],
            seam,
        )
        is SeamState.OPEN
        and (
            not match_nonfocal_context
            or nonfocal_target_context(
                blocked[train_start + offset],
                seam,
            )
            == target_context
        )
    ]
    if match_global_counterfactual:
        wall_context = {
            tuple(
                partition
                for partition in blocked[session]
                if partition != seam.source
            )
            for session in matching_wall
        }
        open_context = {
            tuple(
                partition
                for partition in blocked[session]
                if partition != seam.source
            )
            for session in matching_open
        }
        shared_context = wall_context & open_context
        matching_wall = [
            session
            for session in matching_wall
            if tuple(
                partition
                for partition in blocked[session]
                if partition != seam.source
            )
            in shared_context
        ]
        matching_open = [
            session
            for session in matching_open
            if tuple(
                partition
                for partition in blocked[session]
                if partition != seam.source
            )
            in shared_context
        ]
    if not matching_wall or not matching_open:
        return None

    required = [
        train_baseline,
        test_baseline,
        target_test,
        *matching_wall,
        *matching_open,
    ]
    cells = np.flatnonzero(
        np.logical_and.reduce([registered[item] for item in required])
    )
    if cells.size < minimum_cells:
        return None
    support = common_support_bins(
        [occupancy[item] for item in required],
        strip,
        minimum_seconds=minimum_seconds,
    )
    if len(support) < minimum_bins:
        return None

    correlations: dict[str, dict[str, float]] = {}
    vectors: dict[str, tuple[np.ndarray, ...]] = {}
    for mode, rate_function in RATE_MODES.items():
        wall = _mean_vector(
            matching_wall,
            baseline=train_baseline,
            rate=rate,
            occupancy=occupancy,
            cells=cells,
            bins=support,
            rate_function=rate_function,
        )
        open_value = _mean_vector(
            matching_open,
            baseline=train_baseline,
            rate=rate,
            occupancy=occupancy,
            cells=cells,
            bins=support,
            rate_function=rate_function,
        )
        target = _target_vector(
            target_test,
            baseline=test_baseline,
            rate=rate,
            occupancy=occupancy,
            cells=cells,
            bins=support,
            rate_function=rate_function,
        )
        wall_r = spearman_correlation(wall, target)
        open_r = spearman_correlation(open_value, target)
        if not (np.isfinite(wall_r) and np.isfinite(open_r)):
            return None
        correlations[mode] = {
            "matching_wall_r": wall_r,
            "matching_open_r": open_r,
            "wall_minus_open": wall_r - open_r,
        }
        vectors[mode] = (wall, open_value, target)

    return (
        {
            "training_exposure": training_exposure + 1,
            "test_exposure": training_exposure + 2,
            "target_environment": environment[target_test],
            "withheld_training_session": target_train + 1,
            "test_session": target_test + 1,
            "training_square_session": train_baseline + 1,
            "test_square_session": test_baseline + 1,
            "seam": [seam.source, seam.target],
            "orientation": _orientation(seam),
            "nonfocal_context_matched": match_nonfocal_context,
            "global_counterfactual_matched": (
                match_global_counterfactual
            ),
            "target_nonfocal_context": [
                [partition, "blocked" if is_blocked else "accessible"]
                for partition, is_blocked in target_context
            ],
            "matching_wall_environments": [
                environment[item] for item in matching_wall
            ],
            "matching_open_environments": [
                environment[item] for item in matching_open
            ],
            "cells": int(cells.size),
            "bins": int(len(support)),
            "correlations": correlations,
        },
        vectors,
    )


def _summarize_records(
    records: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    if not records:
        return {
            "local_predictions": 0,
            "matching_wall_mean_r": None,
            "matching_open_mean_r": None,
            "wall_minus_open_mean": None,
            "positive_local_predictions": 0,
            "median_cells": None,
            "median_bins": None,
            "orientation": {
                label: {
                    "eligible_local_predictions": 0,
                    "mean_wall_minus_open": None,
                }
                for label in ("vertical_wall", "horizontal_wall")
            },
        }
    wall = np.asarray(
        [
            item["correlations"][mode]["matching_wall_r"]
            for item in records
        ]
    )
    open_value = np.asarray(
        [
            item["correlations"][mode]["matching_open_r"]
            for item in records
        ]
    )
    contrast = wall - open_value
    orientation = {}
    for label in ("vertical_wall", "horizontal_wall"):
        keep = np.asarray(
            [item["orientation"] == label for item in records]
        )
        orientation[label] = {
            "eligible_local_predictions": int(np.count_nonzero(keep)),
            "mean_wall_minus_open": (
                float(np.mean(contrast[keep]))
                if np.any(keep)
                else None
            ),
        }
    return {
        "local_predictions": len(records),
        "matching_wall_mean_r": float(np.mean(wall)),
        "matching_open_mean_r": float(np.mean(open_value)),
        "wall_minus_open_mean": float(np.mean(contrast)),
        "positive_local_predictions": int(np.count_nonzero(contrast > 0)),
        "median_cells": float(
            np.median([item["cells"] for item in records])
        ),
        "median_bins": float(
            np.median([item["bins"] for item in records])
        ),
        "orientation": orientation,
    }


def analyze_animal(
    path: Path,
    *,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
    match_nonfocal_context: bool = False,
    match_global_counterfactual: bool = False,
    strip_depths_cm: tuple[float, ...] = (2.5, 7.5, 12.5),
) -> dict[str, Any]:
    seams = internal_seams()
    strip = {
        seam: seam_strip_bins(seam, depths_cm=strip_depths_cm)
        for seam in seams
    }
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

    records: list[dict[str, Any]] = []
    for training_exposure in range(repetitions - 1):
        for target_offset in range(1, 10):
            target_test = (training_exposure + 1) * 10 + target_offset
            for seam in seams:
                if (
                    seam_state(blocked[target_test], seam)
                    is not SeamState.WALL
                ):
                    continue
                result = _record(
                    target_offset=target_offset,
                    training_exposure=training_exposure,
                    environment=environment,
                    blocked=blocked,
                    rate=rate,
                    occupancy=occupancy,
                    registered=registered,
                    seam=seam,
                    strip=strip[seam],
                    minimum_seconds=minimum_seconds,
                    minimum_bins=minimum_bins,
                    minimum_cells=minimum_cells,
                    match_nonfocal_context=match_nonfocal_context,
                    match_global_counterfactual=(
                        match_global_counterfactual
                    ),
                )
                if result is None:
                    continue
                record, _vectors = result
                records.append(record)

    summaries = {
        mode: _summarize_records(records, mode=mode)
        for mode in RATE_MODES
    }
    exposure_pairs = {}
    for training_exposure in range(repetitions - 1):
        label = f"{training_exposure + 1}_to_{training_exposure + 2}"
        subset = [
            item
            for item in records
            if item["training_exposure"] == training_exposure + 1
        ]
        exposure_pairs[label] = {
            mode: _summarize_records(subset, mode=mode)
            for mode in RATE_MODES
        }

    return {
        "animal": path.name.removesuffix(".complete.mat"),
        "repetitions": repetitions,
        "summaries": summaries,
        "exposure_pairs": exposure_pairs,
        "records": records,
    }


def cohort_summary(animals: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for mode in RATE_MODES:
        value = np.asarray(
            [
                item["summaries"][mode]["wall_minus_open_mean"]
                for item in animals
            ]
        )
        wall = np.asarray(
            [
                item["summaries"][mode]["matching_wall_mean_r"]
                for item in animals
            ]
        )
        open_value = np.asarray(
            [
                item["summaries"][mode]["matching_open_mean_r"]
                for item in animals
            ]
        )
        output[mode] = {
            "animals": int(value.size),
            "positive_animals": int(np.count_nonzero(value > 0)),
            "animal_mean_wall_minus_open": float(np.mean(value)),
            "animal_median_wall_minus_open": float(np.median(value)),
            "animal_mean_matching_wall_r": float(np.mean(wall)),
            "animal_mean_matching_open_r": float(np.mean(open_value)),
            "animal_values": {
                item["animal"]: (
                    item["summaries"][mode]["wall_minus_open_mean"]
                )
                for item in animals
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
                    value["wall_minus_open_mean"],
                    4,
                )
                for mode, value in result["summaries"].items()
            },
        )

    report = {
        "status": "exploratory_cross_exposure_validation",
        "question": (
            "Do exact-location wall-conditioned cell profiles learned from "
            "other global shapes better match a target local registered-cell "
            "rate-change vector in the next exposure than open-seam profiles?"
        ),
        "design": {
            "target_neural_rates_excluded_from_template_fitting": True,
            "target_wall_label_used_to_select_test_queries": True,
            "target_occupancy_and_registration_used_for_support": True,
            "predicted_object": (
                "one square-residual local strip rate per registered cell; "
                "not a full spatial map or global shape"
            ),
            "training_and_test_exposures_nonoverlapping": True,
            "training_baseline": "square_before_training_cycle",
            "test_baseline": "square_after_test_cycle",
            "training_and_test_baselines_nonoverlapping": True,
            "matching_wall_predictor": (
                "mean residual from other shapes with the exact oriented "
                "physical seam walled"
            ),
            "control_predictor": (
                "mean residual from other shapes with that seam open"
            ),
            "rate_modes": list(RATE_MODES),
            "inferential_unit": "animal",
            "registered_cell_identity_diagnostic": (
                "run_boundary_fragment_session_permutation.py; one coherent "
                "global-ID mapping per animal and draw"
            ),
        },
        "settings": {
            "minimum_seconds_per_common_bin": argument.minimum_seconds,
            "minimum_common_bins": argument.minimum_bins,
            "minimum_common_cells": argument.minimum_cells,
        },
        "cohort": cohort_summary(animals),
        "animals": animals,
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["cohort"], indent=2))


if __name__ == "__main__":
    main()
