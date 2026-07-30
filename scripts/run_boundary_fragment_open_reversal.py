"""Run the symmetric open-target reversal of the wall-profile analysis.

For an oriented physical seam that is open in a target geometry, this script
learns exact-location OPEN and WALL local population profiles from the other
global geometries in the preceding exposure cycle.  It then asks which
profile better matches the target geometry's local registered-cell
square-residual vector in the next cycle.

Target neural rates are withheld from template fitting.  The target seam
label, target occupancy, and target registration mask are deliberately used
to construct the local query, so this is a target-aware local prediction
rather than blind prediction of a full rate map or arena identity.
"""

from __future__ import annotations

import argparse
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
    seam_state,
    seam_strip_bins,
)


RateFunction = Callable[
    [np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, int], ...]],
    np.ndarray,
]


@dataclass(frozen=True)
class ReversalQuery:
    """Session indices for one target-aware open-seam reversal query."""

    train_baseline: int
    test_baseline: int
    withheld_target_train: int
    target_test: int
    matching_open: tuple[int, ...]
    matching_wall: tuple[int, ...]


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
            / "boundary_fragment_open_reversal.json"
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


def _reversal_query(
    *,
    target_offset: int,
    training_exposure: int,
    blocked: list[tuple[int, ...]],
    seam: OrientedSeam,
) -> ReversalQuery | None:
    """Construct one open-target query without using target neural rates."""

    if not 1 <= target_offset <= 9:
        raise ValueError("target_offset must lie in [1, 9]")
    train_start = training_exposure * 10
    test_start = (training_exposure + 1) * 10
    train_baseline = train_start
    test_baseline = (training_exposure + 2) * 10
    withheld_target_train = train_start + target_offset
    target_test = test_start + target_offset
    if test_baseline >= len(blocked):
        raise ValueError("query requires a square session after the test cycle")
    if seam_state(blocked[target_test], seam) is not SeamState.OPEN:
        return None

    candidates = [
        train_start + offset
        for offset in range(1, 10)
        if offset != target_offset
    ]
    matching_open = tuple(
        session
        for session in candidates
        if seam_state(blocked[session], seam) is SeamState.OPEN
    )
    matching_wall = tuple(
        session
        for session in candidates
        if seam_state(blocked[session], seam) is SeamState.WALL
    )
    if not matching_open or not matching_wall:
        return None
    return ReversalQuery(
        train_baseline=train_baseline,
        test_baseline=test_baseline,
        withheld_target_train=withheld_target_train,
        target_test=target_test,
        matching_open=matching_open,
        matching_wall=matching_wall,
    )


def _mean_residual_vector(
    sessions: tuple[int, ...],
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
    return np.mean(
        np.stack(
            [
                rate_function(
                    rate[session],
                    occupancy[session],
                    cells,
                    bins,
                )
                - baseline_value
                for session in sessions
            ]
        ),
        axis=0,
    )


def _target_residual_vector(
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
    return (
        "vertical_seam"
        if abs(seam.source - seam.target) == 1
        else "horizontal_seam"
    )


def _evaluate_query(
    query: ReversalQuery,
    *,
    environment: list[str],
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    seam: OrientedSeam,
    strip: tuple[tuple[int, int], ...],
    training_exposure: int,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> tuple[dict[str, Any] | None, str]:
    required = [
        query.train_baseline,
        query.test_baseline,
        query.target_test,
        *query.matching_open,
        *query.matching_wall,
    ]
    cells = np.flatnonzero(
        np.logical_and.reduce([registered[session] for session in required])
    )
    if cells.size < minimum_cells:
        return None, "insufficient_registered_cells"
    support = common_support_bins(
        [occupancy[session] for session in required],
        strip,
        minimum_seconds=minimum_seconds,
    )
    if len(support) < minimum_bins:
        return None, "insufficient_common_bins"

    correlations: dict[str, dict[str, float]] = {}
    for mode, rate_function in RATE_MODES.items():
        open_profile = _mean_residual_vector(
            query.matching_open,
            baseline=query.train_baseline,
            rate=rate,
            occupancy=occupancy,
            cells=cells,
            bins=support,
            rate_function=rate_function,
        )
        wall_profile = _mean_residual_vector(
            query.matching_wall,
            baseline=query.train_baseline,
            rate=rate,
            occupancy=occupancy,
            cells=cells,
            bins=support,
            rate_function=rate_function,
        )
        target = _target_residual_vector(
            query.target_test,
            baseline=query.test_baseline,
            rate=rate,
            occupancy=occupancy,
            cells=cells,
            bins=support,
            rate_function=rate_function,
        )
        open_r = spearman_correlation(open_profile, target)
        wall_r = spearman_correlation(wall_profile, target)
        if not (np.isfinite(open_r) and np.isfinite(wall_r)):
            return None, f"nonfinite_{mode}_correlation"
        correlations[mode] = {
            "matching_open_r": open_r,
            "matching_wall_r": wall_r,
            "open_minus_wall": open_r - wall_r,
        }

    return (
        {
            "training_exposure": training_exposure + 1,
            "test_exposure": training_exposure + 2,
            "target_environment": environment[query.target_test],
            "withheld_training_session": query.withheld_target_train + 1,
            "test_session": query.target_test + 1,
            "training_square_session": query.train_baseline + 1,
            "test_square_session": query.test_baseline + 1,
            "seam": [seam.source, seam.target],
            "orientation": _orientation(seam),
            "target_state": "open",
            "matching_open_environments": [
                environment[session] for session in query.matching_open
            ],
            "matching_wall_environments": [
                environment[session] for session in query.matching_wall
            ],
            "matching_open_sessions": [
                session + 1 for session in query.matching_open
            ],
            "matching_wall_sessions": [
                session + 1 for session in query.matching_wall
            ],
            "matching_open_count": len(query.matching_open),
            "matching_wall_count": len(query.matching_wall),
            "cells": int(cells.size),
            "bins": int(len(support)),
            "correlations": correlations,
        },
        "eligible",
    )


def _summarize_records(
    records: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    if not records:
        return {
            "local_predictions": 0,
            "matching_open_mean_r": None,
            "matching_wall_mean_r": None,
            "open_minus_wall_mean": None,
            "positive_local_predictions": 0,
            "median_cells": None,
            "median_bins": None,
            "median_matching_open_shapes": None,
            "median_matching_wall_shapes": None,
            "orientation": {
                label: {
                    "eligible_local_predictions": 0,
                    "mean_open_minus_wall": None,
                }
                for label in ("vertical_seam", "horizontal_seam")
            },
        }
    open_value = np.asarray(
        [
            record["correlations"][mode]["matching_open_r"]
            for record in records
        ],
        dtype=np.float64,
    )
    wall = np.asarray(
        [
            record["correlations"][mode]["matching_wall_r"]
            for record in records
        ],
        dtype=np.float64,
    )
    contrast = open_value - wall
    orientation = {}
    for label in ("vertical_seam", "horizontal_seam"):
        keep = np.asarray(
            [record["orientation"] == label for record in records]
        )
        orientation[label] = {
            "eligible_local_predictions": int(np.count_nonzero(keep)),
            "mean_open_minus_wall": (
                float(np.mean(contrast[keep])) if np.any(keep) else None
            ),
        }
    return {
        "local_predictions": len(records),
        "matching_open_mean_r": float(np.mean(open_value)),
        "matching_wall_mean_r": float(np.mean(wall)),
        "open_minus_wall_mean": float(np.mean(contrast)),
        "positive_local_predictions": int(np.count_nonzero(contrast > 0)),
        "median_cells": float(
            np.median([record["cells"] for record in records])
        ),
        "median_bins": float(
            np.median([record["bins"] for record in records])
        ),
        "median_matching_open_shapes": float(
            np.median(
                [record["matching_open_count"] for record in records]
            )
        ),
        "median_matching_wall_shapes": float(
            np.median(
                [record["matching_wall_count"] for record in records]
            )
        ),
        "orientation": orientation,
    }


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

    records: list[dict[str, Any]] = []
    accounting = {
        "oriented_open_target_queries": 0,
        "queries_with_both_training_states": 0,
        "missing_training_open_or_wall": 0,
        "eligible": 0,
        "insufficient_registered_cells": 0,
        "insufficient_common_bins": 0,
        "nonfinite_correlation": 0,
    }
    for training_exposure in range(repetitions - 1):
        for target_offset in range(1, 10):
            target_test = (training_exposure + 1) * 10 + target_offset
            for seam in seams:
                if (
                    seam_state(blocked[target_test], seam)
                    is not SeamState.OPEN
                ):
                    continue
                accounting["oriented_open_target_queries"] += 1
                query = _reversal_query(
                    target_offset=target_offset,
                    training_exposure=training_exposure,
                    blocked=blocked,
                    seam=seam,
                )
                if query is None:
                    accounting["missing_training_open_or_wall"] += 1
                    continue
                accounting["queries_with_both_training_states"] += 1
                record, status = _evaluate_query(
                    query,
                    environment=environment,
                    rate=rate,
                    occupancy=occupancy,
                    registered=registered,
                    seam=seam,
                    strip=strips[seam],
                    training_exposure=training_exposure,
                    minimum_seconds=minimum_seconds,
                    minimum_bins=minimum_bins,
                    minimum_cells=minimum_cells,
                )
                if record is not None:
                    records.append(record)
                    accounting["eligible"] += 1
                elif status.startswith("nonfinite_"):
                    accounting["nonfinite_correlation"] += 1
                else:
                    accounting[status] += 1

    summaries = {
        mode: _summarize_records(records, mode=mode)
        for mode in RATE_MODES
    }
    exposure_pairs = {}
    for training_exposure in range(repetitions - 1):
        label = f"{training_exposure + 1}_to_{training_exposure + 2}"
        subset = [
            record
            for record in records
            if record["training_exposure"] == training_exposure + 1
        ]
        exposure_pairs[label] = {
            mode: _summarize_records(subset, mode=mode)
            for mode in RATE_MODES
        }

    return {
        "animal": path.name.removesuffix(".complete.mat"),
        "repetitions": repetitions,
        "query_accounting": accounting,
        "summaries": summaries,
        "exposure_pairs": exposure_pairs,
        "records": records,
    }


def cohort_summary(animals: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "animals_loaded": len(animals),
        "animal_names": [animal["animal"] for animal in animals],
        "query_accounting": {
            key: int(
                sum(
                    animal["query_accounting"][key]
                    for animal in animals
                )
            )
            for key in animals[0]["query_accounting"]
        },
        "modes": {},
    }
    for mode in RATE_MODES:
        eligible = [
            animal
            for animal in animals
            if animal["summaries"][mode]["open_minus_wall_mean"] is not None
        ]
        values = np.asarray(
            [
                animal["summaries"][mode]["open_minus_wall_mean"]
                for animal in eligible
            ],
            dtype=np.float64,
        )
        open_value = np.asarray(
            [
                animal["summaries"][mode]["matching_open_mean_r"]
                for animal in eligible
            ],
            dtype=np.float64,
        )
        wall = np.asarray(
            [
                animal["summaries"][mode]["matching_wall_mean_r"]
                for animal in eligible
            ],
            dtype=np.float64,
        )
        output["modes"][mode] = {
            "animals_loaded": len(animals),
            "eligible_animals": len(eligible),
            "positive_animals": int(np.count_nonzero(values > 0)),
            "animal_mean_open_minus_wall": (
                float(np.mean(values)) if values.size else None
            ),
            "animal_median_open_minus_wall": (
                float(np.median(values)) if values.size else None
            ),
            "animal_mean_matching_open_r": (
                float(np.mean(open_value)) if open_value.size else None
            ),
            "animal_mean_matching_wall_r": (
                float(np.mean(wall)) if wall.size else None
            ),
            "animal_values": {
                animal["animal"]: (
                    animal["summaries"][mode]["open_minus_wall_mean"]
                )
                for animal in animals
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
                mode: summary["open_minus_wall_mean"]
                for mode, summary in result["summaries"].items()
            },
            result["query_accounting"],
        )

    report = {
        "status": "exploratory_symmetric_open_target_reversal",
        "question": (
            "When the exact oriented seam is open in the target geometry, "
            "does an exact-location open profile learned from other global "
            "shapes match the next-cycle local population vector better "
            "than the corresponding wall profile?"
        ),
        "design": {
            "target_neural_rates_excluded_from_template_fitting": True,
            "target_open_label_used_to_select_test_queries": True,
            "target_occupancy_and_registration_used_for_support": True,
            "target_support_selection_is_test_aware": True,
            "predicted_object": (
                "one square-residual local strip rate per registered cell; "
                "not a full spatial map or global shape"
            ),
            "open_query_orientation": (
                "an open physical seam can contribute one local query on "
                "each side (one per directed seam) when both satisfy support "
                "thresholds"
            ),
            "training_templates": (
                "means over other global geometries in the preceding "
                "exposure cycle; the target geometry's training rates are "
                "withheld from both templates"
            ),
            "training_baseline": "square_before_training_cycle",
            "test_baseline": "square_after_test_cycle",
            "independent_outer_square_baselines": True,
            "training_and_test_neural_sessions_nonoverlapping": True,
            "contrast": "matching_open_r_minus_matching_wall_r",
            "rate_modes": list(RATE_MODES),
            "inferential_unit": "animal",
            "inference": (
                "descriptive falsification control; no confirmatory "
                "population p-value"
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
