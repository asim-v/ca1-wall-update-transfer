"""Raw-event split-half sensitivity for the experience difference-in-differences.

The primary endpoint is defined in
``run_boundary_fragment_experience_did.py``.  This script reconstructs only
the exposure-1 and exposure-3 sessions from raw positions and event traces,
splits every session into contiguous halves, and requires common cells and
spatial bins across all 16 half maps contributing to an endpoint record.

Each within-exposure recurrence estimate is the symmetric mean of the four
crossed, non-overlapping-side correlations used by the static raw-split audit.
The cost of that strict temporal independence is lower spatial coverage, which
is reported rather than relaxed post hoc.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ca1_geometry.boundary_fragments import (  # noqa: E402
    local_cell_rate,
    spearman_correlation,
)
from ca1_geometry.io import Mat73Animal  # noqa: E402
from ca1_geometry.seams import (  # noqa: E402
    OrientedSeam,
    internal_seams,
    seam_state,
    seam_strip_bins,
)
import run_boundary_fragment_experience_did as endpoint  # noqa: E402
import run_boundary_fragment_raw_split as raw_split  # noqa: E402


FloatArray = NDArray[np.float64]


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
            / "boundary_fragment_experience_raw_split.json"
        ),
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.25)
    parser.add_argument(
        "--sensitivity-minimum-seconds",
        type=float,
        default=0.50,
    )
    parser.add_argument("--minimum-bins", type=int, default=6)
    parser.add_argument("--minimum-cells", type=int, default=20)
    parser.add_argument("--minimum-environment-pairs", type=int, default=12)
    parser.add_argument(
        "--animal",
        action="append",
        default=[],
        help="Optional animal stem; repeat to select multiple animals.",
    )
    return parser.parse_args()


def _split_correlation(
    value: dict[tuple[int, int], FloatArray],
    *,
    pre_square: int,
    first_target: int,
    second_target: int,
    post_square: int,
) -> tuple[float, list[float]]:
    correlations = np.asarray(
        [
            spearman_correlation(
                value[first_target, 0] - value[pre_square, 0],
                value[second_target, 1] - value[post_square, 1],
            ),
            spearman_correlation(
                value[first_target, 0] - value[post_square, 0],
                value[second_target, 1] - value[pre_square, 1],
            ),
            spearman_correlation(
                value[first_target, 1] - value[pre_square, 1],
                value[second_target, 0] - value[post_square, 0],
            ),
            spearman_correlation(
                value[first_target, 1] - value[post_square, 1],
                value[second_target, 0] - value[pre_square, 0],
            ),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(correlations).all():
        return float("nan"), correlations.tolist()
    return float(np.mean(correlations)), correlations.tolist()


def _common_half_support(
    sessions: tuple[int, ...],
    occupancy: dict[int, list[FloatArray]],
    candidate_bins: tuple[tuple[int, int], ...],
    *,
    minimum_seconds: float,
) -> tuple[tuple[int, int], ...]:
    support = []
    for row, column in candidate_bins:
        dwell = np.asarray(
            [
                occupancy[session][half][row, column]
                for session in sessions
                for half in (0, 1)
            ],
            dtype=np.float64,
        )
        if np.isfinite(dwell).all() and np.min(dwell) >= minimum_seconds:
            support.append((row, column))
    return tuple(support)


def _record(
    *,
    first_offset: int,
    second_offset: int,
    rate: dict[int, list[FloatArray]],
    occupancy: dict[int, list[FloatArray]],
    registered: dict[int, NDArray[np.bool_]],
    candidate_bins: tuple[tuple[int, int], ...],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> tuple[dict[str, Any] | None, str]:
    required = (
        0,
        first_offset,
        second_offset,
        10,
        20,
        20 + first_offset,
        20 + second_offset,
        30,
    )
    cells = np.flatnonzero(
        np.logical_and.reduce([registered[session] for session in required])
    ).astype(np.int64, copy=False)
    if cells.size < minimum_cells:
        return None, "insufficient_common_cells"
    support = _common_half_support(
        required,
        occupancy,
        candidate_bins,
        minimum_seconds=minimum_seconds,
    )
    if len(support) < minimum_bins:
        return None, "insufficient_common_bins"
    value = {
        (session, half): local_cell_rate(
            rate[session][half],
            cells,
            support,
        )
        for session in required
        for half in (0, 1)
    }
    first, first_parts = _split_correlation(
        value,
        pre_square=0,
        first_target=first_offset,
        second_target=second_offset,
        post_square=10,
    )
    third, third_parts = _split_correlation(
        value,
        pre_square=20,
        first_target=20 + first_offset,
        second_target=20 + second_offset,
        post_square=30,
    )
    if not (np.isfinite(first) and np.isfinite(third)):
        return None, "nonfinite_correlation"
    return (
        {
            "cells": int(cells.size),
            "bins": len(support),
            "exposure_1_r": first,
            "exposure_3_r": third,
            "exposure_3_minus_1": float(third - first),
            "exposure_1_crossed_correlations": first_parts,
            "exposure_3_crossed_correlations": third_parts,
        },
        "eligible",
    )


def _pair_summary(
    records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    classes = {}
    for label in endpoint.STATE_LABELS:
        values = records[label]
        if values:
            first = float(
                np.mean([item["exposure_1_r"] for item in values])
            )
            third = float(
                np.mean([item["exposure_3_r"] for item in values])
            )
            classes[label] = {
                "seam_records": len(values),
                "exposure_1_mean_r": first,
                "exposure_3_mean_r": third,
                "exposure_3_minus_1": float(third - first),
            }
        else:
            classes[label] = None
    primary = None
    if classes["shared_wall"] and classes["shared_open"]:
        wall = classes["shared_wall"]
        control = classes["shared_open"]
        primary = {
            "shared_wall_exposure_1_mean_r": (
                wall["exposure_1_mean_r"]
            ),
            "shared_wall_exposure_3_mean_r": (
                wall["exposure_3_mean_r"]
            ),
            "shared_wall_exposure_3_minus_1": (
                wall["exposure_3_minus_1"]
            ),
            "shared_open_exposure_1_mean_r": (
                control["exposure_1_mean_r"]
            ),
            "shared_open_exposure_3_mean_r": (
                control["exposure_3_mean_r"]
            ),
            "shared_open_exposure_3_minus_1": (
                control["exposure_3_minus_1"]
            ),
            "exposure_1_wall_minus_open_open": float(
                wall["exposure_1_mean_r"]
                - control["exposure_1_mean_r"]
            ),
            "exposure_3_wall_minus_open_open": float(
                wall["exposure_3_mean_r"]
                - control["exposure_3_mean_r"]
            ),
            "wall_specific_stabilization_did": (
                endpoint.difference_in_differences(
                    wall["exposure_1_mean_r"],
                    control["exposure_1_mean_r"],
                    wall["exposure_3_mean_r"],
                    control["exposure_3_mean_r"],
                )
            ),
        }
    all_records = [
        item
        for label in endpoint.STATE_LABELS
        for item in records[label]
    ]
    return {
        "classes": classes,
        "primary_shared_wall_vs_shared_open": primary,
        "median_common_cells": (
            float(np.median([item["cells"] for item in all_records]))
            if all_records
            else None
        ),
        "median_common_bins": (
            float(np.median([item["bins"] for item in all_records]))
            if all_records
            else None
        ),
    }


def _threshold_analysis(
    *,
    rate: dict[int, list[FloatArray]],
    occupancy: dict[int, list[FloatArray]],
    registered: dict[int, NDArray[np.bool_]],
    environment: list[str],
    blocked: list[tuple[int, ...]],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
    minimum_environment_pairs: int,
) -> dict[str, Any]:
    seams = internal_seams()
    strips = {seam: seam_strip_bins(seam) for seam in seams}
    accounting = {
        label: {
            "state_candidates": 0,
            "eligible": 0,
            "insufficient_common_cells": 0,
            "insufficient_common_bins": 0,
            "nonfinite_correlation": 0,
        }
        for label in endpoint.STATE_LABELS
    }
    pair_summaries = []
    for first_offset, second_offset in itertools.combinations(
        range(1, 10),
        2,
    ):
        records: dict[str, list[dict[str, Any]]] = {
            label: [] for label in endpoint.STATE_LABELS
        }
        for seam in seams:
            label = endpoint.state_pair_label(
                seam_state(blocked[first_offset], seam),
                seam_state(blocked[second_offset], seam),
            )
            if label is None:
                continue
            accounting[label]["state_candidates"] += 1
            record, status = _record(
                first_offset=first_offset,
                second_offset=second_offset,
                rate=rate,
                occupancy=occupancy,
                registered=registered,
                candidate_bins=strips[seam],
                minimum_seconds=minimum_seconds,
                minimum_bins=minimum_bins,
                minimum_cells=minimum_cells,
            )
            accounting[label][status] += 1
            if record is not None:
                records[label].append(record)
        summary = _pair_summary(records)
        pair_summaries.append(
            {
                "first_environment": environment[first_offset],
                "second_environment": environment[second_offset],
                **summary,
            }
        )
    primary_rows = [
        (
            pair["first_environment"],
            pair["second_environment"],
            pair["primary_shared_wall_vs_shared_open"],
        )
        for pair in pair_summaries
        if pair["primary_shared_wall_vs_shared_open"] is not None
    ]
    did = [
        row["wall_specific_stabilization_did"]
        for _, _, row in primary_rows
    ]
    leave_one_pair_out = [
        float(np.mean(np.delete(np.asarray(did), index)))
        for index in range(len(did))
        if len(did) > 1
    ]
    return {
        "minimum_seconds_per_bin_in_each_half_map": minimum_seconds,
        "record_accounting": accounting,
        "primary_comparable_environment_pairs": len(primary_rows),
        "primary_coverage_eligible": (
            len(primary_rows) >= minimum_environment_pairs
        ),
        "primary_wall_specific_stabilization_did": (
            float(np.mean(did)) if did else None
        ),
        "primary_pair_median_did": (
            float(np.median(did)) if did else None
        ),
        "primary_positive_pair_dids": int(
            np.count_nonzero(np.asarray(did) > 0)
        ),
        "primary_leave_one_pair_out_min_mean_max": (
            [
                float(np.min(leave_one_pair_out)),
                float(np.mean(leave_one_pair_out)),
                float(np.max(leave_one_pair_out)),
            ]
            if leave_one_pair_out
            else None
        ),
        "environment_pairs": pair_summaries,
    }


def analyze_animal(
    path: Path,
    *,
    primary_minimum_seconds: float,
    sensitivity_minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
    minimum_environment_pairs: int,
) -> dict[str, Any]:
    with Mat73Animal(path) as animal:
        repetitions = (animal.n_sessions - 1) // 10
        environment = [
            animal.environment(session)
            for session in range(animal.n_sessions)
        ]
        blocked = [
            animal.blocked(session) for session in range(animal.n_sessions)
        ]
        base = {
            "animal": path.name.removesuffix(".complete.mat"),
            "sessions": animal.n_sessions,
            "exposure_cycles": repetitions,
            "first_cycle_order": environment[1:10],
            "same_shape_order_repeated": all(
                environment[cycle * 10 + 1 : cycle * 10 + 10]
                == environment[1:10]
                for cycle in range(1, repetitions)
            ),
        }
        if repetitions < 3:
            return {
                **base,
                "endpoint_eligible": False,
                "exclusion_reason": "fewer than three exposure cycles",
            }
        required_sessions = list(range(0, 11)) + list(range(20, 31))
        rate: dict[int, list[FloatArray]] = {}
        occupancy: dict[int, list[FloatArray]] = {}
        registered: dict[int, NDArray[np.bool_]] = {}
        audit = []
        for session in required_sessions:
            (
                rate[session],
                occupancy[session],
                registered[session],
                session_audit,
            ) = raw_split._session_half_maps(animal, session)
            audit.append({"session": session + 1, **session_audit})

    return {
        **base,
        "endpoint_eligible": True,
        "raw_reconstruction_audit": {
            "sessions_reconstructed": len(audit),
            "maximum_rate_absolute_error": float(
                np.nanmax(
                    [
                        item["rate_maximum_absolute_error"]
                        for item in audit
                    ]
                )
            ),
            "total_rate_finite_mask_mismatches": int(
                sum(
                    item["rate_finite_mask_mismatches"]
                    for item in audit
                )
            ),
            "maximum_occupancy_absolute_error_seconds": float(
                np.max(
                    [
                        item[
                            "occupancy_maximum_absolute_error_seconds"
                        ]
                        for item in audit
                    ]
                )
            ),
        },
        "primary": _threshold_analysis(
            rate=rate,
            occupancy=occupancy,
            registered=registered,
            environment=environment,
            blocked=blocked,
            minimum_seconds=primary_minimum_seconds,
            minimum_bins=minimum_bins,
            minimum_cells=minimum_cells,
            minimum_environment_pairs=minimum_environment_pairs,
        ),
        "sensitivity": _threshold_analysis(
            rate=rate,
            occupancy=occupancy,
            registered=registered,
            environment=environment,
            blocked=blocked,
            minimum_seconds=sensitivity_minimum_seconds,
            minimum_bins=minimum_bins,
            minimum_cells=minimum_cells,
            minimum_environment_pairs=minimum_environment_pairs,
        ),
    }


def _cohort_summary(
    animals: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    rows = [
        (animal["animal"], animal[key])
        for animal in animals
        if animal.get("endpoint_eligible")
        and animal[key]["primary_wall_specific_stabilization_did"] is not None
    ]
    eligible = [
        (name, row) for name, row in rows if row["primary_coverage_eligible"]
    ]
    values = [
        row["primary_wall_specific_stabilization_did"] for _, row in rows
    ]
    eligible_values = [
        row["primary_wall_specific_stabilization_did"]
        for _, row in eligible
    ]
    return {
        "endpoint_animals": len(rows),
        "positive_endpoint_animals": int(
            np.count_nonzero(np.asarray(values) > 0)
        ),
        "endpoint_animal_mean_did": (
            float(np.mean(values)) if values else None
        ),
        "animal_values": {
            name: row["primary_wall_specific_stabilization_did"]
            for name, row in rows
        },
        "animal_pair_counts": {
            name: row["primary_comparable_environment_pairs"]
            for name, row in rows
        },
        "coverage_eligible_animals": len(eligible),
        "positive_coverage_eligible_animals": int(
            np.count_nonzero(np.asarray(eligible_values) > 0)
        ),
        "coverage_eligible_animal_mean_did": (
            float(np.mean(eligible_values)) if eligible_values else None
        ),
        "population_inference_performed": False,
    }


def main() -> None:
    argument = parse_arguments()
    paths = sorted(argument.data_dir.glob("QLAK-CA1-*.complete.mat"))
    if argument.animal:
        selected = set(argument.animal)
        paths = [
            path
            for path in paths
            if path.name.removesuffix(".complete.mat") in selected
        ]
    if not paths:
        raise FileNotFoundError("no selected complete animal files found")

    animals = []
    for path in paths:
        print(f"Reconstructing endpoint halves for {path.name}...", flush=True)
        animals.append(
            analyze_animal(
                path,
                primary_minimum_seconds=argument.minimum_seconds,
                sensitivity_minimum_seconds=(
                    argument.sensitivity_minimum_seconds
                ),
                minimum_bins=argument.minimum_bins,
                minimum_cells=argument.minimum_cells,
                minimum_environment_pairs=(
                    argument.minimum_environment_pairs
                ),
            )
        )
        print(
            animals[-1]["animal"],
            (
                animals[-1].get("primary", {}).get(
                    "primary_wall_specific_stabilization_did"
                )
            ),
            flush=True,
        )

    report = {
        "status": "exploratory_raw_event_endpoint_sensitivity",
        "question": (
            "Does the exposure-3 versus exposure-1 shared-wall/open-open "
            "difference-in-differences survive independent raw-event halves?"
        ),
        "design": {
            "parent_endpoint": (
                "results/source_data/"
                "boundary_fragment_experience_did.json"
            ),
            "map_source": "raw positions and event traces",
            "temporal_split": "contiguous first and second session halves",
            "common_support": (
                "identical cells and bins across all 16 endpoint half maps"
            ),
            "four_crossed_nonoverlapping_side_assignments": True,
            "coverage_gate": (
                "12 comparable environment pairs inherited from the static "
                "discovery analysis"
            ),
            "inferential_unit": "mouse; no population inference",
        },
        "settings": {
            "primary_minimum_seconds_per_bin_in_each_half": (
                argument.minimum_seconds
            ),
            "sensitivity_minimum_seconds_per_bin_in_each_half": (
                argument.sensitivity_minimum_seconds
            ),
            "minimum_common_bins": argument.minimum_bins,
            "minimum_common_cells": argument.minimum_cells,
            "minimum_comparable_environment_pairs": (
                argument.minimum_environment_pairs
            ),
        },
        "interpretation_limit": (
            "Failure to meet coverage is an underidentification result, not "
            "evidence that the full-map endpoint is false; thresholds are "
            "not relaxed after seeing the outcome."
        ),
        "cohort": {
            "animals_loaded": len(animals),
            "primary": _cohort_summary(animals, "primary"),
            "sensitivity": _cohort_summary(animals, "sensitivity"),
        },
        "animals": animals,
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(argument.output),
                "cohort": report["cohort"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
