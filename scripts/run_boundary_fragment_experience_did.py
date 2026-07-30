"""Audit experience-related change in exact-location wall-profile recurrence.

This analysis asks a deliberately narrower question than whether CA1 "learns
walls."  For animals with three complete exposure cycles, it compares cycle 1
with cycle 3 using:

* the same pair of global environments;
* the same oriented physical seam;
* the same registered cells and spatial bins in all eight neural/baseline
  sessions contributing to the endpoint comparison; and
* disjoint target sessions and disjoint bracketing square baselines.

For each environment pair, recurrence at seams walled in both environments is
contrasted with recurrence at seams open in both environments.  The primary
estimand is therefore a difference in differences:

    (wall/wall - open/open)_cycle3 - (wall/wall - open/open)_cycle1.

Open/open seams are a control for the source dataset's general exposure-related
increase in map reliability.  This remains an exploratory observational
diagnostic: exposure cycle is confounded with elapsed recording time, the
parallel-trends assumption is not testable with only three cycles, and mice
rather than seam records are the biological units.
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

from ca1_geometry.boundary_fragments import (  # noqa: E402
    common_support_bins,
    globally_demeaned_local_cell_rate,
    independent_square_residual_correlation,
    local_cell_rate,
)
from ca1_geometry.io import Mat73Animal  # noqa: E402
from ca1_geometry.seams import (  # noqa: E402
    OrientedSeam,
    SeamState,
    internal_seams,
    seam_state,
    seam_strip_bins,
)


FloatArray = NDArray[np.float64]
STATE_LABELS = ("shared_wall", "shared_open", "wall_open")
RATE_MODES = ("raw_local_rate", "global_rate_demeaned")
STRIP_SPECS = {
    "primary_15cm": {
        "depths_cm": (2.5, 7.5, 12.5),
        "minimum_bins": 6,
    },
    "near_10cm": {
        "depths_cm": (2.5, 7.5),
        "minimum_bins": 4,
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
            / "boundary_fragment_experience_did.json"
        ),
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-cells", type=int, default=20)
    parser.add_argument(
        "--minimum-environment-pairs",
        type=int,
        default=12,
        help=(
            "Coverage gate inherited from the discovery screen; it is not "
            "used to tune or filter individual seam records."
        ),
    )
    return parser.parse_args()


def state_pair_label(
    first: SeamState,
    second: SeamState,
) -> str | None:
    """Classify an oriented seam across a fixed pair of environments."""

    if first is SeamState.WALL and second is SeamState.WALL:
        return "shared_wall"
    if first is SeamState.OPEN and second is SeamState.OPEN:
        return "shared_open"
    if {first, second} == {SeamState.WALL, SeamState.OPEN}:
        return "wall_open"
    return None


def difference_in_differences(
    first_wall: float,
    first_control: float,
    later_wall: float,
    later_control: float,
) -> float:
    """Return (wall-control)_later minus (wall-control)_first."""

    value = np.asarray(
        [first_wall, first_control, later_wall, later_control],
        dtype=np.float64,
    )
    if not np.isfinite(value).all():
        return float("nan")
    return float((later_wall - later_control) - (first_wall - first_control))


def _finite_mean(values: list[float]) -> float | None:
    finite = np.asarray(
        [value for value in values if np.isfinite(value)],
        dtype=np.float64,
    )
    return float(np.mean(finite)) if finite.size else None


def _finite_median(values: list[float]) -> float | None:
    finite = np.asarray(
        [value for value in values if np.isfinite(value)],
        dtype=np.float64,
    )
    return float(np.median(finite)) if finite.size else None


def _session_cache(
    animal: Mat73Animal,
) -> tuple[
    dict[int, FloatArray],
    dict[int, FloatArray],
    dict[int, NDArray[np.bool_]],
    list[str],
    list[tuple[int, ...]],
]:
    sessions = range(animal.n_sessions)
    rate = {
        session: animal.stored_rate_maps(session, smoothed=False)
        for session in sessions
    }
    occupancy = {
        session: animal.sampling_map(session)
        for session in sessions
    }
    registered = {
        session: animal.registered_cells(session)
        for session in sessions
    }
    environment = [
        animal.environment(session) for session in range(animal.n_sessions)
    ]
    blocked = [
        animal.blocked(session) for session in range(animal.n_sessions)
    ]
    return rate, occupancy, registered, environment, blocked


def _local_vector(
    mode: str,
    *,
    rate: FloatArray,
    occupancy: FloatArray,
    cells: NDArray[np.int64],
    bins: tuple[tuple[int, int], ...],
) -> FloatArray:
    if mode == "raw_local_rate":
        return local_cell_rate(rate, cells, bins)
    if mode == "global_rate_demeaned":
        return globally_demeaned_local_cell_rate(
            rate,
            occupancy,
            cells,
            bins,
        )
    raise ValueError(f"unknown rate mode: {mode}")


def _record(
    *,
    first_offset: int,
    second_offset: int,
    seam: OrientedSeam,
    rate: dict[int, FloatArray],
    occupancy: dict[int, FloatArray],
    registered: dict[int, NDArray[np.bool_]],
    candidate_bins: tuple[tuple[int, int], ...],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> tuple[dict[str, Any] | None, str]:
    first_sessions = (0, first_offset, second_offset, 10)
    third_sessions = (
        20,
        20 + first_offset,
        20 + second_offset,
        30,
    )
    required = first_sessions + third_sessions
    cells = np.flatnonzero(
        np.logical_and.reduce([registered[session] for session in required])
    ).astype(np.int64, copy=False)
    if cells.size < minimum_cells:
        return None, "insufficient_common_cells"
    support = common_support_bins(
        [occupancy[session] for session in required],
        candidate_bins,
        minimum_seconds=minimum_seconds,
    )
    if len(support) < minimum_bins:
        return None, "insufficient_common_bins"

    by_mode: dict[str, dict[str, float]] = {}
    for mode in RATE_MODES:
        vector = {
            session: _local_vector(
                mode,
                rate=rate[session],
                occupancy=occupancy[session],
                cells=cells,
                bins=support,
            )
            for session in required
        }
        first = independent_square_residual_correlation(
            vector[first_offset],
            vector[second_offset],
            vector[0],
            vector[10],
        ).mean
        third = independent_square_residual_correlation(
            vector[20 + first_offset],
            vector[20 + second_offset],
            vector[20],
            vector[30],
        ).mean
        if not (np.isfinite(first) and np.isfinite(third)):
            return None, "nonfinite_correlation"
        by_mode[mode] = {
            "exposure_1_r": float(first),
            "exposure_3_r": float(third),
            "exposure_3_minus_1": float(third - first),
        }
    return (
        {
            "cells": int(cells.size),
            "bins": int(len(support)),
            "correlations": by_mode,
        },
        "eligible",
    )


def _class_summary(
    records: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any] | None:
    if not records:
        return None
    first = np.asarray(
        [
            record["correlations"][mode]["exposure_1_r"]
            for record in records
        ],
        dtype=np.float64,
    )
    third = np.asarray(
        [
            record["correlations"][mode]["exposure_3_r"]
            for record in records
        ],
        dtype=np.float64,
    )
    return {
        "seam_records": len(records),
        "exposure_1_mean_r": float(np.mean(first)),
        "exposure_3_mean_r": float(np.mean(third)),
        "exposure_3_minus_1": float(np.mean(third - first)),
    }


def _pair_summary(
    records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "record_counts": {
            label: len(records[label]) for label in STATE_LABELS
        },
        "median_common_cells": (
            float(
                np.median(
                    [
                        item["cells"]
                        for label in STATE_LABELS
                        for item in records[label]
                    ]
                )
            )
            if any(records.values())
            else None
        ),
        "median_common_bins": (
            float(
                np.median(
                    [
                        item["bins"]
                        for label in STATE_LABELS
                        for item in records[label]
                    ]
                )
            )
            if any(records.values())
            else None
        ),
        "rate_modes": {},
    }
    for mode in RATE_MODES:
        classes = {
            label: _class_summary(records[label], mode)
            for label in STATE_LABELS
        }
        primary = None
        secondary = None
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
                    difference_in_differences(
                        wall["exposure_1_mean_r"],
                        control["exposure_1_mean_r"],
                        wall["exposure_3_mean_r"],
                        control["exposure_3_mean_r"],
                    )
                ),
            }
        if classes["shared_wall"] and classes["wall_open"]:
            wall = classes["shared_wall"]
            control = classes["wall_open"]
            secondary = {
                "exposure_1_wall_minus_wall_open": float(
                    wall["exposure_1_mean_r"]
                    - control["exposure_1_mean_r"]
                ),
                "exposure_3_wall_minus_wall_open": float(
                    wall["exposure_3_mean_r"]
                    - control["exposure_3_mean_r"]
                ),
                "state_consistency_stabilization_did": (
                    difference_in_differences(
                        wall["exposure_1_mean_r"],
                        control["exposure_1_mean_r"],
                        wall["exposure_3_mean_r"],
                        control["exposure_3_mean_r"],
                    )
                ),
            }
        result["rate_modes"][mode] = {
            "classes": classes,
            "primary_shared_wall_vs_shared_open": primary,
            "secondary_shared_wall_vs_wall_open": secondary,
        }
    return result


def _animal_mode_summary(
    pair_summaries: list[dict[str, Any]],
    *,
    mode: str,
    minimum_environment_pairs: int,
) -> dict[str, Any]:
    primary_rows = [
        (
            pair["first_environment"],
            pair["second_environment"],
            pair["rate_modes"][mode][
                "primary_shared_wall_vs_shared_open"
            ],
        )
        for pair in pair_summaries
        if pair["rate_modes"][mode][
            "primary_shared_wall_vs_shared_open"
        ]
        is not None
    ]
    primary = [row[2] for row in primary_rows]
    secondary = [
        pair["rate_modes"][mode]["secondary_shared_wall_vs_wall_open"]
        for pair in pair_summaries
        if pair["rate_modes"][mode][
            "secondary_shared_wall_vs_wall_open"
        ]
        is not None
    ]
    primary_did = [
        item["wall_specific_stabilization_did"] for item in primary
    ]
    secondary_did = [
        item["state_consistency_stabilization_did"] for item in secondary
    ]
    leave_one_pair_out = [
        float(np.mean(np.delete(np.asarray(primary_did), index)))
        for index in range(len(primary_did))
        if len(primary_did) > 1
    ]
    environments = sorted(
        {
            environment
            for first, second, _ in primary_rows
            for environment in (first, second)
        }
    )
    leave_one_environment_out = []
    for heldout in environments:
        kept = [
            item["wall_specific_stabilization_did"]
            for first, second, item in primary_rows
            if heldout not in (first, second)
        ]
        if kept:
            leave_one_environment_out.append(
                {
                    "heldout_environment": heldout,
                    "remaining_pairs": len(kept),
                    "mean_did": float(np.mean(kept)),
                }
            )
    return {
        "primary_comparable_environment_pairs": len(primary),
        "primary_coverage_eligible": (
            len(primary) >= minimum_environment_pairs
        ),
        "primary_exposure_1_wall_minus_open_open": _finite_mean(
            [
                item["exposure_1_wall_minus_open_open"]
                for item in primary
            ]
        ),
        "primary_exposure_3_wall_minus_open_open": _finite_mean(
            [
                item["exposure_3_wall_minus_open_open"]
                for item in primary
            ]
        ),
        "primary_wall_specific_stabilization_did": _finite_mean(
            primary_did
        ),
        "primary_shared_wall_exposure_1_mean_r": _finite_mean(
            [
                item["shared_wall_exposure_1_mean_r"]
                for item in primary
            ]
        ),
        "primary_shared_wall_exposure_3_mean_r": _finite_mean(
            [
                item["shared_wall_exposure_3_mean_r"]
                for item in primary
            ]
        ),
        "primary_shared_wall_exposure_3_minus_1": _finite_mean(
            [
                item["shared_wall_exposure_3_minus_1"]
                for item in primary
            ]
        ),
        "primary_shared_open_exposure_1_mean_r": _finite_mean(
            [
                item["shared_open_exposure_1_mean_r"]
                for item in primary
            ]
        ),
        "primary_shared_open_exposure_3_mean_r": _finite_mean(
            [
                item["shared_open_exposure_3_mean_r"]
                for item in primary
            ]
        ),
        "primary_shared_open_exposure_3_minus_1": _finite_mean(
            [
                item["shared_open_exposure_3_minus_1"]
                for item in primary
            ]
        ),
        "primary_pair_median_did": _finite_median(primary_did),
        "primary_positive_pair_dids": int(
            np.count_nonzero(np.asarray(primary_did) > 0)
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
        "primary_positive_leave_one_pair_out_estimates": int(
            np.count_nonzero(np.asarray(leave_one_pair_out) > 0)
        ),
        "primary_leave_one_environment_out": leave_one_environment_out,
        "primary_positive_leave_one_environment_out_estimates": int(
            np.count_nonzero(
                np.asarray(
                    [item["mean_did"] for item in leave_one_environment_out]
                )
                > 0
            )
        ),
        "secondary_comparable_environment_pairs": len(secondary),
        "secondary_state_consistency_stabilization_did": _finite_mean(
            secondary_did
        ),
        "secondary_pair_median_did": _finite_median(secondary_did),
        "secondary_positive_pair_dids": int(
            np.count_nonzero(np.asarray(secondary_did) > 0)
        ),
    }


def _analyze_strip(
    *,
    seams: list[OrientedSeam],
    depths_cm: tuple[float, ...],
    minimum_bins: int,
    minimum_seconds: float,
    minimum_cells: int,
    minimum_environment_pairs: int,
    rate: dict[int, FloatArray],
    occupancy: dict[int, FloatArray],
    registered: dict[int, NDArray[np.bool_]],
    environment: list[str],
    blocked: list[tuple[int, ...]],
) -> dict[str, Any]:
    strips = {
        seam: seam_strip_bins(seam, depths_cm=depths_cm)
        for seam in seams
    }
    accounting = {
        label: {
            "state_candidates": 0,
            "eligible": 0,
            "insufficient_common_cells": 0,
            "insufficient_common_bins": 0,
            "nonfinite_correlation": 0,
        }
        for label in STATE_LABELS
    }
    pair_summaries = []
    for first_offset, second_offset in itertools.combinations(
        range(1, 10),
        2,
    ):
        records: dict[str, list[dict[str, Any]]] = {
            label: [] for label in STATE_LABELS
        }
        for seam in seams:
            label = state_pair_label(
                seam_state(blocked[first_offset], seam),
                seam_state(blocked[second_offset], seam),
            )
            if label is None:
                continue
            accounting[label]["state_candidates"] += 1
            record, status = _record(
                first_offset=first_offset,
                second_offset=second_offset,
                seam=seam,
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
    return {
        "settings": {
            "depths_cm": list(depths_cm),
            "minimum_common_bins": minimum_bins,
        },
        "record_accounting": accounting,
        "rate_modes": {
            mode: _animal_mode_summary(
                pair_summaries,
                mode=mode,
                minimum_environment_pairs=minimum_environment_pairs,
            )
            for mode in RATE_MODES
        },
        "environment_pairs": pair_summaries,
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
        repetitions = (animal.n_sessions - 1) // 10
        (
            rate,
            occupancy,
            registered,
            environment,
            blocked,
        ) = _session_cache(animal)

    order_by_cycle = [
        environment[cycle * 10 + 1 : cycle * 10 + 10]
        for cycle in range(repetitions)
    ]
    base = {
        "animal": path.name.removesuffix(".complete.mat"),
        "sessions": len(environment),
        "exposure_cycles": repetitions,
        "first_cycle_order": order_by_cycle[0],
        "same_shape_order_repeated": all(
            order == order_by_cycle[0] for order in order_by_cycle[1:]
        ),
    }
    if repetitions < 3:
        return {
            **base,
            "endpoint_eligible": False,
            "exclusion_reason": (
                "fewer than three cycles; exposure 1 versus 3 with "
                "non-overlapping target sessions and square baselines is "
                "not available"
            ),
            "strips": {},
        }

    repeated_geometry = all(
        environment[offset] == environment[20 + offset]
        and blocked[offset] == blocked[20 + offset]
        for offset in range(1, 10)
    )
    if not repeated_geometry:
        raise ValueError(
            f"{path.name}: exposure 1 and 3 geometry/order do not match"
        )

    strips = {
        name: _analyze_strip(
            seams=seams,
            depths_cm=spec["depths_cm"],
            minimum_bins=spec["minimum_bins"],
            minimum_seconds=minimum_seconds,
            minimum_cells=minimum_cells,
            minimum_environment_pairs=minimum_environment_pairs,
            rate=rate,
            occupancy=occupancy,
            registered=registered,
            environment=environment,
            blocked=blocked,
        )
        for name, spec in STRIP_SPECS.items()
    }
    return {
        **base,
        "endpoint_eligible": True,
        "exposure_1_sessions": {
            "pre_square": 1,
            "targets": list(range(2, 11)),
            "post_square": 11,
        },
        "exposure_3_sessions": {
            "pre_square": 21,
            "targets": list(range(22, 31)),
            "post_square": 31,
        },
        "strips": strips,
    }


def _cohort_summary(
    animals: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "animals_loaded": len(animals),
        "animals_with_three_cycles": int(
            np.count_nonzero(
                [animal["endpoint_eligible"] for animal in animals]
            )
        ),
        "animals_excluded_for_two_cycles": [
            animal["animal"]
            for animal in animals
            if not animal["endpoint_eligible"]
        ],
        "strips": {},
    }
    for strip_name in STRIP_SPECS:
        result["strips"][strip_name] = {}
        for mode in RATE_MODES:
            rows = [
                (
                    animal["animal"],
                    animal["strips"][strip_name]["rate_modes"][mode],
                )
                for animal in animals
                if animal["endpoint_eligible"]
            ]
            finite = [
                (name, row)
                for name, row in rows
                if row["primary_wall_specific_stabilization_did"] is not None
            ]
            eligible = [
                (name, row)
                for name, row in finite
                if row["primary_coverage_eligible"]
            ]
            all_values = [
                row["primary_wall_specific_stabilization_did"]
                for _, row in finite
            ]
            eligible_values = [
                row["primary_wall_specific_stabilization_did"]
                for _, row in eligible
            ]
            result["strips"][strip_name][mode] = {
                "endpoint_animals": len(finite),
                "positive_endpoint_animals": int(
                    np.count_nonzero(np.asarray(all_values) > 0)
                ),
                "endpoint_animal_mean_did": _finite_mean(all_values),
                "endpoint_animal_median_did": _finite_median(all_values),
                "coverage_eligible_animals": len(eligible),
                "positive_coverage_eligible_animals": int(
                    np.count_nonzero(np.asarray(eligible_values) > 0)
                ),
                "coverage_eligible_animal_mean_did": _finite_mean(
                    eligible_values
                ),
                "coverage_eligible_animal_median_did": _finite_median(
                    eligible_values
                ),
                "animal_values": {
                    name: row["primary_wall_specific_stabilization_did"]
                    for name, row in finite
                },
                "animal_pair_counts": {
                    name: row["primary_comparable_environment_pairs"]
                    for name, row in finite
                },
                "coverage_eligible_animal_names": [
                    name for name, _ in eligible
                ],
                "population_inference_performed": False,
            }
    return result


def main() -> None:
    argument = parse_arguments()
    paths = sorted(argument.data_dir.glob("*.complete.mat"))
    if not paths:
        raise FileNotFoundError(
            f"no complete animal files found in {argument.data_dir}"
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
        "status": "exploratory_identifiability_audit",
        "question": (
            "From exposure 1 to exposure 3, does exact-location recurrence "
            "increase more for seams walled in both global shapes than for "
            "seams open in both shapes, on identical cells and bins?"
        ),
        "design": {
            "primary_estimand": (
                "(shared-wall minus shared-open recurrence in exposure 3) "
                "minus the same contrast in exposure 1"
            ),
            "aggregation": (
                "mean seam correlations within state and fixed environment "
                "pair, state contrast within pair, then equal-weight mean "
                "over comparable environment pairs within animal"
            ),
            "exposure_endpoints": [1, 3],
            "endpoint_reason": (
                "targets and bracketing square baselines are disjoint; "
                "adjacent exposure comparisons share a square baseline"
            ),
            "same_cells_and_bins_across_endpoints": True,
            "same_environment_pairs_and_oriented_seams_across_endpoints": True,
            "target_rate_maps": "released unsmoothed 5 cm event probability",
            "correlation": "Spearman across registered cells",
            "square_residual": (
                "mean of the two opposite pre/post-square assignments"
            ),
            "open_open_role": (
                "negative-control trajectory for general exposure-related "
                "local-map reliability"
            ),
            "inferential_unit": (
                "mouse; environment-pair and seam records are descriptive"
            ),
            "preexisting_parameters": (
                "minimum dwell, cells, bins, strip depths, and 12-pair "
                "coverage gate are inherited from the static discovery "
                "analysis rather than tuned on this endpoint"
            ),
        },
        "settings": {
            "minimum_seconds_per_common_bin": argument.minimum_seconds,
            "minimum_common_cells": argument.minimum_cells,
            "minimum_comparable_environment_pairs": (
                argument.minimum_environment_pairs
            ),
            "strip_specs": {
                name: {
                    "depths_cm": list(spec["depths_cm"]),
                    "minimum_common_bins": spec["minimum_bins"],
                }
                for name, spec in STRIP_SPECS.items()
            },
        },
        "identifiability_limits": [
            (
                "This is a post-outcome exploratory analysis and cannot be "
                "treated as preregistered confirmation."
            ),
            (
                "Exposure cycle is perfectly confounded with elapsed "
                "recording time and repeated handling; the open/open "
                "difference-in-differences removes a measured general "
                "reliability trajectory but does not prove causal learning."
            ),
            (
                "Only six mice have non-overlapping exposure-1 and "
                "exposure-3 endpoints, and support must be coverage-gated "
                "before interpreting animal directions."
            ),
            (
                "The parallel-trends assumption for wall and open locations "
                "cannot be checked before exposure 1 in this dataset."
            ),
            (
                "Wall state remains confounded with the adjacent partition "
                "being inaccessible."
            ),
            (
                "No population p-value is calculated; all resampling or "
                "record counts below the mouse level would be descriptive."
            ),
        ],
        "cohort": _cohort_summary(animals),
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
        )
    )


if __name__ == "__main__":
    main()
