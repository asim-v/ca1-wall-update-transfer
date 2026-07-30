"""Raw-event split-half validation of reusable local boundary edits.

This script reconstructs released-style unsmoothed 5 cm event-probability
maps from raw position and event traces.  Each session is divided into
contiguous first and second halves.  A comparison between target sessions A
and B symmetrically averages four crossed correlations.  Within each
correlation, its left and right target data and corresponding square baselines
come from non-overlapping halves:

    corr(A0 - pre0,  B1 - post1)
    corr(A0 - post0, B1 - pre1)
    corr(A1 - pre1,  B0 - post0)
    corr(A1 - post1, B0 - pre0)

Raw y is already in the released map's north-to-south row convention, so map
row is floor(raw_y / 5) with no vertical flip.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ca1_geometry.boundary_fragments import (  # noqa: E402
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


FRAME_RATE_HZ = 30.0
N_BINS = 15
BIN_SIZE_CM = 5.0
ARENA_SIZE_CM = 75.0


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
            / "boundary_fragment_raw_split.json"
        ),
    )
    parser.add_argument(
        "--full-map-reference",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "boundary_fragment_screen.json"
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
    parser.add_argument(
        "--minimum-environment-pairs",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--animal",
        action="append",
        default=[],
        help="Optional animal stem; repeat to select multiple animals.",
    )
    return parser.parse_args()


def _aggregate_half(
    position: np.ndarray,
    trace: np.ndarray,
    *,
    total_cells: int,
    registered_cells: np.ndarray,
    accessible_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return released-style unsmoothed rates and dwell seconds for one half."""

    finite_position = np.all(np.isfinite(position), axis=1)
    inside = finite_position & np.all(
        (position >= 0.0) & (position <= ARENA_SIZE_CM),
        axis=1,
    )
    xy = position[inside]
    event = trace[inside]

    # The released generator admitted coordinate 75 into a temporary 16th
    # bin and then cropped to 15 bins.  Directly removing row/column 15 is
    # exactly equivalent for unsmoothed maps.
    row = np.floor(xy[:, 1] / BIN_SIZE_CM).astype(np.int64)
    column = np.floor(xy[:, 0] / BIN_SIZE_CM).astype(np.int64)
    retained = (row < N_BINS) & (column < N_BINS)
    row = row[retained]
    column = column[retained]
    event = event[retained]
    flat_bin = row * N_BINS + column

    sample_count = flat_bin.size
    design = csr_matrix(
        (
            np.ones(sample_count, dtype=np.float64),
            (flat_bin, np.arange(sample_count)),
        ),
        shape=(N_BINS**2, sample_count),
    )
    event_sum = np.asarray(design @ event)
    count = np.bincount(
        flat_bin,
        minlength=N_BINS**2,
    ).astype(np.float64)
    visited = count > 0

    local_rate = np.full(
        (registered_cells.size, N_BINS**2),
        np.nan,
        dtype=np.float64,
    )
    local_rate[:, visited] = (
        event_sum[visited] / count[visited, None]
    ).T
    rate = np.full(
        (total_cells, N_BINS**2),
        np.nan,
        dtype=np.float64,
    )
    rate[registered_cells] = local_rate
    rate = rate.reshape(total_cells, N_BINS, N_BINS)
    # Released sampling maps retain rare tracking samples inside blocked
    # partitions, while released rate maps mask those partitions.  Preserve
    # that asymmetry so the raw reconstruction matches both stored products.
    rate[:, ~accessible_bins] = np.nan
    return rate, (count / FRAME_RATE_HZ).reshape(N_BINS, N_BINS)


def _accessible_bin_mask(blocked: tuple[int, ...]) -> np.ndarray:
    """Return the released-map support implied by blocked 25 cm tiles."""

    result = np.ones((N_BINS, N_BINS), dtype=bool)
    bins_per_partition = N_BINS // 3
    for partition in blocked:
        row, column = divmod(partition, 3)
        result[
            row * bins_per_partition : (row + 1) * bins_per_partition,
            column * bins_per_partition : (column + 1) * bins_per_partition,
        ] = False
    return result


def _session_half_maps(
    animal: Mat73Animal,
    session: int,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
    dict[str, float | int],
]:
    """Build both half maps and audit their full-session recombination."""

    position = animal.position(session)
    registered_index = np.flatnonzero(animal.registered_cells(session))
    trace = animal.trace(session, registered_index)

    # Released cells are expected to be either finite for the whole session
    # or absent (all NaN).  Drop any anomalous partly non-finite columns rather
    # than allowing a frame-varying cell sample into the reconstruction.
    completely_finite = np.all(np.isfinite(trace), axis=0)
    registered_index = registered_index[completely_finite]
    trace = trace[:, completely_finite]
    accessible_bins = _accessible_bin_mask(animal.blocked(session))

    split = position.shape[0] // 2
    ranges = ((0, split), (split, position.shape[0]))
    rate: list[np.ndarray] = []
    occupancy: list[np.ndarray] = []
    for low, high in ranges:
        half_rate, half_occupancy = _aggregate_half(
            position[low:high],
            trace[low:high],
            total_cells=animal.n_cells,
            registered_cells=registered_index,
            accessible_bins=accessible_bins,
        )
        rate.append(half_rate)
        occupancy.append(half_occupancy)

    registered = np.zeros(animal.n_cells, dtype=bool)
    registered[registered_index] = True

    total_occupancy = occupancy[0] + occupancy[1]
    visited = (total_occupancy > 0) & accessible_bins
    combined = np.full_like(rate[0], np.nan)
    numerator = (
        np.nan_to_num(rate[0]) * occupancy[0][None, :, :]
        + np.nan_to_num(rate[1]) * occupancy[1][None, :, :]
    )
    combined[:, visited] = (
        numerator[:, visited] / total_occupancy[visited]
    )

    stored_rate = animal.stored_rate_maps(session, smoothed=False)
    registered_combined = combined[registered_index]
    registered_stored = stored_rate[registered_index]
    finite = np.isfinite(registered_combined) & np.isfinite(
        registered_stored
    )
    rate_error = (
        float(
            np.max(
                np.abs(
                    registered_combined[finite]
                    - registered_stored[finite]
                )
            )
        )
        if np.any(finite)
        else float("nan")
    )
    finite_mismatch = int(
        np.count_nonzero(
            np.isfinite(registered_combined)
            != np.isfinite(registered_stored)
        )
    )
    occupancy_error = float(
        np.max(
            np.abs(
                total_occupancy - animal.sampling_map(session)
            )
        )
    )
    return (
        rate,
        occupancy,
        registered,
        {
            "rate_maximum_absolute_error": rate_error,
            "rate_finite_mask_mismatches": finite_mismatch,
            "occupancy_maximum_absolute_error_seconds": occupancy_error,
            "registered_cells": int(registered_index.size),
        },
    )


def _animal_map_cache(
    animal: Mat73Animal,
) -> tuple[
    dict[int, list[np.ndarray]],
    dict[int, list[np.ndarray]],
    dict[int, np.ndarray],
    dict[str, Any],
]:
    rate: dict[int, list[np.ndarray]] = {}
    occupancy: dict[int, list[np.ndarray]] = {}
    registered: dict[int, np.ndarray] = {}
    audit = []
    for session in range(animal.n_sessions):
        (
            rate[session],
            occupancy[session],
            registered[session],
            session_audit,
        ) = _session_half_maps(animal, session)
        audit.append({"session": session + 1, **session_audit})
    return (
        rate,
        occupancy,
        registered,
        {
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
            "sessions": audit,
        },
    )


def _common_half_support(
    sessions: tuple[int, int, int, int],
    occupancy: dict[int, list[np.ndarray]],
    candidate_bins: tuple[tuple[int, int], ...],
    *,
    minimum_seconds: float,
) -> tuple[tuple[int, int], ...]:
    result = []
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
            result.append((row, column))
    return tuple(result)


def _comparison(
    *,
    sessions: tuple[int, int, int, int],
    seam: OrientedSeam,
    rate: dict[int, list[np.ndarray]],
    occupancy: dict[int, list[np.ndarray]],
    registered: dict[int, np.ndarray],
    strip: dict[OrientedSeam, tuple[tuple[int, int], ...]],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> dict[str, Any] | None:
    pre_square, first_target, second_target, post_square = sessions
    cells = np.flatnonzero(
        registered[pre_square]
        & registered[first_target]
        & registered[second_target]
        & registered[post_square]
    )
    if cells.size < minimum_cells:
        return None
    support = _common_half_support(
        sessions,
        occupancy,
        strip[seam],
        minimum_seconds=minimum_seconds,
    )
    if len(support) < minimum_bins:
        return None

    value = {
        (session, half): local_cell_rate(
            rate[session][half],
            cells,
            support,
        )
        for session in sessions
        for half in (0, 1)
    }
    correlation = np.asarray(
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
    if not np.isfinite(correlation).all():
        return None
    return {
        "correlation": float(np.mean(correlation)),
        "four_crossed_nonoverlapping_side_correlations": (
            correlation.tolist()
        ),
        "cells": int(cells.size),
        "bins": int(len(support)),
    }


def _environment_pair_summary(
    record: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    if not record["shared_wall"] or not record["changed_wall"]:
        return None
    shared = np.asarray(
        [item["correlation"] for item in record["shared_wall"]]
    )
    changed = np.asarray(
        [item["correlation"] for item in record["changed_wall"]]
    )
    all_record = record["shared_wall"] + record["changed_wall"]
    return {
        "shared_wall_mean": float(np.mean(shared)),
        "changed_wall_mean": float(np.mean(changed)),
        "shared_minus_changed": float(
            np.mean(shared) - np.mean(changed)
        ),
        "shared_wall_seams": int(shared.size),
        "changed_wall_seams": int(changed.size),
        "median_cells": float(
            np.median([item["cells"] for item in all_record])
        ),
        "median_bins": float(
            np.median([item["bins"] for item in all_record])
        ),
    }


def _sequence_screen(
    *,
    exposure: int,
    pre_square: int,
    post_square: int,
    rate: dict[int, list[np.ndarray]],
    occupancy: dict[int, list[np.ndarray]],
    registered: dict[int, np.ndarray],
    environment: list[str],
    blocked: list[tuple[int, ...]],
    seams: list[OrientedSeam],
    strip: dict[OrientedSeam, tuple[tuple[int, int], ...]],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
    minimum_environment_pairs: int,
) -> dict[str, Any]:
    pair_summary = []
    for first_target, second_target in itertools.combinations(
        range(pre_square + 1, post_square),
        2,
    ):
        record: dict[str, list[dict[str, Any]]] = {
            "shared_wall": [],
            "changed_wall": [],
        }
        for seam in seams:
            first_state = seam_state(blocked[first_target], seam)
            second_state = seam_state(blocked[second_target], seam)
            if (
                first_state is SeamState.WALL
                and second_state is SeamState.WALL
            ):
                label = "shared_wall"
            elif {first_state, second_state} == {
                SeamState.WALL,
                SeamState.OPEN,
            }:
                label = "changed_wall"
            else:
                continue
            result = _comparison(
                sessions=(
                    pre_square,
                    first_target,
                    second_target,
                    post_square,
                ),
                seam=seam,
                rate=rate,
                occupancy=occupancy,
                registered=registered,
                strip=strip,
                minimum_seconds=minimum_seconds,
                minimum_bins=minimum_bins,
                minimum_cells=minimum_cells,
            )
            if result is not None:
                record[label].append(result)
        summary = _environment_pair_summary(record)
        if summary is not None:
            pair_summary.append(
                {
                    "first_environment": environment[first_target],
                    "second_environment": environment[second_target],
                    **summary,
                }
            )

    eligible = len(pair_summary) >= minimum_environment_pairs
    if pair_summary:
        shared = float(
            np.mean(
                [item["shared_wall_mean"] for item in pair_summary]
            )
        )
        changed = float(
            np.mean(
                [item["changed_wall_mean"] for item in pair_summary]
            )
        )
        contrast = float(
            np.mean(
                [item["shared_minus_changed"] for item in pair_summary]
            )
        )
        seam_records = int(
            sum(
                item["shared_wall_seams"]
                + item["changed_wall_seams"]
                for item in pair_summary
            )
        )
        median_cells = float(
            np.median(
                [item["median_cells"] for item in pair_summary]
            )
        )
        median_bins = float(
            np.median(
                [item["median_bins"] for item in pair_summary]
            )
        )
    else:
        shared = changed = contrast = float("nan")
        seam_records = 0
        median_cells = median_bins = float("nan")
    return {
        "exposure": exposure,
        "pre_square_session": pre_square + 1,
        "post_square_session": post_square + 1,
        "comparable_environment_pairs": len(pair_summary),
        "coverage_eligible": eligible,
        "shared_wall_mean": shared,
        "changed_wall_mean": changed,
        "shared_minus_changed": contrast,
        "valid_seam_records": seam_records,
        "median_cells": median_cells,
        "median_bins": median_bins,
        "environment_pairs": pair_summary,
    }


def _threshold_screen(
    *,
    repetitions: int,
    rate: dict[int, list[np.ndarray]],
    occupancy: dict[int, list[np.ndarray]],
    registered: dict[int, np.ndarray],
    environment: list[str],
    blocked: list[tuple[int, ...]],
    seams: list[OrientedSeam],
    strip: dict[OrientedSeam, tuple[tuple[int, int], ...]],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
    minimum_environment_pairs: int,
) -> dict[str, Any]:
    sequences = [
        _sequence_screen(
            exposure=repetition + 1,
            pre_square=repetition * 10,
            post_square=repetition * 10 + 10,
            rate=rate,
            occupancy=occupancy,
            registered=registered,
            environment=environment,
            blocked=blocked,
            seams=seams,
            strip=strip,
            minimum_seconds=minimum_seconds,
            minimum_bins=minimum_bins,
            minimum_cells=minimum_cells,
            minimum_environment_pairs=minimum_environment_pairs,
        )
        for repetition in range(repetitions)
    ]
    finite = [
        item["shared_minus_changed"]
        for item in sequences
        if np.isfinite(item["shared_minus_changed"])
    ]
    eligible = [
        item["shared_minus_changed"]
        for item in sequences
        if item["coverage_eligible"]
        and np.isfinite(item["shared_minus_changed"])
    ]
    return {
        "minimum_seconds_per_half_bin": minimum_seconds,
        "all_sequence_mean": (
            float(np.mean(finite)) if finite else None
        ),
        "eligible_sequence_mean": (
            float(np.mean(eligible)) if eligible else None
        ),
        "finite_sequences": len(finite),
        "eligible_sequences": len(eligible),
        "positive_finite_sequences": int(
            np.count_nonzero(np.asarray(finite) > 0)
        ),
        "positive_eligible_sequences": int(
            np.count_nonzero(np.asarray(eligible) > 0)
        ),
        "sequences": sequences,
    }


def _reference_animal(
    full_map_reference: dict[str, Any] | None,
    animal_name: str,
) -> dict[str, Any] | None:
    if full_map_reference is None:
        return None
    match = [
        item
        for item in full_map_reference.get("animals", [])
        if item.get("animal") == animal_name
    ]
    if not match:
        return None
    item = match[0]
    return {
        "all_sequence_mean": item.get("all_sequence_mean"),
        "eligible_animal_mean": item.get("eligible_animal_mean"),
        "sequences": [
            {
                "exposure": sequence["exposure"],
                "comparable_environment_pairs": sequence[
                    "comparable_environment_pairs"
                ],
                "coverage_eligible": sequence["coverage_eligible"],
                "shared_wall_mean": sequence["shared_wall_mean"],
                "changed_wall_mean": sequence["changed_wall_mean"],
                "shared_minus_changed": sequence[
                    "shared_minus_changed"
                ],
            }
            for sequence in item.get("sequences", [])
        ],
    }


def analyze_animal(
    path: Path,
    *,
    primary_minimum_seconds: float,
    sensitivity_minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
    minimum_environment_pairs: int,
    full_map_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    seams = internal_seams()
    strip = {seam: seam_strip_bins(seam) for seam in seams}
    with Mat73Animal(path) as animal:
        environment = [
            animal.environment(session)
            for session in range(animal.n_sessions)
        ]
        blocked = [
            animal.blocked(session)
            for session in range(animal.n_sessions)
        ]
        rate, occupancy, registered, audit = _animal_map_cache(animal)
        repetitions = (animal.n_sessions - 1) // 10
        primary = _threshold_screen(
            repetitions=repetitions,
            rate=rate,
            occupancy=occupancy,
            registered=registered,
            environment=environment,
            blocked=blocked,
            seams=seams,
            strip=strip,
            minimum_seconds=primary_minimum_seconds,
            minimum_bins=minimum_bins,
            minimum_cells=minimum_cells,
            minimum_environment_pairs=minimum_environment_pairs,
        )
        sensitivity = _threshold_screen(
            repetitions=repetitions,
            rate=rate,
            occupancy=occupancy,
            registered=registered,
            environment=environment,
            blocked=blocked,
            seams=seams,
            strip=strip,
            minimum_seconds=sensitivity_minimum_seconds,
            minimum_bins=minimum_bins,
            minimum_cells=minimum_cells,
            minimum_environment_pairs=minimum_environment_pairs,
        )
    name = path.name.removesuffix(".complete.mat")
    return {
        "animal": name,
        "sessions": len(environment),
        "reconstruction_audit": audit,
        "primary": primary,
        "sensitivity": sensitivity,
        "full_map_reference": _reference_animal(
            full_map_reference,
            name,
        ),
    }


def _screen_cohort(
    animals: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    finite_sequences = [
        sequence
        for animal in animals
        for sequence in animal[key]["sequences"]
        if np.isfinite(sequence["shared_minus_changed"])
    ]
    eligible_sequences = [
        sequence
        for sequence in finite_sequences
        if sequence["coverage_eligible"]
    ]
    all_animal = [
        animal[key]["all_sequence_mean"]
        for animal in animals
        if animal[key]["all_sequence_mean"] is not None
    ]
    eligible_animal = [
        animal[key]["eligible_sequence_mean"]
        for animal in animals
        if animal[key]["eligible_sequence_mean"] is not None
    ]
    return {
        "finite_sequences": len(finite_sequences),
        "positive_finite_sequences": int(
            np.count_nonzero(
                [
                    item["shared_minus_changed"] > 0
                    for item in finite_sequences
                ]
            )
        ),
        "eligible_sequences": len(eligible_sequences),
        "positive_eligible_sequences": int(
            np.count_nonzero(
                [
                    item["shared_minus_changed"] > 0
                    for item in eligible_sequences
                ]
            )
        ),
        "animals_with_finite_mean": len(all_animal),
        "positive_animals_with_finite_mean": int(
            np.count_nonzero(np.asarray(all_animal) > 0)
        ),
        "all_animal_mean": (
            float(np.mean(all_animal)) if all_animal else None
        ),
        "eligible_animals": len(eligible_animal),
        "positive_eligible_animals": int(
            np.count_nonzero(np.asarray(eligible_animal) > 0)
        ),
        "eligible_animal_mean": (
            float(np.mean(eligible_animal))
            if eligible_animal
            else None
        ),
    }


def _full_map_comparison(
    animals: list[dict[str, Any]],
) -> dict[str, Any] | None:
    split = []
    full = []
    for animal in animals:
        reference = animal["full_map_reference"]
        if reference is None:
            continue
        reference_by_exposure = {
            item["exposure"]: item
            for item in reference["sequences"]
        }
        for sequence in animal["primary"]["sequences"]:
            if (
                not sequence["coverage_eligible"]
                or not np.isfinite(
                    sequence["shared_minus_changed"]
                )
            ):
                continue
            other = reference_by_exposure.get(sequence["exposure"])
            if other is None or not np.isfinite(
                other["shared_minus_changed"]
            ):
                continue
            split.append(sequence["shared_minus_changed"])
            full.append(other["shared_minus_changed"])
    if not split:
        return None
    split_array = np.asarray(split)
    full_array = np.asarray(full)
    return {
        "matched_primary_eligible_blocks": len(split),
        "split_half_block_mean": float(np.mean(split_array)),
        "full_map_block_mean": float(np.mean(full_array)),
        "mean_attenuation": float(
            np.mean(split_array - full_array)
        ),
        "split_to_full_ratio": float(
            np.mean(split_array) / np.mean(full_array)
        ),
        "pearson_across_blocks": float(
            np.corrcoef(split_array, full_array)[0, 1]
        ),
        "spearman_across_blocks": spearman_correlation(
            split_array,
            full_array,
        ),
    }


def cohort_summary(
    animals: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "primary": _screen_cohort(animals, "primary"),
        "sensitivity": _screen_cohort(animals, "sensitivity"),
        "full_map_comparison": _full_map_comparison(animals),
        "maximum_reconstruction_rate_error": float(
            np.max(
                [
                    animal["reconstruction_audit"][
                        "maximum_rate_absolute_error"
                    ]
                    for animal in animals
                ]
            )
        ),
        "total_reconstruction_rate_finite_mask_mismatches": int(
            sum(
                animal["reconstruction_audit"][
                    "total_rate_finite_mask_mismatches"
                ]
                for animal in animals
            )
        ),
        "maximum_reconstruction_occupancy_error_seconds": float(
            np.max(
                [
                    animal["reconstruction_audit"][
                        "maximum_occupancy_absolute_error_seconds"
                    ]
                    for animal in animals
                ]
            )
        ),
    }


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
    if (
        argument.minimum_seconds < 0
        or argument.sensitivity_minimum_seconds < 0
        or argument.minimum_bins <= 0
        or argument.minimum_cells <= 0
        or argument.minimum_environment_pairs <= 0
    ):
        raise ValueError("thresholds and minimum counts must be positive")

    paths = sorted(
        argument.data_dir.glob("QLAK-CA1-*.complete.mat")
    )
    if argument.animal:
        selected = set(argument.animal)
        paths = [
            path
            for path in paths
            if path.name.removesuffix(".complete.mat") in selected
        ]
    if not paths:
        raise FileNotFoundError("no selected complete animal files found")

    full_map_reference = (
        json.loads(
            argument.full_map_reference.read_text(encoding="utf-8")
        )
        if argument.full_map_reference.exists()
        else None
    )
    animals = []
    for path in paths:
        print(f"Reconstructing {path.name}...", flush=True)
        animal = analyze_animal(
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
            full_map_reference=full_map_reference,
        )
        animals.append(animal)
        print(
            animal["animal"],
            animal["primary"]["eligible_sequence_mean"],
            animal["sensitivity"]["eligible_sequence_mean"],
            flush=True,
        )

    report = {
        "status": "exploratory_raw_event_split_half_validation",
        "question": (
            "Does the shared-wall versus same-seam-open advantage survive "
            "temporally independent raw-event map estimates?"
        ),
        "settings": {
            "temporal_split": (
                "contiguous first and second session halves"
            ),
            "map": (
                "released-style unsmoothed 5 cm event probability; "
                "row=floor(raw_y/5), no vertical flip"
            ),
            "primary_minimum_seconds_per_bin_in_each_of_eight_half_maps": (
                argument.minimum_seconds
            ),
            "sensitivity_minimum_seconds_per_bin_in_each_of_eight_half_maps": (
                argument.sensitivity_minimum_seconds
            ),
            "minimum_common_bins": argument.minimum_bins,
            "minimum_common_cells": argument.minimum_cells,
            "minimum_environment_pairs_per_sequence": (
                argument.minimum_environment_pairs
            ),
            "correlation": "spearman across registered cells",
            "four_crossed_nonoverlapping_side_assignments": [
                "A_half0-pre_half0 vs B_half1-post_half1",
                "A_half0-post_half0 vs B_half1-pre_half1",
                "A_half1-pre_half1 vs B_half0-post_half0",
                "A_half1-post_half1 vs B_half0-pre_half0",
            ],
            "aggregation": (
                "symmetric mean of four crossed assignments per seam; "
                "assignments are not treated as independent replicates; "
                "mean seams within wall class; shared-wall minus "
                "changed-wall within environment pair; mean environment "
                "pairs within exposure"
            ),
            "inferential_unit": "animal",
        },
        "cohort": cohort_summary(animals),
        "animals": animals,
    }
    report = _json_safe(report)
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["cohort"], indent=2), flush=True)


if __name__ == "__main__":
    main()
