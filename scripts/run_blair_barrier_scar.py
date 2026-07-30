"""Outcome-locked analysis of a persistent neutral-barrier CA1 trace.

The design is frozen in ``reports/blair_barrier_scar_design_lock.md``.
Only the six neutral-barrier rats are used.  Neural endpoints compare
barrier-free pre-training and post-training sessions; the under-sampled
post-insertion portion of the training session is never analyzed.

For each rat and running direction, raw event-rate maps are reconstructed on
the official 23-bin short route.  Odd/even selected beelines provide a
location-specific within-session reliability control.  The primary score at
each bin is mean within-session Fisher-z reliability minus between-session
Fisher-z similarity.  A "scar" is the score in bins 11--13 minus the
equal-weight score in symmetric three-bin control windows 5--7 and 17--19.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    ROOT
    / "data"
    / "raw"
    / "blair_et_al"
    / "tadblair"
    / "Blair_et_al_DATA"
)

N_BINS = 23
MINIMUM_CELLS = 20
RECONSTRUCTION_TOLERANCE = 1e-12
CORRELATION_CLIP = 1.0 - 1e-12

# Session numbers are copied from rows 7--12 in the official Figure 2/3 code.
BARRIER_CASES = (
    {
        "rat": "Hipp12",
        "pre_session": 5,
        "training_session": 6,
        "post_session": 9,
    },
    {
        "rat": "Hipp13",
        "pre_session": 5,
        "training_session": 6,
        "post_session": 9,
    },
    {
        "rat": "Hipp18",
        "pre_session": 6,
        "training_session": 7,
        "post_session": 9,
    },
    {
        "rat": "Hipp35",
        "pre_session": 7,
        "training_session": 8,
        "post_session": 11,
    },
    {
        "rat": "Hipp30",
        "pre_session": 5,
        "training_session": 6,
        "post_session": 9,
    },
    {
        "rat": "Hipp34",
        "pre_session": 7,
        "training_session": 8,
        "post_session": 11,
    },
)

# Zero-based Python indices for the frozen one-based MATLAB regions.
CENTER = (10, 11, 12)
CONTROL_LEFT = (4, 5, 6)
CONTROL_RIGHT = (16, 17, 18)
OFFICIAL_CENTER = tuple(range(8, 15))
OFFICIAL_LEFT = tuple(range(1, 8))
OFFICIAL_RIGHT = tuple(range(15, 22))
RADIAL_RINGS = {
    "center_11_13": ((10, 11, 12),),
    "inner_8_10_14_16": ((7, 8, 9), (13, 14, 15)),
    "control_5_7_17_19": ((4, 5, 6), (16, 17, 18)),
    "outer_2_4_20_22": ((1, 2, 3), (19, 20, 21)),
}

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
Method = Literal["pearson", "spearman"]


@dataclass(frozen=True)
class DirectionMaps:
    """Full and odd/even event-rate maps for one running direction."""

    full: FloatArray
    odd: FloatArray
    even: FloatArray
    trips: int
    frames_full: IntArray
    frames_odd: IntArray
    frames_even: IntArray
    frame_interval_ms: float


@dataclass(frozen=True)
class SessionMaps:
    """Reconstructed maps and schema audit for one session."""

    local_activity: BoolArray
    directions: dict[str, DirectionMaps]
    event_field: str
    candidate_reconstruction: dict[str, dict[str, float]]
    official_vmap_reconstruction: dict[str, dict[str, float]]
    time_normalization: str
    processed_cells: int
    raw_cells: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "blair_barrier_scar.json"
        ),
    )
    parser.add_argument("--minimum-cells", type=int, default=MINIMUM_CELLS)
    return parser.parse_args()


def _load_struct(path: Path, name: str) -> dict[str, Any]:
    """Load one MATLAB structure as a plain dictionary."""

    if not path.exists():
        raise FileNotFoundError(path)
    value = loadmat(path, simplify_cells=True)
    if name not in value or not isinstance(value[name], dict):
        raise ValueError(f"{path.name} has no scalar structure {name!r}")
    return value[name]


def normalize_intervals(value: Any, n_frames: int) -> IntArray:
    """Return validated zero-based inclusive interval endpoints."""

    intervals = np.asarray(value, dtype=np.int64)
    if intervals.ndim == 1:
        intervals = intervals.reshape(1, -1)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("beeline intervals must have shape (trips, 2)")
    intervals = intervals - 1
    if (
        np.any(intervals < 0)
        or np.any(intervals[:, 0] > intervals[:, 1])
        or np.any(intervals[:, 1] >= n_frames)
    ):
        raise ValueError("invalid MATLAB-one-based beeline interval")
    return intervals


def interval_mask(
    n_frames: int,
    intervals: IntArray,
    trip_indices: Iterable[int] | None = None,
) -> BoolArray:
    """Return a frame mask for inclusive beeline intervals."""

    selected = (
        range(len(intervals)) if trip_indices is None else trip_indices
    )
    mask = np.zeros(n_frames, dtype=np.bool_)
    for index in selected:
        start, stop = intervals[index]
        mask[start : stop + 1] = True
    return mask


def _frames_by_bin(mask: BoolArray, position_bin: IntArray) -> IntArray:
    return np.asarray(
        [
            np.count_nonzero(mask & (position_bin == bin_number))
            for bin_number in range(1, N_BINS + 1)
        ],
        dtype=np.int64,
    )


def _rates(
    events: FloatArray,
    mask: BoolArray,
    position_bin: IntArray,
    occupancy_ms: FloatArray,
) -> FloatArray:
    """Compute cells-by-bins event rates in events per millisecond."""

    result = np.full((events.shape[0], N_BINS), np.nan, dtype=np.float64)
    for bin_index in range(N_BINS):
        selected = mask & (position_bin == bin_index + 1)
        if occupancy_ms[bin_index] > 0 and np.any(selected):
            result[:, bin_index] = (
                np.sum(events[:, selected], axis=1)
                / occupancy_ms[bin_index]
            )
    return result


def _direction_candidate(
    *,
    events: FloatArray,
    position_bin: IntArray,
    intervals: IntArray,
    frame_interval_ms: float,
) -> DirectionMaps:
    """Construct full and alternating-trip maps for one event field."""

    n_frames = position_bin.size
    full_mask = interval_mask(n_frames, intervals)
    odd_mask = interval_mask(
        n_frames,
        intervals,
        range(0, len(intervals), 2),
    )
    even_mask = interval_mask(
        n_frames,
        intervals,
        range(1, len(intervals), 2),
    )
    if np.any(odd_mask & even_mask) or not np.array_equal(
        odd_mask | even_mask,
        full_mask,
    ):
        raise AssertionError("alternating trip halves do not partition trips")

    full_frames = _frames_by_bin(full_mask, position_bin)
    valid = full_frames > 0
    if not np.any(valid):
        raise ValueError("a selected direction has no sampled route bin")

    odd_frames = _frames_by_bin(odd_mask, position_bin)
    even_frames = _frames_by_bin(even_mask, position_bin)
    return DirectionMaps(
        full=_rates(
            events,
            full_mask,
            position_bin,
            full_frames.astype(np.float64) * frame_interval_ms,
        ),
        odd=_rates(
            events,
            odd_mask,
            position_bin,
            odd_frames.astype(np.float64) * frame_interval_ms,
        ),
        even=_rates(
            events,
            even_mask,
            position_bin,
            even_frames.astype(np.float64) * frame_interval_ms,
        ),
        trips=len(intervals),
        frames_full=full_frames,
        frames_odd=odd_frames,
        frames_even=even_frames,
        frame_interval_ms=frame_interval_ms,
    )


def load_session_maps(
    data_dir: Path,
    *,
    rat: str,
    session: int,
    role: Literal["pre", "post"],
) -> SessionMaps:
    """Load and reconstruct one released pre/post session."""

    processed_name = f"{role}data"
    processed = _load_struct(
        data_dir
        / "prepost"
        / f"{rat}_linear{session}_{processed_name}.mat",
        processed_name,
    )
    frame = _load_struct(
        data_dir / "sessiondata" / f"{rat}_linear{session}_sess.mat",
        f"frame{session}",
    )
    position_bin = np.asarray(frame["posbin"], dtype=np.int64).ravel()
    if (
        position_bin.size == 0
        or np.min(position_bin) != 1
        or np.max(position_bin) != N_BINS
    ):
        raise ValueError(f"{rat} session {session}: invalid position bins")
    time = np.asarray(frame["time"], dtype=np.float64).ravel()
    positive_time_steps = np.diff(time)
    positive_time_steps = positive_time_steps[
        np.isfinite(positive_time_steps) & (positive_time_steps > 0)
    ]
    if not positive_time_steps.size:
        raise ValueError(f"{rat} session {session}: no positive time steps")
    frame_interval_ms = float(np.mean(positive_time_steps))

    candidate: dict[str, dict[str, DirectionMaps]] = {}
    reconstruction: dict[str, dict[str, float]] = {}
    official_reconstruction: dict[str, dict[str, float]] = {}
    for event_field in ("S", "deconv"):
        events = np.asarray(frame[event_field], dtype=np.float64)
        if events.ndim != 2 or events.shape[1] != position_bin.size:
            raise ValueError(
                f"{rat} session {session}: invalid {event_field} shape"
            )
        candidate[event_field] = {}
        reconstruction[event_field] = {}
        official_reconstruction[event_field] = {}
        for direction in ("LR", "RL"):
            intervals = normalize_intervals(
                processed[f"{direction}int"],
                position_bin.size,
            )
            maps = _direction_candidate(
                events=events,
                position_bin=position_bin,
                intervals=intervals,
                frame_interval_ms=frame_interval_ms,
            )
            stored = np.asarray(
                processed[f"dcurve_{direction}"],
                dtype=np.float64,
            )
            if stored.shape != maps.full.shape:
                raise ValueError(
                    f"{rat} session {session}: processed/raw cell mismatch"
                )
            # Released curves encode some unsampled bins as zero.  Compare
            # only bins whose official occupancy is positive; missing bins
            # remain NaN in all downstream profiles and are never imputed.
            difference = np.abs(maps.full - stored)
            reconstruction[event_field][direction] = float(
                np.nanmax(difference)
            )
            full_mask = interval_mask(position_bin.size, intervals)
            official_full = _rates(
                events,
                full_mask,
                position_bin,
                np.asarray(
                    frame[f"vmap_{direction}"],
                    dtype=np.float64,
                ).ravel(),
            )
            official_reconstruction[event_field][direction] = float(
                np.nanmax(np.abs(official_full - stored))
            )
            candidate[event_field][direction] = maps

    field_error = {
        field: max(direction_error.values())
        for field, direction_error in reconstruction.items()
    }
    chosen_field = min(field_error, key=field_error.get)
    if field_error[chosen_field] > RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"{rat} session {session}: no event field reconstructs released "
            f"maps within {RECONSTRUCTION_TOLERANCE:g}; errors={field_error}"
        )

    chosen_events = np.asarray(frame[chosen_field], dtype=np.float64)
    selected_mask = np.zeros(position_bin.size, dtype=np.bool_)
    for direction in ("LR", "RL"):
        intervals = normalize_intervals(
            processed[f"{direction}int"],
            position_bin.size,
        )
        selected_mask |= interval_mask(position_bin.size, intervals)
    activity = np.sum(chosen_events[:, selected_mask], axis=1) > 0

    return SessionMaps(
        local_activity=activity,
        directions=candidate[chosen_field],
        event_field=chosen_field,
        candidate_reconstruction=reconstruction,
        official_vmap_reconstruction=official_reconstruction,
        time_normalization=(
            "selected frame count times mean positive interframe interval"
        ),
        processed_cells=int(
            np.asarray(processed["dcurve_LR"]).shape[0]
        ),
        raw_cells=int(chosen_events.shape[0]),
    )


def matched_cell_indices(
    data_dir: Path,
    *,
    rat: str,
    pre_session: int,
    post_session: int,
) -> tuple[IntArray, IntArray, dict[str, Any]]:
    """Return session-local zero-based indices from a CellReg map."""

    path = data_dir / "cellmaps" / f"{rat}_barrier_cmap.mat"
    value = loadmat(path, simplify_cells=True)
    cmap = np.asarray(value["cmap"], dtype=np.int64)
    session_numbers = np.asarray(
        value["sessionNums"],
        dtype=np.int64,
    ).ravel()
    pre_columns = np.flatnonzero(session_numbers == pre_session)
    post_columns = np.flatnonzero(session_numbers == post_session)
    if pre_columns.size != 1 or post_columns.size != 1:
        raise ValueError(
            f"{rat}: pre/post sessions do not uniquely enter CellReg map"
        )
    pre_values = cmap[:, pre_columns[0]]
    post_values = cmap[:, post_columns[0]]
    rows = (pre_values > 0) & (post_values > 0)
    return (
        pre_values[rows].astype(np.int64) - 1,
        post_values[rows].astype(np.int64) - 1,
        {
            "cellreg_rows": int(cmap.shape[0]),
            "cellreg_sessions": session_numbers.tolist(),
            "matched_pre_post_cells": int(np.count_nonzero(rows)),
            "pre_column_one_based": int(pre_columns[0] + 1),
            "post_column_one_based": int(post_columns[0] + 1),
        },
    )


def correlation(
    first: FloatArray,
    second: FloatArray,
    *,
    method: Method,
) -> float:
    """Finite Pearson or Spearman correlation with no inferential p-value."""

    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()
    finite = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(finite) < 3:
        return float("nan")
    first = first[finite]
    second = second[finite]
    if method == "spearman":
        first = rankdata(first, method="average")
        second = rankdata(second, method="average")
    elif method != "pearson":
        raise ValueError(f"unknown correlation method: {method}")
    first = first - np.mean(first)
    second = second - np.mean(second)
    denominator = float(
        np.sqrt(np.dot(first, first) * np.dot(second, second))
    )
    if denominator == 0:
        return float("nan")
    return float(np.dot(first, second) / denominator)


def fisher_z(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(
        np.arctanh(np.clip(value, -CORRELATION_CLIP, CORRELATION_CLIP))
    )


def direction_profile(
    pre: DirectionMaps,
    post: DirectionMaps,
    pre_cells: IntArray,
    post_cells: IntArray,
    *,
    method: Method,
) -> dict[str, list[float]]:
    """Compute full and reliability-adjusted bin profiles."""

    between = []
    within_pre = []
    within_post = []
    excess_change = []
    for bin_index in range(N_BINS):
        between_r = correlation(
            pre.full[pre_cells, bin_index],
            post.full[post_cells, bin_index],
            method=method,
        )
        pre_r = correlation(
            pre.odd[pre_cells, bin_index],
            pre.even[pre_cells, bin_index],
            method=method,
        )
        post_r = correlation(
            post.odd[post_cells, bin_index],
            post.even[post_cells, bin_index],
            method=method,
        )
        between.append(between_r)
        within_pre.append(pre_r)
        within_post.append(post_r)
        excess_change.append(
            float(
                (fisher_z(pre_r) + fisher_z(post_r)) / 2.0
                - fisher_z(between_r)
            )
        )
    return {
        "between_pre_post_r": between,
        "within_pre_odd_even_r": within_pre,
        "within_post_odd_even_r": within_post,
        "excess_change_fisher_z": excess_change,
    }


def _strict_mean(values: FloatArray, indices: tuple[int, ...]) -> float:
    selected = np.asarray(values, dtype=np.float64)[list(indices)]
    if selected.size != len(indices) or not np.isfinite(selected).all():
        return float("nan")
    return float(np.mean(selected))


def _equal_window_mean(
    values: FloatArray,
    windows: tuple[tuple[int, ...], ...],
) -> float:
    window_values = np.asarray(
        [_strict_mean(values, window) for window in windows],
        dtype=np.float64,
    )
    if not np.isfinite(window_values).all():
        return float("nan")
    return float(np.mean(window_values))


def spatial_summary(profile: FloatArray) -> dict[str, Any]:
    """Apply the frozen center/control windows to one bin profile."""

    center = _strict_mean(profile, CENTER)
    left = _strict_mean(profile, CONTROL_LEFT)
    right = _strict_mean(profile, CONTROL_RIGHT)
    control = float((left + right) / 2.0)
    official_center = _strict_mean(profile, OFFICIAL_CENTER)
    official_left = _strict_mean(profile, OFFICIAL_LEFT)
    official_right = _strict_mean(profile, OFFICIAL_RIGHT)
    ring_values = {
        label: _equal_window_mean(profile, windows)
        for label, windows in RADIAL_RINGS.items()
    }
    finite_rings = np.asarray(list(ring_values.values()), dtype=np.float64)
    ring_slope = (
        float(np.polyfit(np.arange(4), finite_rings, 1)[0])
        if np.isfinite(finite_rings).all()
        else None
    )
    return {
        "former_barrier_bins_one_based": [11, 12, 13],
        "left_control_bins_one_based": [5, 6, 7],
        "right_control_bins_one_based": [17, 18, 19],
        "former_barrier_mean": center,
        "left_control_mean": left,
        "right_control_mean": right,
        "equal_weight_control_mean": control,
        "primary_center_minus_control": float(center - control),
        "official_zone_center_minus_equal_sides": float(
            official_center - (official_left + official_right) / 2.0
        ),
        "radial_ring_values": ring_values,
        "radial_ring_linear_slope": ring_slope,
    }


def analyze_mode(
    pre: SessionMaps,
    post: SessionMaps,
    pre_cells: IntArray,
    post_cells: IntArray,
    *,
    method: Method,
) -> dict[str, Any]:
    direction = {
        name: direction_profile(
            pre.directions[name],
            post.directions[name],
            pre_cells,
            post_cells,
            method=method,
        )
        for name in ("LR", "RL")
    }
    direction_score = np.asarray(
        [
            direction[name]["excess_change_fisher_z"]
            for name in ("LR", "RL")
        ],
        dtype=np.float64,
    )
    # Require both directions at a bin; no direction is silently substituted.
    averaged_score = np.where(
        np.isfinite(direction_score).all(axis=0),
        np.mean(direction_score, axis=0),
        np.nan,
    )
    between_z = np.asarray(
        [
            [
                fisher_z(value)
                for value in direction[name]["between_pre_post_r"]
            ]
            for name in ("LR", "RL")
        ],
        dtype=np.float64,
    )
    within_z = np.asarray(
        [
            [
                [
                    fisher_z(value)
                    for value in direction[name][field]
                ]
                for field in (
                    "within_pre_odd_even_r",
                    "within_post_odd_even_r",
                )
            ]
            for name in ("LR", "RL")
        ],
        dtype=np.float64,
    )
    averaged_between_similarity = np.where(
        np.isfinite(between_z).all(axis=0),
        np.mean(between_z, axis=0),
        np.nan,
    )
    averaged_within_reliability = np.where(
        np.isfinite(within_z).all(axis=(0, 1)),
        np.mean(within_z, axis=(0, 1)),
        np.nan,
    )
    averaged_between_change = np.where(
        np.isfinite(between_z).all(axis=0),
        -np.mean(between_z, axis=0),
        np.nan,
    )
    return {
        "method": method,
        "directions": {
            name: {
                **direction[name],
                "spatial": spatial_summary(
                    np.asarray(
                        direction[name]["excess_change_fisher_z"],
                        dtype=np.float64,
                    )
                ),
            }
            for name in ("LR", "RL")
        },
        "direction_averaged_excess_change_fisher_z": (
            averaged_score.tolist()
        ),
        "primary_spatial": spatial_summary(averaged_score),
        "between_similarity_spatial": spatial_summary(
            averaged_between_similarity
        ),
        "within_reliability_spatial": spatial_summary(
            averaged_within_reliability
        ),
        "between_only_spatial": spatial_summary(
            averaged_between_change
        ),
    }


def analyze_rat(
    data_dir: Path,
    case: dict[str, Any],
    *,
    minimum_cells: int,
) -> dict[str, Any]:
    rat = str(case["rat"])
    pre_session = int(case["pre_session"])
    post_session = int(case["post_session"])
    pre = load_session_maps(
        data_dir,
        rat=rat,
        session=pre_session,
        role="pre",
    )
    post = load_session_maps(
        data_dir,
        rat=rat,
        session=post_session,
        role="post",
    )
    if pre.event_field != post.event_field:
        raise ValueError(f"{rat}: pre/post event-field convention differs")
    if pre.time_normalization != post.time_normalization:
        raise ValueError(f"{rat}: pre/post time normalization differs")

    pre_matched, post_matched, cellreg_audit = matched_cell_indices(
        data_dir,
        rat=rat,
        pre_session=pre_session,
        post_session=post_session,
    )
    if (
        np.any(pre_matched >= pre.processed_cells)
        or np.any(post_matched >= post.processed_cells)
    ):
        raise ValueError(f"{rat}: CellReg index exceeds session cell count")
    active_rows = (
        pre.local_activity[pre_matched]
        & post.local_activity[post_matched]
    )
    pre_active = pre_matched[active_rows]
    post_active = post_matched[active_rows]

    audit = {
        "event_field_selected_by_whole_map_reconstruction": (
            pre.event_field
        ),
        "pre_candidate_max_absolute_errors": (
            pre.candidate_reconstruction
        ),
        "post_candidate_max_absolute_errors": (
            post.candidate_reconstruction
        ),
        "pre_official_vmap_max_absolute_errors": (
            pre.official_vmap_reconstruction
        ),
        "post_official_vmap_max_absolute_errors": (
            post.official_vmap_reconstruction
        ),
        "time_normalization_selected_by_whole_map_reconstruction": (
            pre.time_normalization
        ),
        "pre_processed_raw_cell_counts": [
            pre.processed_cells,
            pre.raw_cells,
        ],
        "post_processed_raw_cell_counts": [
            post.processed_cells,
            post.raw_cells,
        ],
        "trip_counts": {
            "pre_LR": pre.directions["LR"].trips,
            "pre_RL": pre.directions["RL"].trips,
            "post_LR": post.directions["LR"].trips,
            "post_RL": post.directions["RL"].trips,
        },
        "minimum_frames_per_bin_by_half": {
            f"{role}_{direction}_{half}": int(
                np.min(
                    getattr(
                        value.directions[direction],
                        f"frames_{half}",
                    )
                )
            )
            for role, value in (("pre", pre), ("post", post))
            for direction in ("LR", "RL")
            for half in ("odd", "even")
        },
        "minimum_frames_per_analysis_bin_2_to_22_by_half": {
            f"{role}_{direction}_{half}": int(
                np.min(
                    getattr(
                        value.directions[direction],
                        f"frames_{half}",
                    )[1:22]
                )
            )
            for role, value in (("pre", pre), ("post", post))
            for direction in ("LR", "RL")
            for half in ("odd", "even")
        },
        "minimum_frames_per_primary_window_bin_by_half": {
            f"{role}_{direction}_{half}": int(
                np.min(
                    getattr(
                        value.directions[direction],
                        f"frames_{half}",
                    )[
                        list(
                            CENTER
                            + CONTROL_LEFT
                            + CONTROL_RIGHT
                        )
                    ]
                )
            )
            for role, value in (("pre", pre), ("post", post))
            for direction in ("LR", "RL")
            for half in ("odd", "even")
        },
        **cellreg_audit,
        "matched_cells_active_in_both_sessions": int(pre_active.size),
    }

    primary_eligible = pre_active.size >= minimum_cells
    modes: dict[str, Any] = {}
    if primary_eligible:
        modes["pearson_active"] = analyze_mode(
            pre,
            post,
            pre_active,
            post_active,
            method="pearson",
        )
        modes["spearman_active"] = analyze_mode(
            pre,
            post,
            pre_active,
            post_active,
            method="spearman",
        )
    if pre_matched.size >= minimum_cells:
        modes["pearson_all_matched"] = analyze_mode(
            pre,
            post,
            pre_matched,
            post_matched,
            method="pearson",
        )

    primary_value = (
        modes["pearson_active"]["primary_spatial"][
            "primary_center_minus_control"
        ]
        if "pearson_active" in modes
        else None
    )
    if primary_value is not None and not np.isfinite(primary_value):
        primary_eligible = False
    return {
        **case,
        "primary_eligible": bool(primary_eligible),
        "exclusion_reason": (
            None
            if primary_eligible
            else "fewer than minimum cells or nonfinite primary windows"
        ),
        "audit": audit,
        "modes": modes,
    }


def exact_sign_flip(values: list[float]) -> dict[str, Any]:
    """Enumerate exact sign flips of the animal-mean statistic."""

    observed = float(np.mean(values))
    null = np.asarray(
        [
            np.mean(np.asarray(values) * np.asarray(signs))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ],
        dtype=np.float64,
    )
    tolerance = 1e-15
    return {
        "permutations": int(null.size),
        "observed_mean": observed,
        "one_sided_positive_tail": float(
            np.mean(null >= observed - tolerance)
        ),
        "two_sided_absolute_tail": float(
            np.mean(np.abs(null) >= abs(observed) - tolerance)
        ),
        "assumption": (
            "animal effects are exchangeable under sign reversal; exact "
            "descriptive randomization tail, not a substitute for replication"
        ),
    }


def _cohort_metric(
    animals: list[dict[str, Any]],
    *,
    mode: str,
    path: tuple[str, ...],
) -> dict[str, Any]:
    values: dict[str, float] = {}
    for animal in animals:
        if mode not in animal["modes"]:
            continue
        value: Any = animal["modes"][mode]
        for key in path:
            value = value[key]
        if np.isfinite(value):
            values[animal["rat"]] = float(value)
    array = np.asarray(list(values.values()), dtype=np.float64)
    leave_one_out = (
        [
            float(np.mean(np.delete(array, index)))
            for index in range(array.size)
        ]
        if array.size > 1
        else []
    )
    return {
        "animals": int(array.size),
        "positive_animals": int(np.count_nonzero(array > 0)),
        "animal_mean": float(np.mean(array)) if array.size else None,
        "animal_median": float(np.median(array)) if array.size else None,
        "animal_values": values,
        "leave_one_animal_out_min_mean_max": (
            [
                float(np.min(leave_one_out)),
                float(np.mean(leave_one_out)),
                float(np.max(leave_one_out)),
            ]
            if leave_one_out
            else None
        ),
        "exact_sign_flip": exact_sign_flip(array.tolist())
        if array.size
        else None,
    }


def cohort_summary(animals: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rats_loaded": len(animals),
        "primary_eligible_rats": int(
            np.count_nonzero(
                [animal["primary_eligible"] for animal in animals]
            )
        ),
        "modes": {},
    }
    for mode in (
        "pearson_active",
        "spearman_active",
        "pearson_all_matched",
    ):
        result["modes"][mode] = {
            "primary_center_minus_control": _cohort_metric(
                animals,
                mode=mode,
                path=(
                    "primary_spatial",
                    "primary_center_minus_control",
                ),
            ),
            "official_zone_center_minus_sides": _cohort_metric(
                animals,
                mode=mode,
                path=(
                    "primary_spatial",
                    "official_zone_center_minus_equal_sides",
                ),
            ),
            "between_only_center_minus_control": _cohort_metric(
                animals,
                mode=mode,
                path=(
                    "between_only_spatial",
                    "primary_center_minus_control",
                ),
            ),
            "between_similarity_center_minus_control": _cohort_metric(
                animals,
                mode=mode,
                path=(
                    "between_similarity_spatial",
                    "primary_center_minus_control",
                ),
            ),
            "within_reliability_center_minus_control": _cohort_metric(
                animals,
                mode=mode,
                path=(
                    "within_reliability_spatial",
                    "primary_center_minus_control",
                ),
            ),
            "radial_ring_linear_slope": _cohort_metric(
                animals,
                mode=mode,
                path=(
                    "primary_spatial",
                    "radial_ring_linear_slope",
                ),
            ),
            "LR_primary_center_minus_control": _cohort_metric(
                animals,
                mode=mode,
                path=(
                    "directions",
                    "LR",
                    "spatial",
                    "primary_center_minus_control",
                ),
            ),
            "RL_primary_center_minus_control": _cohort_metric(
                animals,
                mode=mode,
                path=(
                    "directions",
                    "RL",
                    "spatial",
                    "primary_center_minus_control",
                ),
            ),
        }
    primary = result["modes"]["pearson_active"][
        "primary_center_minus_control"
    ]
    result["locked_support_decision"] = {
        "minimum_four_eligible": primary["animals"] >= 4,
        "positive_cohort_mean": (
            primary["animal_mean"] is not None
            and primary["animal_mean"] > 0
        ),
        "minimum_four_positive": primary["positive_animals"] >= 4,
    }
    result["locked_support_decision"]["passes_all"] = all(
        result["locked_support_decision"].values()
    )
    return result


def json_safe(value: Any) -> Any:
    """Recursively replace nonfinite scalars with strict-JSON nulls."""

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    argument = parse_arguments()
    animals = []
    for case in BARRIER_CASES:
        print(f"Analyzing {case['rat']}...", flush=True)
        animals.append(
            analyze_rat(
                argument.data_dir,
                case,
                minimum_cells=argument.minimum_cells,
            )
        )
        primary = animals[-1]["modes"].get("pearson_active", {})
        value = primary.get("primary_spatial", {}).get(
            "primary_center_minus_control"
        )
        print(
            case["rat"],
            animals[-1]["audit"][
                "matched_cells_active_in_both_sessions"
            ],
            value,
            flush=True,
        )

    selected_fields = {
        animal["audit"][
            "event_field_selected_by_whole_map_reconstruction"
        ]
        for animal in animals
    }
    report = {
        "status": "locked_external_barrier_scar_test",
        "question": (
            "Does a transient neutral barrier leave excess registered-cell "
            "CA1 population change at its former coordinate after removal?"
        ),
        "design_lock": "reports/blair_barrier_scar_design_lock.md",
        "source": {
            "paper_doi": "10.7554/eLife.80661",
            "dataset_doi": "10.5068/D1ZT2S",
            "official_readme": (
                "data/raw/blair_et_al/tadblair/README.md"
            ),
            "official_code": (
                "tmp/blair-et-al-code/Blair_et_al_MATLAB/"
                "Figure3_analysis.m"
            ),
        },
        "design": {
            "cohort": "all six neutral-barrier rats in official rows 7-12",
            "neural_sessions": (
                "CellReg-matched barrier-free pre and post sessions"
            ),
            "training_post_insertion_neural_data_used": False,
            "former_barrier_coordinate": (
                "central route bin 12 from official 23-bin scheme"
            ),
            "primary_window_one_based": [11, 12, 13],
            "control_windows_one_based": [[5, 6, 7], [17, 18, 19]],
            "population": (
                "CellReg matched and active on selected beelines in both "
                "sessions; no place-field selection"
            ),
            "within_session_control": (
                "alternating selected beelines within direction"
            ),
            "primary_metric": (
                "mean within-session Fisher-z population correlation minus "
                "pre/post Fisher-z correlation"
            ),
            "aggregation": (
                "bins within fixed windows, LR/RL within rat, then equal "
                "rat weights"
            ),
            "inferential_unit": "rat",
            "population_inference": (
                "exact animal-level sign-flip tails reported descriptively"
            ),
        },
        "schema_audit": {
            "candidate_event_fields": ["S", "deconv"],
            "selected_fields_across_rats": sorted(selected_fields),
            "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
            "bin_width_discrepancy": (
                "paper/README imply about 10.8 cm while official MATLAB code "
                "uses 13.33 cm interior bins; indexed bins are used"
            ),
        },
        "settings": {
            "minimum_active_matched_cells": argument.minimum_cells,
            "route_bins": N_BINS,
            "correlation_clip": CORRELATION_CLIP,
        },
        "cohort": cohort_summary(animals),
        "animals": animals,
        "interpretation_limits": [
            (
                "The source paper already analyzed broad center/side "
                "position-resolved remapping; novelty is limited to the "
                "locked narrow within-barrier reliability-adjusted contrast."
            ),
            (
                "Barrier exposure is confounded with forced detour and "
                "additional elapsed experience."
            ),
            (
                "Only six rats exist, so exact animal-level resolution is "
                "coarse and no cell/bin pseudoreplication is used."
            ),
            (
                "A positive persistent trace is not an acute onset measure "
                "or a direct test of prediction error."
            ),
        ],
    }
    report = json_safe(report)
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
