"""Behavior-adjusted validation of exact-location CA1 wall-conditioned profiles.

For each session, this script estimates 5 cm spatial-bin fixed effects jointly
with speed and allocentric movement-direction covariates. It then repeats the
target-rate-held-out cross-exposure analysis from
``run_boundary_component_validation.py``: other geometries in one exposure
cycle train exact-seam wall and open templates, and the target geometry's local
registered-cell rate-change vector is tested in the next cycle with an
independent outer-square baseline. The target wall label, occupancy support,
and registration mask are known when constructing that test vector.

The analysis is exploratory.  Raw experimental files remain outside Git; the
compact JSON result is safe to track.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ca1_geometry.boundary_fragments import (  # noqa: E402
    common_support_bins,
    local_cell_rate,
    spearman_correlation,
)
from ca1_geometry.arena import spatial_accessibility  # noqa: E402
from ca1_geometry.io import Mat73Animal  # noqa: E402
from ca1_geometry.seams import (  # noqa: E402
    OrientedSeam,
    SeamState,
    internal_seams,
    seam_state,
    seam_strip_bins,
)
from run_boundary_component_validation import (  # noqa: E402
    nonfocal_target_context,
)


MODE_NO_TIME = "speed_movement_direction"
MODE_WITH_TIME = "speed_movement_direction_time"
MODES = (MODE_NO_TIME, MODE_WITH_TIME)


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
            / "boundary_fragment_behavior_adjusted.json"
        ),
    )
    parser.add_argument("--frames-per-second", type=float, default=30.0)
    parser.add_argument("--velocity-half-window-frames", type=int, default=3)
    parser.add_argument("--speed-cap-quantile", type=float, default=0.995)
    parser.add_argument(
        "--minimum-direction-speed-cm-s",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--behavior-ridge-fraction",
        type=float,
        default=0.001,
        help=(
            "Ridge penalty as a fraction of the mean diagonal of the "
            "within-bin behavior Gram matrix."
        ),
    )
    parser.add_argument(
        "--trace-cell-chunk",
        type=int,
        default=64,
        help=(
            "Maximum registered-cell traces loaded together. Chunking changes "
            "only memory use; every cell uses the same session-level fit."
        ),
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-bins", type=int, default=6)
    parser.add_argument("--minimum-cells", type=int, default=20)
    parser.add_argument(
        "--match-nonfocal-context",
        action="store_true",
    )
    parser.add_argument(
        "--match-global-counterfactual",
        action="store_true",
    )
    parser.add_argument(
        "--animal",
        action="append",
        default=[],
        help=(
            "Analyze only this animal stem (for example QLAK-CA1-08). "
            "May be supplied more than once."
        ),
    )
    parser.add_argument(
        "--animal-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for validated per-animal checkpoints. "
            "Completed animals are reused only when their input-file "
            "fingerprint and every analysis setting match."
        ),
    )
    return parser.parse_args()


def _central_velocity(
    position: np.ndarray,
    *,
    frames_per_second: float,
    half_window_frames: int,
) -> np.ndarray:
    """Estimate velocity with a centered finite difference."""

    sample = np.arange(position.shape[0])
    before = np.maximum(sample - half_window_frames, 0)
    after = np.minimum(sample + half_window_frames, position.shape[0] - 1)
    elapsed = (after - before) / frames_per_second
    if np.any(elapsed <= 0):
        raise ValueError("a session must contain at least two position frames")
    return (position[after] - position[before]) / elapsed[:, None]


def _behavior_design(
    position: np.ndarray,
    valid: np.ndarray,
    *,
    frames_per_second: float,
    half_window_frames: int,
    speed_cap_cm_s: float,
    minimum_direction_speed_cm_s: float,
) -> np.ndarray:
    """Return speed, movement-direction harmonics, and session time."""

    velocity = _central_velocity(
        position,
        frames_per_second=frames_per_second,
        half_window_frames=half_window_frames,
    )[valid]
    raw_speed = np.linalg.norm(velocity, axis=1)
    speed = np.minimum(raw_speed, speed_cap_cm_s)

    moving = speed >= minimum_direction_speed_cm_s
    cosine = np.zeros(speed.size, dtype=np.float64)
    sine = np.zeros(speed.size, dtype=np.float64)
    safe_speed = np.maximum(raw_speed, np.finfo(np.float64).eps)
    cosine[moving] = velocity[moving, 0] / safe_speed[moving]
    sine[moving] = velocity[moving, 1] / safe_speed[moving]

    if position.shape[0] == 1:
        session_time = np.zeros(1, dtype=np.float64)
    else:
        session_time = np.linspace(
            -1.0,
            1.0,
            position.shape[0],
            dtype=np.float64,
        )
    return np.column_stack(
        (
            speed,
            cosine,
            sine,
            cosine**2 - sine**2,
            2.0 * cosine * sine,
            session_time[valid],
        )
    )


def _adjusted_maps(
    position: np.ndarray,
    response: np.ndarray,
    analysis_valid: np.ndarray,
    standardized_design: np.ndarray,
    standardized_reference: np.ndarray,
    *,
    frames_per_second: float,
    ridge_fraction: float,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, dict[str, float]],
]:
    """Estimate behavior-adjusted bin intercepts by within-bin regression.

    Frisch-Waugh-Lovell residualization removes each covariate's 5 cm bin
    mean before estimating cell-wise behavior coefficients.  The adjusted
    map is the bin mean after subtracting the fitted covariate contribution,
    evaluated at one animal-level physical reference shared by all sessions.
    """

    if position.ndim != 2 or position.shape[1] != 2:
        raise ValueError("position must have shape (frame, 2)")
    if response.ndim != 2 or response.shape[0] != position.shape[0]:
        raise ValueError("response must have shape (frame, cell)")
    if not np.isfinite(position).all() or not np.isfinite(response).all():
        raise ValueError("position and registered-cell response must be finite")

    physical_valid = np.all(
        (position >= 0.0) & (position < 75.0),
        axis=1,
    )
    selected = np.asarray(analysis_valid, dtype=bool)
    if selected.shape != (position.shape[0],):
        raise ValueError("analysis_valid must have one value per frame")
    if np.any(selected & ~physical_valid):
        raise ValueError("analysis-valid frames must lie inside the arena")

    raw_xy = position[physical_valid]
    raw_event = response[physical_valid]
    raw_x_bin = np.floor(raw_xy[:, 0] / 5.0).astype(np.int64)
    raw_y_bin = np.floor(raw_xy[:, 1] / 5.0).astype(np.int64)
    raw_flat_bin = raw_y_bin * 15 + raw_x_bin
    raw_aggregate = csr_matrix(
        (
            np.ones(raw_flat_bin.size, dtype=np.float64),
            (raw_flat_bin, np.arange(raw_flat_bin.size)),
        ),
        shape=(225, raw_flat_bin.size),
    )
    raw_count = np.asarray(raw_aggregate.sum(axis=1)).ravel()
    raw_event_sum = np.asarray(raw_aggregate @ raw_event)
    raw_event_mean = np.divide(
        raw_event_sum,
        raw_count[:, None],
        out=np.full_like(raw_event_sum, np.nan),
        where=raw_count[:, None] > 0,
    )

    xy = position[selected]
    event = response[selected]
    if xy.shape[0] == 0:
        raise ValueError("session has no frames inside the released map")

    design = np.asarray(standardized_design, dtype=np.float64)
    reference = np.asarray(standardized_reference, dtype=np.float64)
    if design.shape != (xy.shape[0], 6) or reference.shape != (6,):
        raise ValueError("behavior design or common reference has wrong shape")
    x_bin = np.floor(xy[:, 0] / 5.0).astype(np.int64)
    # Raw trajectory y and released-map rows use the same image-style order.
    y_bin = np.floor(xy[:, 1] / 5.0).astype(np.int64)
    flat_bin = y_bin * 15 + x_bin

    aggregate = csr_matrix(
        (
            np.ones(flat_bin.size, dtype=np.float64),
            (flat_bin, np.arange(flat_bin.size)),
        ),
        shape=(225, flat_bin.size),
    )
    count = np.asarray(aggregate.sum(axis=1)).ravel()
    behavior_sum = np.asarray(aggregate @ design)
    event_sum = np.asarray(aggregate @ event)
    behavior_mean = np.divide(
        behavior_sum,
        count[:, None],
        out=np.zeros_like(behavior_sum),
        where=count[:, None] > 0,
    )
    event_mean = np.divide(
        event_sum,
        count[:, None],
        out=np.full_like(event_sum, np.nan),
        where=count[:, None] > 0,
    )

    maps: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    for mode, n_covariates in (
        (MODE_NO_TIME, 5),
        (MODE_WITH_TIME, 6),
    ):
        x = design[:, :n_covariates]
        bin_x = behavior_mean[:, :n_covariates]
        within_x = x - bin_x[flat_bin]
        gram = within_x.T @ within_x
        condition = float(np.linalg.cond(gram))
        ridge = float(ridge_fraction * np.trace(gram) / n_covariates)
        beta = np.linalg.solve(
            gram + ridge * np.eye(n_covariates),
            within_x.T @ event,
        )
        # Every session is evaluated at the same animal-level physical
        # reference, not at its own mean kinematics.
        intercept = event_mean - (
            bin_x - reference[:n_covariates]
        ) @ beta
        intercept[count == 0] = np.nan
        maps[mode] = intercept.T.reshape(response.shape[1], 15, 15)
        diagnostics[mode] = {
            "condition_number": condition,
            "ridge_penalty": ridge,
        }

    occupancy = count.reshape(15, 15) / frames_per_second
    raw_map = raw_event_mean.T.reshape(response.shape[1], 15, 15)
    raw_occupancy = raw_count.reshape(15, 15) / frames_per_second
    diagnostics["trajectory"] = {
        "inside_frames": int(np.count_nonzero(physical_valid)),
        "analysis_frames": int(np.count_nonzero(selected)),
        "blocked_tile_frames_excluded": int(
            np.count_nonzero(physical_valid & ~selected)
        ),
    }
    return maps, occupancy, raw_map, raw_occupancy, diagnostics


def _mean_edit(
    sessions: list[int],
    *,
    baseline: int,
    rates: dict[int, np.ndarray],
    cells: np.ndarray,
    bins: tuple[tuple[int, int], ...],
) -> np.ndarray:
    baseline_value = local_cell_rate(rates[baseline], cells, bins)
    return np.mean(
        np.stack(
            [
                local_cell_rate(rates[session], cells, bins)
                - baseline_value
                for session in sessions
            ]
        ),
        axis=0,
    )


def _target_edit(
    session: int,
    *,
    baseline: int,
    rates: dict[int, np.ndarray],
    cells: np.ndarray,
    bins: tuple[tuple[int, int], ...],
) -> np.ndarray:
    return (
        local_cell_rate(rates[session], cells, bins)
        - local_cell_rate(rates[baseline], cells, bins)
    )


def _orientation(seam: OrientedSeam) -> str:
    return (
        "vertical_wall"
        if abs(seam.source - seam.target) == 1
        else "horizontal_wall"
    )


def _prediction_record(
    *,
    target_offset: int,
    training_exposure: int,
    environment: list[str],
    blocked: list[tuple[int, ...]],
    rates: dict[str, dict[int, np.ndarray]],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    seam: OrientedSeam,
    strip: tuple[tuple[int, int], ...],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
    match_nonfocal_context: bool = False,
    match_global_counterfactual: bool = False,
) -> dict[str, Any] | None:
    """Predict one next-exposure target edit from other geometries."""

    train_start = training_exposure * 10
    test_start = (training_exposure + 1) * 10
    training_square = train_start
    test_square = (training_exposure + 2) * 10
    withheld_training = train_start + target_offset
    target_test = test_start + target_offset
    target_context = nonfocal_target_context(
        blocked[target_test],
        seam,
    )

    matching_wall = [
        train_start + offset
        for offset in range(1, 10)
        if offset != target_offset
        and seam_state(blocked[train_start + offset], seam)
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
        and seam_state(blocked[train_start + offset], seam)
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
        training_square,
        test_square,
        target_test,
        *matching_wall,
        *matching_open,
    ]
    cells = np.flatnonzero(
        np.logical_and.reduce([registered[session] for session in required])
    )
    if cells.size < minimum_cells:
        return None
    support = common_support_bins(
        [occupancy[session] for session in required],
        strip,
        minimum_seconds=minimum_seconds,
    )
    if len(support) < minimum_bins:
        return None

    correlations = {}
    for mode in MODES:
        wall = _mean_edit(
            matching_wall,
            baseline=training_square,
            rates=rates[mode],
            cells=cells,
            bins=support,
        )
        open_value = _mean_edit(
            matching_open,
            baseline=training_square,
            rates=rates[mode],
            cells=cells,
            bins=support,
        )
        target = _target_edit(
            target_test,
            baseline=test_square,
            rates=rates[mode],
            cells=cells,
            bins=support,
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

    return {
        "training_exposure": training_exposure + 1,
        "test_exposure": training_exposure + 2,
        "target_environment": environment[target_test],
        "withheld_training_session": withheld_training + 1,
        "test_session": target_test + 1,
        "training_square_session": training_square + 1,
        "test_square_session": test_square + 1,
        "seam": [seam.source, seam.target],
        "orientation": _orientation(seam),
        "nonfocal_context_matched": match_nonfocal_context,
        "global_counterfactual_matched": match_global_counterfactual,
        "matching_wall_environments": [
            environment[session] for session in matching_wall
        ],
        "matching_open_environments": [
            environment[session] for session in matching_open
        ],
        "cells": int(cells.size),
        "bins": int(len(support)),
        "correlations": correlations,
    }


def _summarize_records(
    records: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    if not records:
        return {
            "records": 0,
            "matching_wall_mean_r": None,
            "matching_open_mean_r": None,
            "wall_minus_open_mean": None,
            "positive_records": 0,
            "median_cells": None,
            "median_bins": None,
            "orientation": {
                label: {
                    "records": 0,
                    "mean_wall_minus_open": None,
                }
                for label in ("vertical_wall", "horizontal_wall")
            },
        }
    wall = np.asarray(
        [
            record["correlations"][mode]["matching_wall_r"]
            for record in records
        ]
    )
    open_value = np.asarray(
        [
            record["correlations"][mode]["matching_open_r"]
            for record in records
        ]
    )
    contrast = wall - open_value
    orientation = {}
    for label in ("vertical_wall", "horizontal_wall"):
        selected = np.asarray(
            [record["orientation"] == label for record in records]
        )
        orientation[label] = {
            "records": int(np.count_nonzero(selected)),
            "mean_wall_minus_open": (
                float(np.mean(contrast[selected]))
                if np.any(selected)
                else None
            ),
        }
    return {
        "records": len(records),
        "matching_wall_mean_r": float(np.mean(wall)),
        "matching_open_mean_r": float(np.mean(open_value)),
        "wall_minus_open_mean": float(np.mean(contrast)),
        "positive_records": int(np.count_nonzero(contrast > 0)),
        "median_cells": float(
            np.median([record["cells"] for record in records])
        ),
        "median_bins": float(
            np.median([record["bins"] for record in records])
        ),
        "orientation": orientation,
    }


def analyze_animal(
    path: Path,
    *,
    frames_per_second: float,
    half_window_frames: int,
    speed_cap_quantile: float,
    minimum_direction_speed_cm_s: float,
    ridge_fraction: float,
    trace_cell_chunk: int,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
    match_nonfocal_context: bool = False,
    match_global_counterfactual: bool = False,
) -> dict[str, Any]:
    seams = internal_seams()
    strip = {seam: seam_strip_bins(seam) for seam in seams}
    rates: dict[str, dict[int, np.ndarray]] = {
        mode: {} for mode in MODES
    }
    occupancy: dict[int, np.ndarray] = {}
    registered: dict[int, np.ndarray] = {}
    session_diagnostics = []
    maximum_rate_error = 0.0
    maximum_occupancy_error = 0.0
    maximum_raw_mask_mismatch = 0
    maximum_adjusted_mask_mismatch = {mode: 0 for mode in MODES}

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
        if environment[0] != "square" or blocked[0]:
            raise ValueError(
                "session 1 must be the unblocked familiar-square calibration"
            )
        for offset in range(1, 10):
            repeated_environment = {
                environment[10 * repetition + offset]
                for repetition in range(repetitions)
            }
            repeated_blocked = {
                blocked[10 * repetition + offset]
                for repetition in range(repetitions)
            }
            if (
                len(repeated_environment) != 1
                or len(repeated_blocked) != 1
            ):
                raise ValueError(
                    "target offsets must repeat the same geometry each cycle"
                )

        position = {
            session: animal.position(session)
            for session in range(animal.n_sessions)
        }
        physical_valid = {
            session: np.all(
                (position[session] >= 0.0) & (position[session] < 75.0),
                axis=1,
            )
            for session in range(animal.n_sessions)
        }
        accessible = {
            session: spatial_accessibility(blocked[session])
            for session in range(animal.n_sessions)
        }
        analysis_valid = {}
        for session in range(animal.n_sessions):
            selected = np.zeros(position[session].shape[0], dtype=bool)
            xy = position[session][physical_valid[session]]
            x_bin = np.floor(xy[:, 0] / 5.0).astype(np.int64)
            y_bin = np.floor(xy[:, 1] / 5.0).astype(np.int64)
            selected[physical_valid[session]] = accessible[session][
                y_bin,
                x_bin,
            ]
            analysis_valid[session] = selected
        raw_speed = {}
        for session in range(animal.n_sessions):
            velocity = _central_velocity(
                position[session],
                frames_per_second=frames_per_second,
                half_window_frames=half_window_frames,
            )
            raw_speed[session] = np.linalg.norm(
                velocity[analysis_valid[session]],
                axis=1,
            )
        calibration_speed = raw_speed[0]
        animal_speed_cap = float(
            np.quantile(calibration_speed, speed_cap_quantile)
        )
        animal_speed_reference = float(np.median(calibration_speed))

        raw_design = {}
        for session in range(animal.n_sessions):
            value = _behavior_design(
                position[session],
                analysis_valid[session],
                frames_per_second=frames_per_second,
                half_window_frames=half_window_frames,
                speed_cap_cm_s=animal_speed_cap,
                minimum_direction_speed_cm_s=minimum_direction_speed_cm_s,
            )
            raw_design[session] = value
        # Freeze all nuisance calibration from the initial familiar square.
        # No deformation-session position or neural activity sets the scale
        # or the physical reference used for subsequent adjusted maps.
        design_mean = np.mean(raw_design[0], axis=0)
        design_scale = np.std(raw_design[0], axis=0)
        design_scale[
            design_scale <= np.finfo(np.float64).eps
        ] = 1.0
        # One physical reference is used for every session in this animal:
        # median locomotor speed, direction-averaged movement, and mid-session.
        physical_reference = np.array(
            [animal_speed_reference, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        standardized_reference = (
            physical_reference - design_mean
        ) / design_scale

        for session in range(animal.n_sessions):
            candidate_cells = np.flatnonzero(
                animal.registered_cells(session)
            )
            cell_parts = []
            map_parts = {mode: [] for mode in MODES}
            reconstruction_parts = []
            session_occupancy = None
            raw_occupancy = None
            diagnostics = None
            standardized_design = (
                raw_design[session] - design_mean
            ) / design_scale
            for start in range(0, candidate_cells.size, trace_cell_chunk):
                chunk_cells = candidate_cells[
                    start : start + trace_cell_chunk
                ]
                response = animal.trace(session, chunk_cells)
                fully_finite = np.all(np.isfinite(response), axis=0)
                chunk_cells = chunk_cells[fully_finite]
                response = response[:, fully_finite]
                if chunk_cells.size == 0:
                    continue
                (
                    chunk_maps,
                    chunk_occupancy,
                    chunk_reconstruction,
                    chunk_raw_occupancy,
                    chunk_diagnostics,
                ) = _adjusted_maps(
                    position[session],
                    response,
                    analysis_valid[session],
                    standardized_design,
                    standardized_reference,
                    frames_per_second=frames_per_second,
                    ridge_fraction=ridge_fraction,
                )
                cell_parts.append(chunk_cells)
                reconstruction_parts.append(chunk_reconstruction)
                for mode in MODES:
                    map_parts[mode].append(chunk_maps[mode])
                if session_occupancy is None:
                    session_occupancy = chunk_occupancy
                    raw_occupancy = chunk_raw_occupancy
                    diagnostics = chunk_diagnostics
                elif (
                    not np.array_equal(
                        session_occupancy,
                        chunk_occupancy,
                    )
                    or not np.array_equal(
                        raw_occupancy,
                        chunk_raw_occupancy,
                    )
                ):
                    raise RuntimeError(
                        "cell chunks produced inconsistent occupancy"
                    )
            if not cell_parts:
                raise ValueError(
                    f"session {session + 1} has no fully finite traces"
                )
            cells = np.concatenate(cell_parts)
            session_maps = {
                mode: np.concatenate(map_parts[mode], axis=0)
                for mode in MODES
            }
            raw_reconstruction = np.concatenate(
                reconstruction_parts,
                axis=0,
            )
            if (
                session_occupancy is None
                or raw_occupancy is None
                or diagnostics is None
            ):
                raise RuntimeError("missing chunked session outputs")
            registration = np.zeros(animal.n_cells, dtype=bool)
            registration[cells] = True
            registered[session] = registration

            occupancy[session] = session_occupancy
            released_reconstruction = raw_reconstruction.copy()
            released_reconstruction[:, ~accessible[session]] = np.nan
            for mode in MODES:
                full = np.full(
                    (animal.n_cells, 15, 15),
                    np.nan,
                    dtype=np.float64,
                )
                full[cells] = session_maps[mode]
                full[:, ~accessible[session]] = np.nan
                rates[mode][session] = full

            stored_rate = animal.stored_rate_maps(
                session,
                smoothed=False,
            )[cells]
            raw_mask_mismatch = int(
                np.count_nonzero(
                    np.isfinite(stored_rate)
                    != np.isfinite(released_reconstruction)
                )
            )
            pre_mask_blocked_finite_values = int(
                np.count_nonzero(
                    np.isfinite(raw_reconstruction)[
                        :,
                        ~accessible[session],
                    ]
                )
            )
            expected_adjusted_finite = np.broadcast_to(
                session_occupancy[None, :, :] > 0,
                raw_reconstruction.shape,
            )
            adjusted_mask_mismatch = {
                mode: int(
                    np.count_nonzero(
                        np.isfinite(session_maps[mode])
                        != expected_adjusted_finite
                    )
                )
                for mode in MODES
            }
            common = np.isfinite(stored_rate) & np.isfinite(
                released_reconstruction
            )
            rate_error = float(
                np.max(
                    np.abs(
                        stored_rate[common] - released_reconstruction[common]
                    )
                )
            )
            occupancy_error = float(
                np.max(
                    np.abs(
                        animal.sampling_map(session) - raw_occupancy
                    )
                )
            )
            maximum_rate_error = max(maximum_rate_error, rate_error)
            maximum_occupancy_error = max(
                maximum_occupancy_error,
                occupancy_error,
            )
            maximum_raw_mask_mismatch = max(
                maximum_raw_mask_mismatch,
                raw_mask_mismatch,
            )
            for mode in MODES:
                maximum_adjusted_mask_mismatch[mode] = max(
                    maximum_adjusted_mask_mismatch[mode],
                    adjusted_mask_mismatch[mode],
                )
            session_diagnostics.append(
                {
                    "session": session + 1,
                    "environment": environment[session],
                    "registered_cells": int(cells.size),
                    "median_raw_speed_cm_s": float(
                        np.median(raw_speed[session])
                    ),
                    "fraction_speed_capped": float(
                        np.mean(raw_speed[session] > animal_speed_cap)
                    ),
                    "inside_frames": diagnostics["trajectory"][
                        "inside_frames"
                    ],
                    "analysis_frames": diagnostics["trajectory"][
                        "analysis_frames"
                    ],
                    "blocked_tile_frames_excluded": diagnostics[
                        "trajectory"
                    ]["blocked_tile_frames_excluded"],
                    "blocked_partition_dwell_excluded_seconds": float(
                        np.sum(raw_occupancy[~accessible[session]])
                    ),
                    "pre_mask_blocked_rate_finite_values": (
                        pre_mask_blocked_finite_values
                    ),
                    "condition_number": {
                        mode: diagnostics[mode]["condition_number"]
                        for mode in MODES
                    },
                    "ridge_penalty": {
                        mode: diagnostics[mode]["ridge_penalty"]
                        for mode in MODES
                    },
                    "released_rate_reconstruction_max_abs_error": (
                        rate_error
                    ),
                    "released_rate_finite_mask_mismatches": (
                        raw_mask_mismatch
                    ),
                    "adjusted_rate_finite_mask_mismatches": (
                        adjusted_mask_mismatch
                    ),
                    "occupancy_reconstruction_max_abs_error_seconds": (
                        occupancy_error
                    ),
                }
            )

    records = []
    for training_exposure in range(repetitions - 1):
        for target_offset in range(1, 10):
            target_test = (training_exposure + 1) * 10 + target_offset
            for seam in seams:
                if (
                    seam_state(blocked[target_test], seam)
                    is not SeamState.WALL
                ):
                    continue
                record = _prediction_record(
                    target_offset=target_offset,
                    training_exposure=training_exposure,
                    environment=environment,
                    blocked=blocked,
                    rates=rates,
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
                if record is not None:
                    records.append(record)

    summary = {
        mode: _summarize_records(records, mode=mode)
        for mode in MODES
    }
    condition = {
        mode: {
            "median": float(
                np.median(
                    [
                        item["condition_number"][mode]
                        for item in session_diagnostics
                    ]
                )
            ),
            "maximum": float(
                np.max(
                    [
                        item["condition_number"][mode]
                        for item in session_diagnostics
                    ]
                )
            ),
        }
        for mode in MODES
    }
    exposure_pairs = {}
    for training_exposure in range(repetitions - 1):
        label = f"{training_exposure + 1}_to_{training_exposure + 2}"
        selected = [
            record
            for record in records
            if record["training_exposure"] == training_exposure + 1
        ]
        exposure_pairs[label] = {
            mode: _summarize_records(selected, mode=mode)
            for mode in MODES
        }

    return {
        "animal": path.name.removesuffix(".complete.mat"),
        "sessions": len(environment),
        "records": len(records),
        "summary": summary,
        "no_time_sign_flip": bool(
            np.sign(summary[MODE_NO_TIME]["wall_minus_open_mean"])
            != np.sign(summary[MODE_WITH_TIME]["wall_minus_open_mean"])
        ),
        "conditioning": condition,
        "familiar_square_behavior_calibration": {
            "session": 1,
            "environment": environment[0],
            "speed_cap_cm_s": animal_speed_cap,
            "reference_speed_cm_s": animal_speed_reference,
            "reference_movement_direction_harmonics": [0.0, 0.0, 0.0, 0.0],
            "reference_normalized_session_time": 0.0,
            "calibration_covariate_mean": design_mean.tolist(),
            "calibration_covariate_scale": design_scale.tolist(),
        },
        "reconstruction": {
            "maximum_rate_absolute_error": maximum_rate_error,
            "maximum_occupancy_absolute_error_seconds": (
                maximum_occupancy_error
            ),
            "maximum_released_rate_finite_mask_mismatches": (
                maximum_raw_mask_mismatch
            ),
            "maximum_adjusted_rate_finite_mask_mismatches": (
                maximum_adjusted_mask_mismatch
            ),
        },
        "exposure_pairs": exposure_pairs,
        "session_diagnostics": session_diagnostics,
        "prediction_records": records,
    }


def cohort_summary(animals: list[dict[str, Any]]) -> dict[str, Any]:
    modes = {}
    for mode in MODES:
        values = np.asarray(
            [
                animal["summary"][mode]["wall_minus_open_mean"]
                for animal in animals
            ]
        )
        modes[mode] = {
            "animals": int(values.size),
            "positive_animals": int(np.count_nonzero(values > 0)),
            "animal_mean_wall_minus_open": float(np.mean(values)),
            "animal_median_wall_minus_open": float(np.median(values)),
            "animal_values": {
                animal["animal"]: (
                    animal["summary"][mode]["wall_minus_open_mean"]
                )
                for animal in animals
            },
        }
    return {
        "modes": modes,
        "animals_flipping_when_time_added": [
            animal["animal"]
            for animal in animals
            if animal["no_time_sign_flip"]
        ],
        "maximum_released_rate_reconstruction_error": float(
            np.max(
                [
                    animal["reconstruction"][
                        "maximum_rate_absolute_error"
                    ]
                    for animal in animals
                ]
            )
        ),
        "maximum_occupancy_reconstruction_error_seconds": float(
            np.max(
                [
                    animal["reconstruction"][
                        "maximum_occupancy_absolute_error_seconds"
                    ]
                    for animal in animals
                ]
            )
        ),
        "maximum_released_rate_finite_mask_mismatches": int(
            np.max(
                [
                    animal["reconstruction"][
                        "maximum_released_rate_finite_mask_mismatches"
                    ]
                    for animal in animals
                ]
            )
        ),
        "maximum_adjusted_rate_finite_mask_mismatches": {
            mode: int(
                np.max(
                    [
                        animal["reconstruction"][
                            "maximum_adjusted_rate_finite_mask_mismatches"
                        ][mode]
                        for animal in animals
                    ]
                )
            )
            for mode in MODES
        },
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


def _cache_metadata(
    path: Path,
    argument: argparse.Namespace,
) -> dict[str, Any]:
    """Return the exact input fingerprint and settings guarded by a cache."""

    stat = path.stat()
    return {
        "schema": 1,
        "source": {
            "name": path.name,
            "size_bytes": stat.st_size,
            "modified_time_ns": stat.st_mtime_ns,
        },
        "settings": {
            "frames_per_second": argument.frames_per_second,
            "velocity_half_window_frames": (
                argument.velocity_half_window_frames
            ),
            "speed_cap_quantile": argument.speed_cap_quantile,
            "minimum_direction_speed_cm_s": (
                argument.minimum_direction_speed_cm_s
            ),
            "behavior_ridge_fraction": argument.behavior_ridge_fraction,
            "trace_cell_chunk": argument.trace_cell_chunk,
            "minimum_seconds": argument.minimum_seconds,
            "minimum_bins": argument.minimum_bins,
            "minimum_cells": argument.minimum_cells,
            "match_nonfocal_context": argument.match_nonfocal_context,
            "match_global_counterfactual": (
                argument.match_global_counterfactual
            ),
        },
    }


def _read_cached_animal(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Load a checkpoint only when its complete provenance still matches."""

    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if payload.get("cache_metadata") != expected_metadata:
        return None
    result = payload.get("animal_result")
    if not isinstance(result, dict):
        return None
    return result


def _write_cached_animal(
    cache_path: Path,
    metadata: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Atomically checkpoint one expensive completed animal analysis."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(
                {
                    "cache_metadata": metadata,
                    "animal_result": result,
                }
            ),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(cache_path)


def main() -> None:
    argument = parse_arguments()
    if argument.frames_per_second <= 0:
        raise ValueError("frames per second must be positive")
    if argument.velocity_half_window_frames <= 0:
        raise ValueError("velocity half-window must be positive")
    if not 0 < argument.speed_cap_quantile <= 1:
        raise ValueError("speed cap quantile must lie in (0, 1]")
    if argument.minimum_direction_speed_cm_s < 0:
        raise ValueError("minimum direction speed must be non-negative")
    if argument.behavior_ridge_fraction < 0:
        raise ValueError("behavior ridge fraction must be non-negative")
    if argument.trace_cell_chunk <= 0:
        raise ValueError("trace cell chunk must be positive")

    paths = sorted(
        argument.data_dir.glob("QLAK-CA1-*.complete.mat")
    )
    if not paths:
        raise FileNotFoundError(
            f"no complete animal files found beneath {argument.data_dir}"
        )
    if argument.animal:
        requested = {
            name.removesuffix(".complete.mat")
            for name in argument.animal
        }
        available = {
            path.name.removesuffix(".complete.mat"): path
            for path in paths
        }
        missing = sorted(requested - available.keys())
        if missing:
            raise ValueError(
                "requested animal files were not found: "
                + ", ".join(missing)
            )
        paths = [available[name] for name in sorted(requested)]

    animals = []
    for path in paths:
        metadata = _cache_metadata(path, argument)
        cache_path = (
            argument.animal_cache_dir
            / f"{path.name.removesuffix('.complete.mat')}.json"
            if argument.animal_cache_dir is not None
            else None
        )
        result = (
            _read_cached_animal(cache_path, metadata)
            if cache_path is not None
            else None
        )
        reused_cache = result is not None
        if result is None:
            result = analyze_animal(
                path,
                frames_per_second=argument.frames_per_second,
                half_window_frames=argument.velocity_half_window_frames,
                speed_cap_quantile=argument.speed_cap_quantile,
                minimum_direction_speed_cm_s=(
                    argument.minimum_direction_speed_cm_s
                ),
                ridge_fraction=argument.behavior_ridge_fraction,
                trace_cell_chunk=argument.trace_cell_chunk,
                minimum_seconds=argument.minimum_seconds,
                minimum_bins=argument.minimum_bins,
                minimum_cells=argument.minimum_cells,
                match_nonfocal_context=argument.match_nonfocal_context,
                match_global_counterfactual=(
                    argument.match_global_counterfactual
                ),
            )
            if cache_path is not None:
                _write_cached_animal(
                    cache_path,
                    metadata,
                    result,
                )
        animals.append(result)
        print(
            result["animal"],
            {
                mode: round(
                    result["summary"][mode]["wall_minus_open_mean"],
                    4,
                )
                for mode in MODES
            },
            {"cache": "reused" if reused_cache else "computed"},
            flush=True,
        )

    report = {
        "status": "exploratory_behavior_adjusted_cross_exposure_validation",
        "question": (
            "Does an exact-location wall-conditioned cell profile predict "
            "a target local registered-cell rate-change vector in the next "
            "exposure after jointly adjusting 5 cm spatial rates for speed, "
            "allocentric movement direction, and session time?"
        ),
        "design": {
            "spatial_estimator": (
                "5 cm bin fixed effects estimated jointly with behavior by "
                "within-bin Frisch-Waugh-Lovell residualization"
            ),
            "behavior_covariates": [
                "speed",
                "cos_movement_direction",
                "sin_movement_direction",
                "cos_2movement_direction",
                "sin_2movement_direction",
                "linear_session_time",
            ],
            "movement_direction_when_speed_below_threshold": (
                "all movement-direction harmonics set to zero before "
                "standardization"
            ),
            "common_physical_reference": (
                "initial familiar-square median speed, zero "
                "movement-direction harmonics, and normalized session time "
                "zero for every session"
            ),
            "blocked_partition_handling": (
                "adjusted rates masked and analysis occupancy set to zero "
                "from authoritative blocked-partition metadata"
            ),
            "target_neural_map_excluded_from_training_templates": True,
            "target_wall_label_is_test_aware": True,
            "target_occupancy_and_registration_support_is_test_aware": True,
            "nonfocal_target_context_matched": (
                argument.match_nonfocal_context
            ),
            "global_counterfactual_training_pair_matched": (
                argument.match_global_counterfactual
            ),
            "target_adjustment": (
                "target-session neural outcomes estimate that session's "
                "nuisance coefficients before scoring; target rates never "
                "enter training profiles"
            ),
            "predicted_object": (
                "one adjusted square-residual local strip rate per "
                "registered cell; not a full spatial map or global shape"
            ),
            "behavior_control_scope": (
                "additive linear speed, movement-direction harmonics, and "
                "session time; no local interaction matching"
            ),
            "training_and_test_exposures_nonoverlapping": True,
            "training_baseline": "square before training exposure",
            "test_baseline": "square after test exposure",
            "training_and_test_baselines_nonoverlapping": True,
            "matching_wall_predictor": (
                "mean adjusted edit from other shapes with the exact "
                "oriented physical seam walled"
            ),
            "control_predictor": (
                "mean adjusted edit from other shapes with the same seam open"
            ),
            "outcome": (
                "Spearman(target edit, wall template) minus "
                "Spearman(target edit, open template)"
            ),
            "occupancy_support": (
                "test-aware common-bin eligibility; no neural values used"
            ),
            "inferential_unit": "animal",
        },
        "settings": {
            "frames_per_second": argument.frames_per_second,
            "position_bin_cm": 5.0,
            "map_orientation": (
                "released image-style row = floor(raw y / 5 cm)"
            ),
            "velocity_half_window_frames": (
                argument.velocity_half_window_frames
            ),
            "speed_cap_quantile_from_initial_familiar_square": (
                argument.speed_cap_quantile
            ),
            "minimum_direction_speed_cm_s": (
                argument.minimum_direction_speed_cm_s
            ),
            "covariate_standardization": (
                "initial familiar-square mean and scale frozen for every "
                "later session"
            ),
            "behavior_ridge_fraction_of_mean_gram_diagonal": (
                argument.behavior_ridge_fraction
            ),
            "trace_cell_chunk": argument.trace_cell_chunk,
            "minimum_seconds_per_common_bin": argument.minimum_seconds,
            "minimum_common_bins": argument.minimum_bins,
            "minimum_common_cells": argument.minimum_cells,
            "sensitivity_mode_without_session_time": MODE_NO_TIME,
            "primary_mode_with_session_time": MODE_WITH_TIME,
        },
        "cohort": cohort_summary(animals),
        "animals": animals,
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["cohort"], indent=2), flush=True)


if __name__ == "__main__":
    main()
