"""Cross-location transfer on the released raw-event behavior-adjusted maps.

This is a strict transport of the one-grid-step, same-signed-normal primary
from ``run_boundary_fragment_cross_location_transfer.py``.  It reuses the
raw-event map estimator from ``run_boundary_fragment_behavior_adjusted.py``
and the original cross-location source scorer.  Geometry, sampling-map
support, registration masks, source selection, and aggregation are unchanged.

The two map modes are:

* speed plus allocentric movement direction; and
* speed, allocentric movement direction, plus linear within-session time.

Only three same-normal primary quantities are retained.  No orientation-
selectivity contrast is computed or used as a rescue analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ca1_geometry.arena import spatial_accessibility  # noqa: E402
from ca1_geometry.io import Mat73Animal  # noqa: E402
from ca1_geometry.seams import (  # noqa: E402
    SeamState,
    internal_seams,
    seam_state,
    seam_strip_bins,
)
import run_boundary_fragment_behavior_adjusted as behavior  # noqa: E402
import run_boundary_fragment_cross_location_transfer as transfer  # noqa: E402


PRIMARY_METRICS = (
    "source_effect_r_to_target_residual",
    "source_effect_specificity_over_target_open_r",
    "source_wall_minus_open_r_to_target_residual",
)
PRIMARY_DISTANCE_CM = transfer.PRIMARY_DISTANCE_CM


class ProvenanceError(RuntimeError):
    """Raised when exact reuse would require changing the original sample."""


BundleRateFunction = Callable[
    [
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        tuple[tuple[int, int], ...],
    ],
    np.ndarray,
]


def _bundle_local_rate(mode: str) -> BundleRateFunction:
    """Return the original local-rate scorer for one adjusted-map mode."""

    def score(
        rate_bundle: dict[str, np.ndarray],
        occupancy: np.ndarray,
        cells: np.ndarray,
        bins: tuple[tuple[int, int], ...],
    ) -> np.ndarray:
        return transfer._raw_local_rate(
            rate_bundle[mode],
            occupancy,
            cells,
            bins,
        )

    return score


ADJUSTED_RATE_MODES: dict[str, BundleRateFunction] = {
    mode: _bundle_local_rate(mode) for mode in behavior.MODES
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
            / "boundary_fragment_cross_location_behavior_adjusted.json"
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
    )
    parser.add_argument(
        "--trace-cell-chunk",
        type=int,
        default=64,
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-bins", type=int, default=6)
    parser.add_argument("--minimum-cells", type=int, default=20)
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
        default=(
            ROOT
            / "results"
            / "diagnostics"
            / "cross_location_behavior_adjusted_checkpoints"
        ),
        help=(
            "Validated per-animal checkpoint directory. Set to an empty "
            "string only through a programmatic call to disable caching."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore matching per-animal checkpoints and recompute.",
    )
    return parser.parse_args()


def _validate_settings(argument: argparse.Namespace) -> None:
    if argument.frames_per_second <= 0:
        raise ValueError("frames-per-second must be positive")
    if argument.velocity_half_window_frames < 1:
        raise ValueError("velocity half-window must be positive")
    if not 0 < argument.speed_cap_quantile <= 1:
        raise ValueError("speed-cap quantile must lie in (0, 1]")
    if argument.minimum_direction_speed_cm_s < 0:
        raise ValueError("minimum direction speed cannot be negative")
    if argument.behavior_ridge_fraction < 0:
        raise ValueError("behavior ridge fraction cannot be negative")
    if argument.trace_cell_chunk < 1:
        raise ValueError("trace-cell chunk must be positive")
    if argument.minimum_seconds < 0:
        raise ValueError("minimum seconds cannot be negative")
    if argument.minimum_bins < 1:
        raise ValueError("minimum bins must be positive")
    if argument.minimum_cells < 3:
        raise ValueError("minimum cells must be at least three")


def _analysis_settings(argument: argparse.Namespace) -> dict[str, Any]:
    return {
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
        "primary_source_midpoint_distance_cm": PRIMARY_DISTANCE_CM,
        "primary_orientation_relation": "same_signed_normal",
        "map_modes": list(behavior.MODES),
    }


def _maximum_absolute_error(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    difference = np.abs(
        np.asarray(first, dtype=np.float64)
        - np.asarray(second, dtype=np.float64)
    )
    return float(np.max(difference)) if difference.size else 0.0


def _validate_repeated_sequence(
    environment: list[str],
    blocked: list[tuple[int, ...]],
) -> int:
    repetitions = (len(environment) - 1) // 10
    if len(environment) != 10 * repetitions + 1:
        raise ProvenanceError("session count is not 10R+1")
    if environment[0] != "square" or blocked[0]:
        raise ProvenanceError(
            "session 1 is not the unblocked familiar-square calibration"
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
        if len(repeated_environment) != 1 or len(repeated_blocked) != 1:
            raise ProvenanceError(
                "target offsets do not repeat one fixed geometry per cycle"
            )
    return repetitions


def _analysis_valid_frames(
    position: np.ndarray,
    accessibility: np.ndarray,
) -> np.ndarray:
    physical = np.all(
        (position >= 0.0) & (position < 75.0),
        axis=1,
    )
    selected = np.zeros(position.shape[0], dtype=bool)
    xy = position[physical]
    x_bin = np.floor(xy[:, 0] / 5.0).astype(np.int64)
    y_bin = np.floor(xy[:, 1] / 5.0).astype(np.int64)
    selected[physical] = accessibility[y_bin, x_bin]
    return selected


def _estimate_adjusted_map_bundle(
    path: Path,
    *,
    frames_per_second: float,
    half_window_frames: int,
    speed_cap_quantile: float,
    minimum_direction_speed_cm_s: float,
    ridge_fraction: float,
    trace_cell_chunk: int,
) -> dict[str, Any]:
    """Reproduce the existing adjusted maps while freezing transfer support."""

    rate: dict[int, dict[str, np.ndarray]] = {}
    occupancy: dict[int, np.ndarray] = {}
    registered: dict[int, np.ndarray] = {}
    session_diagnostics: list[dict[str, Any]] = []
    maximum_sampling_error = 0.0
    maximum_accessible_sampling_error = 0.0
    maximum_adjusted_mask_mismatch = {
        mode: 0 for mode in behavior.MODES
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
        repetitions = _validate_repeated_sequence(environment, blocked)
        accessibility = {
            session: spatial_accessibility(blocked[session])
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

        calibration_position = animal.position(0)
        calibration_valid = _analysis_valid_frames(
            calibration_position,
            accessibility[0],
        )
        calibration_velocity = behavior._central_velocity(
            calibration_position,
            frames_per_second=frames_per_second,
            half_window_frames=half_window_frames,
        )
        calibration_speed = np.linalg.norm(
            calibration_velocity[calibration_valid],
            axis=1,
        )
        animal_speed_cap = float(
            np.quantile(calibration_speed, speed_cap_quantile)
        )
        animal_speed_reference = float(np.median(calibration_speed))
        calibration_design = behavior._behavior_design(
            calibration_position,
            calibration_valid,
            frames_per_second=frames_per_second,
            half_window_frames=half_window_frames,
            speed_cap_cm_s=animal_speed_cap,
            minimum_direction_speed_cm_s=minimum_direction_speed_cm_s,
        )
        design_mean = np.mean(calibration_design, axis=0)
        design_scale = np.std(calibration_design, axis=0)
        design_scale[
            design_scale <= np.finfo(np.float64).eps
        ] = 1.0
        physical_reference = np.array(
            [animal_speed_reference, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        standardized_reference = (
            physical_reference - design_mean
        ) / design_scale

        for session in range(animal.n_sessions):
            position = (
                calibration_position
                if session == 0
                else animal.position(session)
            )
            analysis_valid = _analysis_valid_frames(
                position,
                accessibility[session],
            )
            raw_design = (
                calibration_design
                if session == 0
                else behavior._behavior_design(
                    position,
                    analysis_valid,
                    frames_per_second=frames_per_second,
                    half_window_frames=half_window_frames,
                    speed_cap_cm_s=animal_speed_cap,
                    minimum_direction_speed_cm_s=(
                        minimum_direction_speed_cm_s
                    ),
                )
            )
            standardized_design = (
                raw_design - design_mean
            ) / design_scale
            cells = np.flatnonzero(registered[session])
            if cells.size == 0:
                raise ProvenanceError(
                    f"session {session + 1} has no registered cells"
                )

            map_parts = {mode: [] for mode in behavior.MODES}
            session_occupancy: np.ndarray | None = None
            raw_occupancy: np.ndarray | None = None
            session_condition: dict[str, float] | None = None
            for start in range(0, cells.size, trace_cell_chunk):
                chunk_cells = cells[start : start + trace_cell_chunk]
                response = animal.trace(session, chunk_cells)
                finite_cells = np.all(np.isfinite(response), axis=0)
                if not np.all(finite_cells):
                    bad = chunk_cells[~finite_cells]
                    raise ProvenanceError(
                        "behavior-map fitting would drop cells from the "
                        f"original transfer eligibility in session "
                        f"{session + 1}: {bad.size} cells"
                    )
                (
                    chunk_maps,
                    chunk_occupancy,
                    _,
                    chunk_raw_occupancy,
                    chunk_diagnostics,
                ) = behavior._adjusted_maps(
                    position,
                    response,
                    analysis_valid,
                    standardized_design,
                    standardized_reference,
                    frames_per_second=frames_per_second,
                    ridge_fraction=ridge_fraction,
                )
                for mode in behavior.MODES:
                    map_parts[mode].append(chunk_maps[mode])
                if session_occupancy is None:
                    session_occupancy = chunk_occupancy
                    raw_occupancy = chunk_raw_occupancy
                    session_condition = {
                        mode: float(
                            chunk_diagnostics[mode]["condition_number"]
                        )
                        for mode in behavior.MODES
                    }
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
                    raise ProvenanceError(
                        "cell chunks produced different session support"
                    )

            if (
                session_occupancy is None
                or raw_occupancy is None
                or session_condition is None
            ):
                raise ProvenanceError("adjusted-map session produced no output")

            sampling_error = _maximum_absolute_error(
                occupancy[session],
                raw_occupancy,
            )
            accessible_sampling_error = _maximum_absolute_error(
                occupancy[session][accessibility[session]],
                session_occupancy[accessibility[session]],
            )
            maximum_sampling_error = max(
                maximum_sampling_error,
                sampling_error,
            )
            maximum_accessible_sampling_error = max(
                maximum_accessible_sampling_error,
                accessible_sampling_error,
            )
            if sampling_error > 1e-9 or accessible_sampling_error > 1e-9:
                raise ProvenanceError(
                    "raw-event trajectory support does not reproduce the "
                    f"released sampling map in session {session + 1}"
                )

            rate[session] = {}
            expected_finite = (
                session_occupancy > 0
            ) & accessibility[session]
            for mode in behavior.MODES:
                fitted = np.concatenate(map_parts[mode], axis=0)
                mismatch = int(
                    np.count_nonzero(
                        np.isfinite(fitted)
                        != np.broadcast_to(
                            (session_occupancy > 0)[None, :, :],
                            fitted.shape,
                        )
                    )
                )
                maximum_adjusted_mask_mismatch[mode] = max(
                    maximum_adjusted_mask_mismatch[mode],
                    mismatch,
                )
                if mismatch:
                    raise ProvenanceError(
                        "adjusted-map finite support differs from its "
                        f"raw-event occupancy in session {session + 1}"
                    )
                full = np.full(
                    (animal.n_cells, 15, 15),
                    np.nan,
                    dtype=np.float64,
                )
                full[cells] = fitted
                full[:, ~accessibility[session]] = np.nan
                full_expected = np.broadcast_to(
                    expected_finite[None, :, :],
                    (cells.size, 15, 15),
                )
                if not np.array_equal(
                    np.isfinite(full[cells]),
                    full_expected,
                ):
                    raise ProvenanceError(
                        "masked adjusted map has unexpected finite support"
                    )
                rate[session][mode] = full

            session_diagnostics.append(
                {
                    "session": session + 1,
                    "environment": environment[session],
                    "registered_cells": int(cells.size),
                    "analysis_frames": int(
                        np.count_nonzero(analysis_valid)
                    ),
                    "released_sampling_map_max_abs_error_seconds": (
                        sampling_error
                    ),
                    "accessible_sampling_map_max_abs_error_seconds": (
                        accessible_sampling_error
                    ),
                    "condition_number": session_condition,
                }
            )

    return {
        "environment": environment,
        "blocked": blocked,
        "repetitions": repetitions,
        "rate": rate,
        "occupancy": occupancy,
        "registered": registered,
        "behavior_calibration": {
            "initial_familiar_square_session": 1,
            "speed_cap_cm_s": animal_speed_cap,
            "reference_speed_cm_s": animal_speed_reference,
            "design_mean": design_mean.tolist(),
            "design_scale": design_scale.tolist(),
            "standardized_reference": standardized_reference.tolist(),
        },
        "map_validation": {
            "maximum_released_sampling_map_abs_error_seconds": (
                maximum_sampling_error
            ),
            "maximum_accessible_sampling_map_abs_error_seconds": (
                maximum_accessible_sampling_error
            ),
            "maximum_adjusted_rate_finite_mask_mismatches": (
                maximum_adjusted_mask_mismatch
            ),
            "original_registration_masks_preserved": True,
        },
        "session_diagnostics": session_diagnostics,
    }


def _is_primary_source(
    target_seam: Any,
    source_seam: Any,
) -> bool:
    return (
        source_seam.unordered != target_seam.unordered
        and transfer._orientation_relation(
            target_seam,
            source_seam,
        )
        == "same_signed_normal"
        and np.isclose(
            transfer._midpoint_distance_cm(
                target_seam,
                source_seam,
            ),
            PRIMARY_DISTANCE_CM,
        )
    )


def _summarize_primary_records(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not records:
        return None
    modes: dict[str, Any] = {}
    for mode in behavior.MODES:
        summary = transfer._record_summary(records, mode=mode)
        if summary is None:
            raise AssertionError("nonempty primary records disappeared")
        modes[mode] = {
            "source_pairs": summary["source_pairs"],
            "median_cells": summary["median_cells"],
            "median_target_bins": summary["median_target_bins"],
            "median_source_bins": summary["median_source_bins"],
            **{
                metric: summary[metric]
                for metric in PRIMARY_METRICS
            },
        }
    return {
        "eligible_primary_source_pairs": len(records),
        "modes": modes,
    }


def _animal_mode_summary(
    queries: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "eligible_primary_target_queries": len(queries),
        "eligible_primary_source_pairs": int(
            sum(
                query["eligible_primary_source_pairs"]
                for query in queries
            )
        ),
    }
    for metric in PRIMARY_METRICS:
        values = np.asarray(
            [
                query["modes"][mode][metric]
                for query in queries
            ],
            dtype=np.float64,
        )
        output[metric] = {
            "target_queries": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "positive_target_queries": int(
                np.count_nonzero(values > 0)
            ),
        }
    return output


def _score_prepared_animal(
    prepared: dict[str, Any],
    *,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> dict[str, Any]:
    seams = internal_seams()
    strips = {seam: seam_strip_bins(seam) for seam in seams}
    environment = prepared["environment"]
    blocked = prepared["blocked"]
    rate = prepared["rate"]
    occupancy = prepared["occupancy"]
    registered = prepared["registered"]
    repetitions = prepared["repetitions"]

    queries: list[dict[str, Any]] = []
    for training_exposure in range(repetitions - 1):
        training_start = training_exposure * 10
        test_start = (training_exposure + 1) * 10
        for target_offset in range(1, 10):
            target_test = test_start + target_offset
            query_cells = transfer._query_common_cells(
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
                records = []
                for source_seam in seams:
                    if not _is_primary_source(target_seam, source_seam):
                        continue
                    record = transfer._source_record(
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
                        rate_modes=ADJUSTED_RATE_MODES,
                    )
                    if record is not None:
                        records.append(record)
                summary = _summarize_primary_records(records)
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
                        "source_pairs": [
                            {
                                "source_seam": record["source_seam"],
                                "cells": record["cells"],
                                "target_bins": record["target_bins"],
                                "source_bins": record["source_bins"],
                            }
                            for record in records
                        ],
                        **summary,
                    }
                )
    if not queries:
        raise ValueError("animal has no eligible adjusted transfer queries")
    return {
        "modes": {
            mode: _animal_mode_summary(queries, mode=mode)
            for mode in behavior.MODES
        },
        "queries": queries,
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
) -> dict[str, Any]:
    prepared = _estimate_adjusted_map_bundle(
        path,
        frames_per_second=frames_per_second,
        half_window_frames=half_window_frames,
        speed_cap_quantile=speed_cap_quantile,
        minimum_direction_speed_cm_s=minimum_direction_speed_cm_s,
        ridge_fraction=ridge_fraction,
        trace_cell_chunk=trace_cell_chunk,
    )
    scored = _score_prepared_animal(
        prepared,
        minimum_seconds=minimum_seconds,
        minimum_bins=minimum_bins,
        minimum_cells=minimum_cells,
    )
    return {
        "animal": path.name.removesuffix(".complete.mat"),
        "sessions": len(prepared["environment"]),
        "repetitions": prepared["repetitions"],
        "modes": scored["modes"],
        "behavior_calibration": prepared["behavior_calibration"],
        "map_validation": prepared["map_validation"],
        "session_diagnostics": prepared["session_diagnostics"],
        "queries": scored["queries"],
    }


def _cohort_summary(
    animals: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in behavior.MODES:
        mode_output: dict[str, Any] = {}
        for metric in PRIMARY_METRICS:
            values = {
                animal["animal"]: animal["modes"][mode][metric]["mean"]
                for animal in animals
            }
            array = np.asarray(list(values.values()), dtype=np.float64)
            mode_output[metric] = {
                "animals": len(values),
                "positive_animals": int(
                    np.count_nonzero(array > 0)
                ),
                "animal_mean": float(np.mean(array)),
                "animal_median": float(np.median(array)),
                "animal_values": values,
            }
        mode_output["eligible_primary_target_queries_by_animal"] = {
            animal["animal"]: animal["modes"][mode][
                "eligible_primary_target_queries"
            ]
            for animal in animals
        }
        mode_output["eligible_primary_source_pairs_by_animal"] = {
            animal["animal"]: animal["modes"][mode][
                "eligible_primary_source_pairs"
            ]
            for animal in animals
        }
        output[mode] = mode_output
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_metadata(
    path: Path,
    argument: argparse.Namespace,
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "schema": 1,
        "source": {
            "name": path.name,
            "size_bytes": stat.st_size,
            "modified_time_ns": stat.st_mtime_ns,
        },
        "settings": _analysis_settings(argument),
        "implementation_sha256": {
            "behavior_map_estimator": _sha256(
                ROOT
                / "scripts"
                / "run_boundary_fragment_behavior_adjusted.py"
            ),
            "cross_location_scorer": _sha256(
                ROOT
                / "scripts"
                / "run_boundary_fragment_cross_location_transfer.py"
            ),
            "adjusted_transfer_adapter": _sha256(Path(__file__).resolve()),
        },
    }


def _read_cached_animal(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if payload.get("cache_metadata") != expected_metadata:
        return None
    result = payload.get("animal_result")
    return result if isinstance(result, dict) else None


def _write_cached_animal(
    cache_path: Path,
    metadata: dict[str, Any],
    result: dict[str, Any],
) -> None:
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


def _select_paths(argument: argparse.Namespace) -> list[Path]:
    paths = sorted(
        argument.data_dir.glob("QLAK-CA1-*.complete.mat")
    )
    if not paths:
        raise FileNotFoundError(
            f"no complete animal files found beneath {argument.data_dir}"
        )
    if not argument.animal:
        return paths
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
    return [available[name] for name in sorted(requested)]


def main() -> None:
    argument = parse_arguments()
    _validate_settings(argument)
    paths = _select_paths(argument)

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
            None
            if argument.force or cache_path is None
            else _read_cached_animal(cache_path, metadata)
        )
        cache_status = "reused"
        if result is None:
            cache_status = "computed"
            result = analyze_animal(
                path,
                frames_per_second=argument.frames_per_second,
                half_window_frames=(
                    argument.velocity_half_window_frames
                ),
                speed_cap_quantile=argument.speed_cap_quantile,
                minimum_direction_speed_cm_s=(
                    argument.minimum_direction_speed_cm_s
                ),
                ridge_fraction=argument.behavior_ridge_fraction,
                trace_cell_chunk=argument.trace_cell_chunk,
                minimum_seconds=argument.minimum_seconds,
                minimum_bins=argument.minimum_bins,
                minimum_cells=argument.minimum_cells,
            )
            if cache_path is not None:
                _write_cached_animal(cache_path, metadata, result)
        animals.append(result)
        print(
            result["animal"],
            {
                mode: {
                    metric: round(
                        result["modes"][mode][metric]["mean"],
                        4,
                    )
                    for metric in PRIMARY_METRICS
                }
                for mode in behavior.MODES
            },
            {"cache": cache_status},
            flush=True,
        )

    report = {
        "status": (
            "exploratory_cross_location_transfer_on_behavior_adjusted_"
            "raw_event_maps"
        ),
        "question": (
            "Does the one-grid-step same-normal wall-minus-open source "
            "profile transfer to a held-out target residual after adjusting "
            "raw-event maps for speed, allocentric movement direction, and "
            "optionally linear within-session time?"
        ),
        "design": {
            "behavior_map_estimator_reused": (
                "run_boundary_fragment_behavior_adjusted._adjusted_maps"
            ),
            "cross_location_scorer_reused": (
                "run_boundary_fragment_cross_location_transfer."
                "_source_record"
            ),
            "eligibility_preserved": (
                "released sampling maps and original registered-cell masks "
                "set common support and query cells exactly as in the "
                "stored-map transfer; any nonfinite registered trace is a "
                "provenance error rather than a dropped cell"
            ),
            "source_selection": (
                "different physical seam, exactly 25 cm midpoint distance, "
                "same signed boundary normal"
            ),
            "aggregation": (
                "equal-weight source seams within each target query, then "
                "equal-weight target queries within animal, then descriptive "
                "equal-weight animals"
            ),
            "target_geometry_training_neural_rates_excluded": True,
            "target_test_neural_rates_used_only_for_evaluation": True,
            "target_adjustment_caveat": (
                "target-session neural outcomes estimate that session's "
                "nuisance coefficients before its adjusted residual is "
                "scored; target rates never enter source training profiles"
            ),
            "behavior_reference": (
                "speed scale, design mean/scale, and physical reference are "
                "frozen from the initial familiar square for every session"
            ),
            "reported_primary_metrics": {
                "source_effect_r_to_target_residual": (
                    "Spearman r between source wall-minus-open residual and "
                    "held-out target wall residual"
                ),
                "source_effect_specificity_over_target_open_r": (
                    "the preceding r minus correlation of the same source "
                    "effect with the trained target-location open profile"
                ),
                "source_wall_minus_open_r_to_target_residual": (
                    "source-wall correlation with target residual minus "
                    "source-open correlation with target residual"
                ),
            },
            "direction_selectivity_contrast_computed": False,
            "inferential_unit": "animal",
            "population_inference_performed": False,
        },
        "settings": _analysis_settings(argument),
        "provenance": {
            "per_animal_cache_guard": (
                "input size/mtime, every analysis setting, and SHA-256 of "
                "the estimator, scorer, and adapter"
            ),
            "implementation_sha256": _cache_metadata(
                paths[0],
                argument,
            )["implementation_sha256"],
        },
        "cohort_descriptive": _cohort_summary(animals),
        "animals": animals,
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(
            _json_safe(report),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report["cohort_descriptive"], indent=2))


if __name__ == "__main__":
    main()
