"""Rate-map utilities for reproduction and fixed-coordinate analyses.

The current public analysis code constructs bins from each session's observed
coordinate maxima. :func:`authors_current_rate_maps` reproduces that helper.
The arrays actually released on Zenodo use a different convention, captured
by :func:`released_rate_maps`. :func:`fixed_rate_maps` is the analysis
implementation: all conditions share fixed physical bin edges and rates are
expressed in Hz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RateMaps:
    """Event-rate and occupancy maps in ``(cell, y, x)`` orientation."""

    rate: FloatArray
    occupancy: FloatArray
    average_event_rate: FloatArray


def _validate_inputs(
    position: ArrayLike, response: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    xy = np.asarray(position, dtype=np.float64)
    event = np.asarray(response, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("position must have shape (frame, 2)")
    if event.ndim != 2 or event.shape[0] != xy.shape[0]:
        raise ValueError("response must have shape (frame, cell)")
    if not np.isfinite(xy).all() or not np.isfinite(event).all():
        raise ValueError("position and response must be finite")
    return xy, event


def _aggregate(
    flat_bin: NDArray[np.int64],
    event: FloatArray,
    n_bins: int,
) -> tuple[FloatArray, FloatArray]:
    count = np.bincount(flat_bin, minlength=n_bins**2).reshape(
        n_bins, n_bins
    )
    event_count = np.empty(
        (event.shape[1], n_bins, n_bins), dtype=np.float64
    )
    for cell in range(event.shape[1]):
        event_count[cell] = np.bincount(
            flat_bin,
            weights=event[:, cell],
            minlength=n_bins**2,
        ).reshape(n_bins, n_bins)
    return event_count, count.astype(np.float64)


def authors_current_rate_maps(
    position: ArrayLike,
    response: ArrayLike,
    *,
    n_bins: int = 15,
    frames_per_second: float = 30.0,
    buffer: float = 1e-5,
    smoothing_sigma_bins: float = 0.0,
) -> RateMaps:
    """Reproduce the current public ``get_rate_maps`` helper.

    Each axis is divided by its session-specific observed maximum plus
    ``buffer``.  The released code accumulates arrays in ``(x, y)`` order;
    this function transposes the result into the package-wide ``(y, x)``
    convention used by the HDF5 files.
    """

    xy, event = _validate_inputs(position, response)
    if n_bins <= 0 or frames_per_second <= 0 or buffer <= 0:
        raise ValueError("bin count, frame rate, and buffer must be positive")
    if smoothing_sigma_bins < 0:
        raise ValueError("smoothing sigma must be non-negative")

    scale = (np.max(xy, axis=0) + buffer) / n_bins
    if np.any(scale <= 0):
        raise ValueError("each position dimension must have positive extent")
    binned = np.floor(xy / scale).astype(np.int64)
    if np.any((binned < 0) | (binned >= n_bins)):
        raise ValueError("legacy binning produced an out-of-range index")

    # Flat indices currently encode (x, y), matching the released loop.
    flat = binned[:, 0] * n_bins + binned[:, 1]
    event_count, count = _aggregate(flat, event, n_bins)
    visited = count > 0

    if smoothing_sigma_bins:
        event_count = gaussian_filter1d(
            gaussian_filter1d(
                event_count,
                sigma=smoothing_sigma_bins,
                axis=1,
            ),
            sigma=smoothing_sigma_bins,
            axis=2,
        )
        count = gaussian_filter1d(
            gaussian_filter1d(
                count,
                sigma=smoothing_sigma_bins,
                axis=0,
            ),
            sigma=smoothing_sigma_bins,
            axis=1,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        rate_xy = event_count / count[None, :, :] * frames_per_second
    rate_xy[:, ~visited] = np.nan
    occupancy_xy = count / xy.shape[0]
    average = event.mean(axis=0) * frames_per_second
    return RateMaps(
        rate=np.transpose(rate_xy, (0, 2, 1)),
        occupancy=occupancy_xy.T,
        average_event_rate=average,
    )


def released_rate_maps(
    position: ArrayLike,
    response: ArrayLike,
    *,
    arena_size_cm: float = 75.0,
    bin_size_cm: float = 5.0,
    frames_per_second: float = 30.0,
    smoothing_sigma_bins: float = 0.0,
) -> RateMaps:
    """Reproduce the empirically identified Zenodo map convention.

    The stored unsmoothed values are event probability per frame, not Hz, and
    ``occupancy`` is dwell time in seconds. The generator first accumulated a
    temporary 16 by 16 grid (so coordinates exactly at 75 cm entered bin 15),
    smoothed that full grid, and only then cropped to 15 by 15. The smoothing
    is separable Gaussian with constant-zero padding and radius two bins.
    """

    xy, event = _validate_inputs(position, response)
    if arena_size_cm <= 0 or bin_size_cm <= 0 or frames_per_second <= 0:
        raise ValueError("arena, bin, and frame-rate values must be positive")
    if smoothing_sigma_bins < 0:
        raise ValueError("smoothing sigma must be non-negative")
    n_bins_float = arena_size_cm / bin_size_cm
    n_bins = int(round(n_bins_float))
    if not np.isclose(n_bins, n_bins_float):
        raise ValueError("arena size must be an integer multiple of bin size")

    inside = np.all((xy >= 0) & (xy <= arena_size_cm), axis=1)
    binned = np.floor(xy[inside] / bin_size_cm).astype(np.int64)
    temporary_bins = n_bins + 1
    flat = binned[:, 1] * temporary_bins + binned[:, 0]
    event_count, count = _aggregate(flat, event[inside], temporary_bins)
    visited = count > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = event_count / count[None, :, :]
    rate[:, ~visited] = 0.0

    if smoothing_sigma_bins:
        rate = gaussian_filter1d(
            gaussian_filter1d(
                np.nan_to_num(rate),
                sigma=smoothing_sigma_bins,
                axis=1,
                mode="constant",
                cval=0.0,
                truncate=2.0,
            ),
            sigma=smoothing_sigma_bins,
            axis=2,
            mode="constant",
            cval=0.0,
            truncate=2.0,
        )
    rate = rate[:, :n_bins, :n_bins]
    count = count[:n_bins, :n_bins]
    visited = visited[:n_bins, :n_bins]
    rate[:, ~visited] = np.nan

    return RateMaps(
        rate=rate,
        occupancy=count / frames_per_second,
        average_event_rate=event[inside].mean(axis=0) * frames_per_second,
    )


def fixed_rate_maps(
    position: ArrayLike,
    response: ArrayLike,
    *,
    arena_size_cm: float = 75.0,
    bin_size_cm: float = 5.0,
    frames_per_second: float = 30.0,
    smoothing_sigma_bins: float = 0.0,
) -> RateMaps:
    """Build maps on condition-invariant physical bin edges."""

    xy, event = _validate_inputs(position, response)
    if arena_size_cm <= 0 or bin_size_cm <= 0 or frames_per_second <= 0:
        raise ValueError("arena, bin, and frame-rate values must be positive")
    if smoothing_sigma_bins < 0:
        raise ValueError("smoothing sigma must be non-negative")
    n_bins_float = arena_size_cm / bin_size_cm
    n_bins = int(round(n_bins_float))
    if not np.isclose(n_bins, n_bins_float):
        raise ValueError("arena size must be an integer multiple of bin size")
    if np.any((xy < 0) | (xy > arena_size_cm)):
        raise ValueError("positions must lie inside the physical arena")

    # The rightmost edge belongs to the last bin, as in numpy.histogram.
    binned = np.floor(xy / bin_size_cm).astype(np.int64)
    binned = np.minimum(binned, n_bins - 1)
    flat = binned[:, 1] * n_bins + binned[:, 0]
    event_count, count = _aggregate(flat, event, n_bins)
    visited = count > 0
    if smoothing_sigma_bins:
        event_count = gaussian_filter1d(
            gaussian_filter1d(
                event_count,
                sigma=smoothing_sigma_bins,
                axis=1,
            ),
            sigma=smoothing_sigma_bins,
            axis=2,
        )
        count = gaussian_filter1d(
            gaussian_filter1d(
                count,
                sigma=smoothing_sigma_bins,
                axis=0,
            ),
            sigma=smoothing_sigma_bins,
            axis=1,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = event_count / count[None, :, :] * frames_per_second
    rate[:, ~visited] = np.nan
    return RateMaps(
        rate=rate,
        occupancy=count / xy.shape[0],
        average_event_rate=event.mean(axis=0) * frames_per_second,
    )
