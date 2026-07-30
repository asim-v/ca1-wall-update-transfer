"""Registered-cell tests of reusable boundary-specific rate-map changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class IndependentBaselineCorrelation:
    """Correlation of two edits using opposite square baselines."""

    first_assignment: float
    second_assignment: float

    @property
    def mean(self) -> float:
        return float(
            np.nanmean([self.first_assignment, self.second_assignment])
        )


def common_support_bins(
    occupancy: list[ArrayLike],
    candidate_bins: tuple[tuple[int, int], ...],
    *,
    minimum_seconds: float,
) -> tuple[tuple[int, int], ...]:
    """Keep bins meeting a dwell threshold in every supplied session."""

    if not occupancy:
        raise ValueError("at least one occupancy map is required")
    if minimum_seconds < 0:
        raise ValueError("minimum_seconds must be non-negative")
    maps = [np.asarray(value, dtype=np.float64) for value in occupancy]
    if any(value.shape != maps[0].shape for value in maps):
        raise ValueError("occupancy maps must have equal shapes")
    result = []
    for y_bin, x_bin in candidate_bins:
        value = np.asarray(
            [item[y_bin, x_bin] for item in maps],
            dtype=np.float64,
        )
        if np.isfinite(value).all() and np.min(value) >= minimum_seconds:
            result.append((y_bin, x_bin))
    return tuple(result)


def local_cell_rate(
    rate_maps: ArrayLike,
    cells: ArrayLike,
    bins: tuple[tuple[int, int], ...],
) -> FloatArray:
    """Average each cell's unsmoothed rate equally across common bins."""

    rate = np.asarray(rate_maps, dtype=np.float64)
    cell = np.asarray(cells, dtype=np.int64).ravel()
    if rate.ndim != 3:
        raise ValueError("rate_maps must have shape (cell, y, x)")
    if not bins:
        raise ValueError("at least one spatial bin is required")
    if cell.size and (
        np.any(cell < 0)
        or np.any(cell >= rate.shape[0])
        or np.unique(cell).size != cell.size
    ):
        raise ValueError("cells must be unique valid indices")
    value = np.stack(
        [rate[cell, y_bin, x_bin] for y_bin, x_bin in bins],
        axis=1,
    )
    return np.nanmean(value, axis=1)


def occupancy_weighted_cell_rate(
    rate_maps: ArrayLike,
    occupancy: ArrayLike,
    cells: ArrayLike,
) -> FloatArray:
    """Return each cell's arena-wide rate under measured spatial sampling."""

    rate = np.asarray(rate_maps, dtype=np.float64)
    dwell = np.asarray(occupancy, dtype=np.float64)
    cell = np.asarray(cells, dtype=np.int64).ravel()
    if rate.ndim != 3:
        raise ValueError("rate_maps must have shape (cell, y, x)")
    if dwell.shape != rate.shape[1:]:
        raise ValueError("occupancy must match the spatial map shape")
    if cell.size and (
        np.any(cell < 0)
        or np.any(cell >= rate.shape[0])
        or np.unique(cell).size != cell.size
    ):
        raise ValueError("cells must be unique valid indices")
    valid_dwell = np.where(np.isfinite(dwell) & (dwell > 0), dwell, 0.0)
    denominator = float(np.sum(valid_dwell))
    if denominator <= 0:
        raise ValueError("occupancy must contain positive finite dwell")
    weighted = (
        np.nan_to_num(rate[cell], nan=0.0)
        * valid_dwell[None, :, :]
    )
    return np.sum(weighted, axis=(1, 2)) / denominator


def globally_demeaned_local_cell_rate(
    rate_maps: ArrayLike,
    occupancy: ArrayLike,
    cells: ArrayLike,
    bins: tuple[tuple[int, int], ...],
) -> FloatArray:
    """Remove each cell/session's arena-wide rate from its local strip rate."""

    return local_cell_rate(rate_maps, cells, bins) - (
        occupancy_weighted_cell_rate(rate_maps, occupancy, cells)
    )


def spearman_correlation(first: ArrayLike, second: ArrayLike) -> float:
    """Pairwise-finite Spearman correlation without warning side effects."""

    x = np.asarray(first, dtype=np.float64).ravel()
    y = np.asarray(second, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("vectors must have equal shape")
    keep = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(keep) < 3:
        return float("nan")
    x_rank = rankdata(x[keep])
    y_rank = rankdata(y[keep])
    x_centered = x_rank - np.mean(x_rank)
    y_centered = y_rank - np.mean(y_rank)
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(x_centered @ y_centered / denominator)


def independent_square_residual_correlation(
    first_target: ArrayLike,
    second_target: ArrayLike,
    pre_square: ArrayLike,
    post_square: ArrayLike,
) -> IndependentBaselineCorrelation:
    """Correlate target edits without sharing square-baseline noise.

    Both assignments are retained: target A minus pre-square versus target B
    minus post-square, and the opposite assignment.  Their mean is symmetric
    in the two target sessions.
    """

    first = np.asarray(first_target, dtype=np.float64)
    second = np.asarray(second_target, dtype=np.float64)
    pre = np.asarray(pre_square, dtype=np.float64)
    post = np.asarray(post_square, dtype=np.float64)
    if not (first.shape == second.shape == pre.shape == post.shape):
        raise ValueError("all cell-rate vectors must have equal shape")
    return IndependentBaselineCorrelation(
        first_assignment=spearman_correlation(first - pre, second - post),
        second_assignment=spearman_correlation(
            first - post, second - pre
        ),
    )
