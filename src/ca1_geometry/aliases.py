"""Tests of whether CA1 distinguishes locally identical arena partitions.

The arena is divided into nine 25 cm square partitions, numbered north to
south and west to east as in the released dataset.  Two accessible partitions
are *local boundary aliases* when the presence or absence of an immediate
wall is identical in allocentric north, east, south, and west directions.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ca1_geometry.arena import partition_accessibility


BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
BoundaryContext = tuple[bool, bool, bool, bool]
PartitionPair = tuple[int, int]


def partition_row_column(partition: int) -> tuple[int, int]:
    """Return north-to-south row and west-to-east column for a partition."""

    if not 0 <= partition <= 8:
        raise ValueError("partition must lie in [0, 8]")
    return divmod(partition, 3)


def boundary_context(
    blocked: ArrayLike,
    partition: int,
) -> BoundaryContext:
    """Return immediate wall presence in ``(north, east, south, west)``.

    The outer arena boundary counts as a wall, as does an adjacent blocked
    partition.  Asking for the context of a blocked partition is an error.
    """

    accessible = partition_accessibility(blocked)
    row, column = partition_row_column(partition)
    if not accessible[row, column]:
        raise ValueError("boundary context is defined only for accessible partitions")

    north = row == 0 or not accessible[row - 1, column]
    east = column == 2 or not accessible[row, column + 1]
    south = row == 2 or not accessible[row + 1, column]
    west = column == 0 or not accessible[row, column - 1]
    return bool(north), bool(east), bool(south), bool(west)


def exact_alias_pairs(blocked: ArrayLike) -> tuple[PartitionPair, ...]:
    """Return accessible pairs with identical immediate boundary context."""

    accessible = partition_accessibility(blocked).ravel()
    groups: dict[BoundaryContext, list[int]] = {}
    for partition in np.flatnonzero(accessible):
        context = boundary_context(blocked, int(partition))
        groups.setdefault(context, []).append(int(partition))
    pairs = [
        pair
        for partitions in groups.values()
        for pair in itertools.combinations(partitions, 2)
    ]
    return tuple(sorted(pairs))


def squared_grid_distance(pair: PartitionPair) -> int:
    """Return squared distance between partition centers in grid units."""

    first, second = pair
    first_row, first_column = partition_row_column(first)
    second_row, second_column = partition_row_column(second)
    return (first_row - second_row) ** 2 + (
        first_column - second_column
    ) ** 2


def grid_displacement_signature(pair: PartitionPair) -> tuple[int, int]:
    """Return absolute row/column displacement between partition centers."""

    first_row, first_column = partition_row_column(pair[0])
    second_row, second_column = partition_row_column(pair[1])
    return abs(first_row - second_row), abs(first_column - second_column)


def distance_matched_control_pairs(
    blocked: ArrayLike,
    target: PartitionPair,
) -> tuple[PartitionPair, ...]:
    """Return non-alias accessible pairs at the target's grid distance."""

    accessible = tuple(
        int(value)
        for value in np.flatnonzero(partition_accessibility(blocked).ravel())
    )
    aliases = set(exact_alias_pairs(blocked))
    distance = squared_grid_distance(target)
    result = []
    for pair in itertools.combinations(accessible, 2):
        if pair in aliases or squared_grid_distance(pair) != distance:
            continue
        result.append(pair)
    return tuple(result)


def displacement_matched_control_pairs(
    blocked: ArrayLike,
    target: PartitionPair,
) -> tuple[PartitionPair, ...]:
    """Return non-alias pairs with the same row/column displacement."""

    accessible = tuple(
        int(value)
        for value in np.flatnonzero(partition_accessibility(blocked).ravel())
    )
    aliases = set(exact_alias_pairs(blocked))
    displacement = grid_displacement_signature(target)
    return tuple(
        pair
        for pair in itertools.combinations(accessible, 2)
        if pair not in aliases
        and grid_displacement_signature(pair) == displacement
    )


def partition_slice(
    partition: int,
    *,
    bins_per_partition: int = 5,
) -> tuple[slice, slice]:
    """Return map-row/map-column slices using the release's paper ordering.

    Although raw physical y increases south-to-north, the released map rows
    are stored north-to-south so that rows 0:5 correspond directly to paper
    partitions 0, 1, and 2.  This was verified from occupancy in asymmetric
    geometries (not inferred from the raw-coordinate convention).
    """

    if bins_per_partition <= 0:
        raise ValueError("bins_per_partition must be positive")
    row, column = partition_row_column(partition)
    y_start = row * bins_per_partition
    x_start = column * bins_per_partition
    return (
        slice(y_start, y_start + bins_per_partition),
        slice(x_start, x_start + bins_per_partition),
    )


def extract_partition(
    values: ArrayLike,
    partition: int,
    *,
    bins_per_partition: int = 5,
) -> FloatArray:
    """Extract one partition from an array whose last axes are ``(y, x)``."""

    array = np.asarray(values, dtype=np.float64)
    expected = 3 * bins_per_partition
    if array.ndim < 2 or array.shape[-2:] != (expected, expected):
        raise ValueError(
            f"values must end in spatial shape ({expected}, {expected})"
        )
    y_slice, x_slice = partition_slice(
        partition,
        bins_per_partition=bins_per_partition,
    )
    return array[..., y_slice, x_slice]


def common_relative_support(
    occupancy_maps: list[ArrayLike],
    pair: PartitionPair,
    *,
    minimum_seconds: float,
    bins_per_partition: int = 5,
) -> BoolArray:
    """Select relative bins sampled in both partitions on every exposure."""

    if not occupancy_maps:
        raise ValueError("at least one occupancy map is required")
    if minimum_seconds < 0:
        raise ValueError("minimum_seconds must be non-negative")
    first, second = pair
    tiles = [
        extract_partition(
            occupancy,
            partition,
            bins_per_partition=bins_per_partition,
        )
        for occupancy in occupancy_maps
        for partition in (first, second)
    ]
    finite_and_sampled = [
        np.isfinite(tile) & (tile >= minimum_seconds) for tile in tiles
    ]
    return np.logical_and.reduce(finite_and_sampled)


def cellwise_partition_correlations(
    rate_maps: ArrayLike,
    pair: PartitionPair,
    support: ArrayLike,
    *,
    bins_per_partition: int = 5,
) -> FloatArray:
    """Correlate each cell's aligned spatial pattern across a partition pair."""

    rate = np.asarray(rate_maps, dtype=np.float64)
    return cellwise_cross_map_correlations(
        rate,
        rate,
        pair,
        support,
        bins_per_partition=bins_per_partition,
    )


def cellwise_cross_map_correlations(
    first_rate_maps: ArrayLike,
    second_rate_maps: ArrayLike,
    pair: PartitionPair,
    support: ArrayLike,
    *,
    bins_per_partition: int = 5,
) -> FloatArray:
    """Correlate the first tile in one map with the second tile in another."""

    first_rate = np.asarray(first_rate_maps, dtype=np.float64)
    second_rate = np.asarray(second_rate_maps, dtype=np.float64)
    if first_rate.shape != second_rate.shape:
        raise ValueError("rate-map arrays must have equal shape")
    rate = first_rate
    if rate.ndim != 3:
        raise ValueError("rate_maps must have shape (cell, y, x)")
    mask = np.asarray(support, dtype=bool)
    expected_support = (bins_per_partition, bins_per_partition)
    if mask.shape != expected_support:
        raise ValueError(f"support must have shape {expected_support}")
    if np.count_nonzero(mask) < 3:
        return np.full(rate.shape[0], np.nan, dtype=np.float64)

    first = extract_partition(
        first_rate,
        pair[0],
        bins_per_partition=bins_per_partition,
    )[:, mask]
    second = extract_partition(
        second_rate,
        pair[1],
        bins_per_partition=bins_per_partition,
    )[:, mask]
    finite = np.all(np.isfinite(first) & np.isfinite(second), axis=1)
    first_centered = first - np.nanmean(first, axis=1, keepdims=True)
    second_centered = second - np.nanmean(second, axis=1, keepdims=True)
    denominator = np.linalg.norm(first_centered, axis=1) * np.linalg.norm(
        second_centered,
        axis=1,
    )
    valid = finite & (denominator > np.finfo(np.float64).eps)
    result = np.full(rate.shape[0], np.nan, dtype=np.float64)
    result[valid] = np.sum(
        first_centered[valid] * second_centered[valid],
        axis=1,
    ) / denominator[valid]
    return result


def mean_fisher_correlation(values: ArrayLike) -> float:
    """Average finite correlations in Fisher-z space and return an r value."""

    correlation = np.asarray(values, dtype=np.float64).ravel()
    correlation = correlation[np.isfinite(correlation)]
    if correlation.size == 0:
        return float("nan")
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(correlation, -1.0 + epsilon, 1.0 - epsilon)
    return float(np.tanh(np.mean(np.arctanh(clipped))))
