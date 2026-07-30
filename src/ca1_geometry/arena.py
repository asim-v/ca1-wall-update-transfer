"""Exact 3 × 3 arena masks and newly introduced boundary segments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BoundarySegment:
    """A shared edge between blocked and accessible partitions."""

    start: tuple[float, float]
    end: tuple[float, float]
    normal: tuple[float, float]
    blocked_partition: int
    accessible_partition: int

    @property
    def tangent(self) -> tuple[float, float]:
        return (-self.normal[1], self.normal[0])


@dataclass(frozen=True)
class BoundaryQueries:
    """Query points and local frames for a collection of wall segments."""

    position: FloatArray
    normal: FloatArray
    tangent: FloatArray
    distance: FloatArray
    segment_index: NDArray[np.int64]


def partition_accessibility(blocked: ArrayLike) -> BoolArray:
    """Return a north-to-south, west-to-east 3 × 3 accessibility mask."""

    blocked_index = np.asarray(blocked, dtype=int).ravel()
    if blocked_index.size and (
        np.any(blocked_index < 0) or np.any(blocked_index > 8)
    ):
        raise ValueError("blocked partition indices must lie in [0, 8]")
    accessible = np.ones((3, 3), dtype=bool)
    accessible.ravel()[blocked_index] = False
    return accessible


def spatial_accessibility(
    blocked: ArrayLike, *, bins_per_partition: int = 5
) -> BoolArray:
    """Expand the partition mask to the fixed 5 cm analysis grid."""

    if bins_per_partition <= 0:
        raise ValueError("bins_per_partition must be positive")
    return np.repeat(
        np.repeat(
            partition_accessibility(blocked),
            bins_per_partition,
            axis=0,
        ),
        bins_per_partition,
        axis=1,
    )


def introduced_boundaries(
    blocked: ArrayLike,
    *,
    arena_size: float = 75.0,
) -> list[BoundarySegment]:
    """Construct internal wall segments from blocked-accessible adjacencies.

    Released trajectory coordinates use x west-to-east and image-style y
    north-to-south, matching the paper's partition-row order. Normals point
    from the blocked partition into the accessible partition.
    """

    accessible = partition_accessibility(blocked)
    cell = arena_size / 3.0
    result: list[BoundarySegment] = []
    neighbor = (
        (-1, 0, (0.0, -1.0)),
        (1, 0, (0.0, 1.0)),
        (0, -1, (-1.0, 0.0)),
        (0, 1, (1.0, 0.0)),
    )

    for row in range(3):
        for column in range(3):
            if accessible[row, column]:
                continue
            blocked_partition = 3 * row + column
            for delta_row, delta_column, normal in neighbor:
                adjacent_row = row + delta_row
                adjacent_column = column + delta_column
                if not (
                    0 <= adjacent_row < 3 and 0 <= adjacent_column < 3
                ):
                    continue
                if not accessible[adjacent_row, adjacent_column]:
                    continue
                accessible_partition = 3 * adjacent_row + adjacent_column

                x0 = column * cell
                x1 = (column + 1) * cell
                y_top = row * cell
                y_bottom = (row + 1) * cell
                if delta_row == -1:
                    start, end = (x0, y_top), (x1, y_top)
                elif delta_row == 1:
                    start, end = (x0, y_bottom), (x1, y_bottom)
                elif delta_column == -1:
                    start, end = (x0, y_bottom), (x0, y_top)
                else:
                    start, end = (x1, y_bottom), (x1, y_top)
                result.append(
                    BoundarySegment(
                        start=start,
                        end=end,
                        normal=normal,
                        blocked_partition=blocked_partition,
                        accessible_partition=accessible_partition,
                    )
                )
    return result


def midpoint_boundary_queries(
    segments: list[BoundarySegment],
    distances: ArrayLike,
) -> BoundaryQueries:
    """Place queries normal to the midpoint of each boundary segment.

    Midpoints avoid segment endpoints, where the definition of a unique local
    boundary normal breaks down. Distances are measured into accessible space.
    """

    return segment_boundary_queries(
        segments,
        distances,
        tangential_fractions=np.array([0.5]),
    )


def segment_boundary_queries(
    segments: list[BoundarySegment],
    distances: ArrayLike,
    *,
    tangential_fractions: ArrayLike,
) -> BoundaryQueries:
    """Place queries at fixed fractions along each boundary segment."""

    distance = np.asarray(distances, dtype=np.float64).ravel()
    if not segments:
        raise ValueError("at least one boundary segment is required")
    if distance.size == 0 or not np.isfinite(distance).all():
        raise ValueError("distances must be a non-empty finite vector")
    if np.any(distance <= 0):
        raise ValueError("distances must be strictly positive")

    fraction = np.asarray(tangential_fractions, dtype=np.float64).ravel()
    if fraction.size == 0 or not np.isfinite(fraction).all():
        raise ValueError("tangential_fractions must be non-empty and finite")
    if np.any((fraction <= 0) | (fraction >= 1)):
        raise ValueError("tangential fractions must lie strictly in (0, 1)")

    position: list[FloatArray] = []
    normal: list[FloatArray] = []
    tangent: list[FloatArray] = []
    distance_output: list[float] = []
    segment_index: list[int] = []
    for index, segment in enumerate(segments):
        start = np.asarray(segment.start, dtype=np.float64)
        end = np.asarray(segment.end, dtype=np.float64)
        normal_vector = np.asarray(segment.normal, dtype=np.float64)
        tangent_vector = np.asarray(segment.tangent, dtype=np.float64)
        for along in fraction:
            anchor = start + along * (end - start)
            for value in distance:
                position.append(anchor + value * normal_vector)
                normal.append(normal_vector)
                tangent.append(tangent_vector)
                distance_output.append(float(value))
                segment_index.append(index)

    return BoundaryQueries(
        position=np.asarray(position),
        normal=np.asarray(normal),
        tangent=np.asarray(tangent),
        distance=np.asarray(distance_output),
        segment_index=np.asarray(segment_index, dtype=np.int64),
    )


def positions_on_accessible_support(
    position: ArrayLike,
    blocked: ArrayLike,
    *,
    arena_size: float = 75.0,
) -> BoolArray:
    """Select samples lying in partitions accessible in both conditions."""

    value = np.asarray(position, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError("position must have shape (sample, 2)")
    inside = np.all((value >= 0) & (value < arena_size), axis=1)
    index = np.floor(value * (3.0 / arena_size)).astype(int)
    keep = np.zeros(value.shape[0], dtype=bool)
    accessibility = partition_accessibility(blocked)
    column = index[inside, 0]
    # Raw position y is an image coordinate and follows the paper's
    # north-to-south partition-row order directly.
    row = index[inside, 1]
    keep[inside] = accessibility[row, column]
    return keep
