"""Coordinate-specific tests for fields made inaccessible by occlusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ca1_geometry.arena import partition_accessibility


@dataclass(frozen=True)
class ReflectionQuery:
    """A reflected source bin and tangential controls across one new wall."""

    source_partition: int
    accessible_partition: int
    direction: str
    distance_bins: float
    projected_bin: tuple[int, int]
    tangential_control_bins: tuple[tuple[int, int], ...]


def map_bin_partition(
    row_bin: int,
    column_bin: int,
    *,
    bins_per_partition: int = 5,
) -> int:
    """Map a released-map bin to the paper's north-to-south partition."""

    size = 3 * bins_per_partition
    if not (0 <= row_bin < size and 0 <= column_bin < size):
        raise ValueError("map bin lies outside the arena")
    return 3 * (row_bin // bins_per_partition) + (
        column_bin // bins_per_partition
    )


def nearest_accessible_reflections(
    blocked: ArrayLike,
    source_bin: tuple[int, int],
    *,
    bins_per_partition: int = 5,
) -> tuple[ReflectionQuery, ...]:
    """Reflect an inaccessible bin across its nearest accessible wall face."""

    row_bin, column_bin = source_bin
    partition = map_bin_partition(
        row_bin,
        column_bin,
        bins_per_partition=bins_per_partition,
    )
    accessible = partition_accessibility(blocked)
    partition_row, partition_column = divmod(partition, 3)
    if accessible[partition_row, partition_column]:
        raise ValueError("source bin must lie in a blocked partition")
    local_row = row_bin % bins_per_partition
    local_column = column_bin % bins_per_partition
    last = bins_per_partition - 1

    candidates: list[ReflectionQuery] = []
    directions = (
        ("north", -1, 0, local_row + 0.5),
        ("east", 0, 1, last - local_column + 0.5),
        ("south", 1, 0, last - local_row + 0.5),
        ("west", 0, -1, local_column + 0.5),
    )
    for name, delta_row, delta_column, distance in directions:
        neighbor_row = partition_row + delta_row
        neighbor_column = partition_column + delta_column
        if not (
            0 <= neighbor_row < 3
            and 0 <= neighbor_column < 3
            and accessible[neighbor_row, neighbor_column]
        ):
            continue
        neighbor = 3 * neighbor_row + neighbor_column
        if name in {"north", "south"}:
            projected_local_row = last - local_row
            projected_local_column = local_column
            projected = (
                neighbor_row * bins_per_partition + projected_local_row,
                neighbor_column * bins_per_partition
                + projected_local_column,
            )
            controls = tuple(
                (
                    projected[0],
                    neighbor_column * bins_per_partition + tangent,
                )
                for tangent in range(bins_per_partition)
                if tangent != local_column
            )
        else:
            projected_local_row = local_row
            projected_local_column = last - local_column
            projected = (
                neighbor_row * bins_per_partition + projected_local_row,
                neighbor_column * bins_per_partition
                + projected_local_column,
            )
            controls = tuple(
                (
                    neighbor_row * bins_per_partition + tangent,
                    projected[1],
                )
                for tangent in range(bins_per_partition)
                if tangent != local_row
            )
        candidates.append(
            ReflectionQuery(
                source_partition=partition,
                accessible_partition=neighbor,
                direction=name,
                distance_bins=float(distance),
                projected_bin=projected,
                tangential_control_bins=controls,
            )
        )
    if not candidates:
        return ()
    minimum = min(item.distance_bins for item in candidates)
    return tuple(
        item for item in candidates if np.isclose(item.distance_bins, minimum)
    )


def nearest_accessible_clamps(
    blocked: ArrayLike,
    source_bin: tuple[int, int],
    *,
    bins_per_partition: int = 5,
) -> tuple[ReflectionQuery, ...]:
    """Project a source bin to the first accessible bin across its nearest face."""

    reflected = nearest_accessible_reflections(
        blocked,
        source_bin,
        bins_per_partition=bins_per_partition,
    )
    last = bins_per_partition - 1
    result = []
    for query in reflected:
        neighbor_row, neighbor_column = divmod(
            query.accessible_partition,
            3,
        )
        source_row, source_column = source_bin
        local_row = source_row % bins_per_partition
        local_column = source_column % bins_per_partition
        if query.direction == "north":
            projected = (
                neighbor_row * bins_per_partition + last,
                neighbor_column * bins_per_partition + local_column,
            )
            controls = tuple(
                (projected[0], neighbor_column * bins_per_partition + tangent)
                for tangent in range(bins_per_partition)
                if tangent != local_column
            )
        elif query.direction == "south":
            projected = (
                neighbor_row * bins_per_partition,
                neighbor_column * bins_per_partition + local_column,
            )
            controls = tuple(
                (projected[0], neighbor_column * bins_per_partition + tangent)
                for tangent in range(bins_per_partition)
                if tangent != local_column
            )
        elif query.direction == "west":
            projected = (
                neighbor_row * bins_per_partition + local_row,
                neighbor_column * bins_per_partition + last,
            )
            controls = tuple(
                (neighbor_row * bins_per_partition + tangent, projected[1])
                for tangent in range(bins_per_partition)
                if tangent != local_row
            )
        else:
            projected = (
                neighbor_row * bins_per_partition + local_row,
                neighbor_column * bins_per_partition,
            )
            controls = tuple(
                (neighbor_row * bins_per_partition + tangent, projected[1])
                for tangent in range(bins_per_partition)
                if tangent != local_row
            )
        result.append(
            ReflectionQuery(
                source_partition=query.source_partition,
                accessible_partition=query.accessible_partition,
                direction=query.direction,
                distance_bins=0.5,
                projected_bin=projected,
                tangential_control_bins=controls,
            )
        )
    return tuple(result)


def reflection_contrast(
    rate_map: ArrayLike,
    queries: tuple[ReflectionQuery, ...],
) -> float:
    """Return projected rate minus same-depth tangential control rate."""

    rate = np.asarray(rate_map, dtype=np.float64)
    if rate.shape != (15, 15):
        raise ValueError("rate_map must have shape (15, 15)")
    if not queries:
        return float("nan")
    values = []
    for query in queries:
        projected = rate[query.projected_bin]
        controls = np.asarray(
            [rate[index] for index in query.tangential_control_bins],
            dtype=np.float64,
        )
        if np.isfinite(projected) and np.isfinite(controls).all():
            values.append(float(projected - np.mean(controls)))
    return float(np.mean(values)) if values else float("nan")
