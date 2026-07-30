"""Internal grid seams and local sampling strips.

The deformation experiment repeatedly changes each physical seam between two
neighboring 25 cm partitions from traversable to walled.  This module exposes
that simple experimental structure without invoking the local tensor
estimator used elsewhere in the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class SeamState(str, Enum):
    """State of an oriented seam in one environment."""

    OPEN = "open"
    WALL = "wall"
    REVERSE_WALL = "reverse_wall"
    CLOSED = "closed"


@dataclass(frozen=True, order=True)
class OrientedSeam:
    """An internal edge directed from one partition toward its neighbor.

    When ``source`` is blocked and ``target`` is accessible, the seam is a
    wall whose normal points from ``source`` into ``target``.
    """

    source: int
    target: int

    def __post_init__(self) -> None:
        if not 0 <= self.source <= 8 or not 0 <= self.target <= 8:
            raise ValueError("partition indices must lie in [0, 8]")
        source_row, source_column = divmod(self.source, 3)
        target_row, target_column = divmod(self.target, 3)
        if (
            abs(source_row - target_row)
            + abs(source_column - target_column)
            != 1
        ):
            raise ValueError("an internal seam must join adjacent partitions")

    @property
    def unordered(self) -> tuple[int, int]:
        """Return the physical seam independent of orientation."""

        return tuple(sorted((self.source, self.target)))


def internal_seams(*, both_orientations: bool = True) -> list[OrientedSeam]:
    """Return all 12 physical seams, optionally in both orientations."""

    result: list[OrientedSeam] = []
    for partition in range(9):
        row, column = divmod(partition, 3)
        for delta_row, delta_column in ((1, 0), (0, 1)):
            adjacent_row = row + delta_row
            adjacent_column = column + delta_column
            if adjacent_row >= 3 or adjacent_column >= 3:
                continue
            adjacent = 3 * adjacent_row + adjacent_column
            result.append(OrientedSeam(partition, adjacent))
            if both_orientations:
                result.append(OrientedSeam(adjacent, partition))
    return result


def seam_state(blocked: ArrayLike, seam: OrientedSeam) -> SeamState:
    """Classify an oriented seam from the authoritative blocked partitions."""

    blocked_set = set(int(value) for value in np.asarray(blocked).ravel())
    source_blocked = seam.source in blocked_set
    target_blocked = seam.target in blocked_set
    if source_blocked and not target_blocked:
        return SeamState.WALL
    if not source_blocked and not target_blocked:
        return SeamState.OPEN
    if not source_blocked and target_blocked:
        return SeamState.REVERSE_WALL
    return SeamState.CLOSED


def _partition_center(partition: int) -> FloatArray:
    """Return a partition center in released trajectory coordinates."""

    row, column = divmod(partition, 3)
    return np.array(
        [column * 25.0 + 12.5, row * 25.0 + 12.5],
        dtype=np.float64,
    )


def seam_frame(
    seam: OrientedSeam,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return segment start, end, and source-to-target unit normal."""

    source = _partition_center(seam.source)
    target = _partition_center(seam.target)
    normal = (target - source) / 25.0
    midpoint = (source + target) / 2.0
    tangent = np.array([-normal[1], normal[0]])
    return midpoint - 12.5 * tangent, midpoint + 12.5 * tangent, normal


def seam_strip_bins(
    seam: OrientedSeam,
    *,
    depths_cm: ArrayLike = (2.5, 7.5, 12.5),
    along_cm: ArrayLike = (2.5, 7.5, 12.5, 17.5, 22.5),
    bin_size_cm: float = 5.0,
    arena_size_cm: float = 75.0,
) -> tuple[tuple[int, int], ...]:
    """Return released-map ``(row, column)`` bins on the target side.

    The default strip covers the full 25 cm seam and extends 15 cm into the
    target partition. Raw trajectory y and released-map rows both follow the
    paper's north-to-south partition ordering. Bin centers are used, so no
    sampled bin crosses the seam itself.
    """

    depth = np.asarray(depths_cm, dtype=np.float64).ravel()
    along = np.asarray(along_cm, dtype=np.float64).ravel()
    if (
        depth.size == 0
        or along.size == 0
        or not np.isfinite(depth).all()
        or not np.isfinite(along).all()
    ):
        raise ValueError("strip coordinates must be non-empty and finite")
    if np.any((depth <= 0) | (depth >= 25)):
        raise ValueError("depths must lie strictly between 0 and 25 cm")
    if np.any((along <= 0) | (along >= 25)):
        raise ValueError("along-seam coordinates must lie in (0, 25) cm")
    if bin_size_cm <= 0 or arena_size_cm <= 0:
        raise ValueError("bin and arena sizes must be positive")
    n_bins_float = arena_size_cm / bin_size_cm
    n_bins = int(round(n_bins_float))
    if not np.isclose(n_bins, n_bins_float):
        raise ValueError("arena size must be an integer multiple of bin size")

    start, end, normal = seam_frame(seam)
    tangent = (end - start) / np.linalg.norm(end - start)
    result: set[tuple[int, int]] = set()
    for tangential_distance in along:
        for normal_distance in depth:
            point = (
                start
                + tangential_distance * tangent
                + normal_distance * normal
            )
            x_bin = int(np.floor(point[0] / bin_size_cm))
            map_row = int(np.floor(point[1] / bin_size_cm))
            result.add((map_row, x_bin))
    return tuple(sorted(result))
