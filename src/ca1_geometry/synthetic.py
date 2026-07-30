"""Synthetic population codes with known local metric structure."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def linear_code(
    position: ArrayLike, loading: ArrayLike, offset: ArrayLike | None = None
) -> FloatArray:
    """Evaluate an exactly linear population code."""

    x = np.asarray(position, dtype=np.float64)
    matrix = np.asarray(loading, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("position must have shape (sample, 2)")
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError("loading must have shape (neuron, 2)")
    result = x @ matrix.T
    if offset is not None:
        intercept = np.asarray(offset, dtype=np.float64)
        if intercept.shape != (matrix.shape[0],):
            raise ValueError("offset must have shape (neuron,)")
        result = result + intercept
    return result


def gaussian_place_code(
    position: ArrayLike,
    centers: ArrayLike,
    width: float,
    *,
    peak_probability: float = 0.30,
    baseline_probability: float = 0.01,
) -> FloatArray:
    """Return smooth place-field event probabilities."""

    x = np.asarray(position, dtype=np.float64)
    center = np.asarray(centers, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("position must have shape (sample, 2)")
    if center.ndim != 2 or center.shape[1] != 2:
        raise ValueError("centers must have shape (neuron, 2)")
    if width <= 0:
        raise ValueError("width must be positive")
    square_distance = np.sum(
        (x[:, None, :] - center[None, :, :]) ** 2, axis=-1
    )
    probability = baseline_probability + peak_probability * np.exp(
        -0.5 * square_distance / width**2
    )
    return np.clip(probability, 0.0, 1.0)


def boundary_normal_warp(
    position: ArrayLike,
    *,
    boundary_x: float,
    amplitude: float,
    decay: float,
    side: str = "right",
) -> FloatArray:
    """Warp the normal coordinate with known boundary-local expansion.

    On the analyzed side, the derivative of the warped x-coordinate is
    ``1 + amplitude * exp(-distance / decay)``.
    """

    x = np.asarray(position, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("position must have shape (sample, 2)")
    if amplitude < 0 or decay <= 0:
        raise ValueError("amplitude must be non-negative and decay positive")
    if side not in {"right", "left"}:
        raise ValueError("side must be 'right' or 'left'")
    sign = 1.0 if side == "right" else -1.0
    distance = sign * (x[:, 0] - boundary_x)
    if np.any(distance < 0):
        raise ValueError("all positions must lie on the selected boundary side")
    warped = x.copy()
    warped[:, 0] = x[:, 0] + sign * amplitude * decay * (
        1.0 - np.exp(-distance / decay)
    )
    return warped


def center_occlusion_normal_warp(
    position: ArrayLike,
    *,
    lower: float = 25.0,
    upper: float = 50.0,
    amplitude: float,
    decay: float,
) -> FloatArray:
    """Expand coordinates normal to the four sides of a square occlusion.

    Points in each cardinal strip outside ``[lower, upper] ** 2`` are displaced
    away from the nearest face.  Along a face normal, the derivative is
    ``1 + amplitude * exp(-distance / decay)``; tangent coordinates are left
    unchanged.  Corner sectors are left unchanged because they do not have a
    unique face normal.  The function is therefore intended for face-centered
    queries with kernels that exclude boundary endpoints.
    """

    x = np.asarray(position, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("position must have shape (sample, 2)")
    if not np.isfinite(x).all():
        raise ValueError("position must be finite")
    if not lower < upper:
        raise ValueError("lower must be strictly less than upper")
    if amplitude < 0 or decay <= 0:
        raise ValueError("amplitude must be non-negative and decay positive")

    warped = x.copy()
    x_coordinate = x[:, 0]
    y_coordinate = x[:, 1]
    within_x = (x_coordinate >= lower) & (x_coordinate <= upper)
    within_y = (y_coordinate >= lower) & (y_coordinate <= upper)

    side_specs = (
        (within_y & (x_coordinate >= upper), 0, upper, 1.0),
        (within_y & (x_coordinate <= lower), 0, lower, -1.0),
        (within_x & (y_coordinate >= upper), 1, upper, 1.0),
        (within_x & (y_coordinate <= lower), 1, lower, -1.0),
    )
    for selected, axis, boundary, sign in side_specs:
        distance = sign * (x[selected, axis] - boundary)
        displacement = amplitude * decay * (
            1.0 - np.exp(-distance / decay)
        )
        warped[selected, axis] += sign * displacement
    return warped


def sample_events(
    probability: ArrayLike, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Sample independent binary calcium-event observations."""

    value = np.asarray(probability, dtype=np.float64)
    if np.any((value < 0) | (value > 1)):
        raise ValueError("probability must lie in [0, 1]")
    return rng.binomial(1, value).astype(np.float64)
