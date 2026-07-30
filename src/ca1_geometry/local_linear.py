"""Boundary-aware local-linear estimates of neural response Jacobians.

The confirmatory metric is built from Jacobians fitted independently in
disjoint temporal folds.  This module deliberately estimates the derivatives
directly instead of differentiating a separately smoothed rate map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class LocalMapConfig:
    """Configuration for a local-linear response map."""

    bandwidth: float
    kernel: str = "tricube"
    min_effective_samples: float = 40.0
    min_design_eigenratio: float = 0.03
    ridge_relative: float = 1e-8

    def __post_init__(self) -> None:
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if self.kernel not in {"tricube", "gaussian"}:
            raise ValueError("kernel must be 'tricube' or 'gaussian'")
        if self.min_effective_samples <= 0:
            raise ValueError("min_effective_samples must be positive")
        if not 0 <= self.min_design_eigenratio <= 1:
            raise ValueError("min_design_eigenratio must lie in [0, 1]")
        if self.ridge_relative < 0:
            raise ValueError("ridge_relative must be non-negative")


@dataclass(frozen=True)
class LocalMap:
    """Position-conditioned response and derivative estimates."""

    mean: FloatArray
    jacobian: FloatArray
    effective_n: FloatArray
    design_eigenratio: FloatArray
    valid: BoolArray


def _kernel_weights(radius: FloatArray, kernel: str) -> FloatArray:
    if kernel == "tricube":
        weights = np.zeros_like(radius)
        inside = radius < 1.0
        weights[inside] = (1.0 - radius[inside] ** 3) ** 3
        return weights
    return np.exp(-0.5 * radius**2)


def fit_local_linear(
    position: ArrayLike,
    response: ArrayLike,
    query: ArrayLike,
    config: LocalMapConfig,
    *,
    sample_weight: ArrayLike | None = None,
    visibility: ArrayLike | None = None,
) -> LocalMap:
    """Fit a local-linear neural response model at every query location.

    Parameters
    ----------
    position
        Array of shape ``(sample, 2)`` in a common physical coordinate system.
    response
        Array of shape ``(sample, neuron)``. Response scaling must be fitted
        once and held fixed across paired conditions before calling this
        function.
    query
        Array of shape ``(query, 2)`` in the same units as ``position``.
    config
        Kernel and quality-control settings.
    sample_weight
        Optional non-negative behavioral standardization weights.
    visibility
        Optional Boolean array of shape ``(query, sample)``. False entries
        prevent smoothing across an occluded partition or wall. The same
        visibility operator must be used for both paired conditions.

    Returns
    -------
    LocalMap
        The Jacobian has shape ``(query, neuron, 2)``. Invalid queries contain
        NaNs and must not be silently interpolated.
    """

    x = np.asarray(position, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)

    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("position must have shape (sample, 2)")
    if y.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError("response must have shape (sample, neuron)")
    if q.ndim != 2 or q.shape[1] != 2:
        raise ValueError("query must have shape (query, 2)")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("position and response must be finite")

    if sample_weight is None:
        base_weight = np.ones(x.shape[0], dtype=np.float64)
    else:
        base_weight = np.asarray(sample_weight, dtype=np.float64)
        if base_weight.shape != (x.shape[0],):
            raise ValueError("sample_weight must have shape (sample,)")
        if not np.isfinite(base_weight).all() or np.any(base_weight < 0):
            raise ValueError("sample_weight must be finite and non-negative")

    if visibility is not None:
        visible = np.asarray(visibility, dtype=bool)
        if visible.shape != (q.shape[0], x.shape[0]):
            raise ValueError("visibility must have shape (query, sample)")
    else:
        visible = None

    n_query = q.shape[0]
    n_neuron = y.shape[1]
    means = np.full((n_query, n_neuron), np.nan, dtype=np.float64)
    jacobians = np.full((n_query, n_neuron, 2), np.nan, dtype=np.float64)
    effective_n = np.zeros(n_query, dtype=np.float64)
    eigenratio = np.zeros(n_query, dtype=np.float64)
    valid = np.zeros(n_query, dtype=bool)

    tree = cKDTree(x)
    radius_limit = (
        config.bandwidth
        if config.kernel == "tricube"
        else 3.0 * config.bandwidth
    )

    for qi, center in enumerate(q):
        index = np.asarray(tree.query_ball_point(center, radius_limit), dtype=int)
        if index.size < 3:
            continue
        if visible is not None:
            index = index[visible[qi, index]]
        if index.size < 3:
            continue

        offset = (x[index] - center) / config.bandwidth
        radius = np.linalg.norm(offset, axis=1)
        weight = _kernel_weights(radius, config.kernel) * base_weight[index]
        positive = weight > 0
        if np.count_nonzero(positive) < 3:
            continue

        index = index[positive]
        offset = offset[positive]
        weight = weight[positive]
        design = np.column_stack((np.ones(index.size), offset))

        total_weight = float(weight.sum())
        effective = total_weight**2 / float(np.square(weight).sum())
        effective_n[qi] = effective
        if effective < config.min_effective_samples:
            continue

        gram = design.T @ (weight[:, None] * design)
        eigval = np.linalg.eigvalsh(gram)
        ratio = float(eigval[0] / eigval[-1]) if eigval[-1] > 0 else 0.0
        eigenratio[qi] = ratio
        if ratio < config.min_design_eigenratio:
            continue

        ridge = config.ridge_relative * float(np.trace(gram)) / gram.shape[0]
        regularized = gram + ridge * np.eye(gram.shape[0])
        right_hand = design.T @ (weight[:, None] * y[index])
        coefficient = np.linalg.solve(regularized, right_hand)

        means[qi] = coefficient[0]
        jacobians[qi] = coefficient[1:].T / config.bandwidth
        valid[qi] = True

    return LocalMap(
        mean=means,
        jacobian=jacobians,
        effective_n=effective_n,
        design_eigenratio=eigenratio,
        valid=valid,
    )
