"""Exploratory local-linear control for speed and heading covariates.

This module is intentionally separate from the frozen primary estimator.  It
implements the same spatial kernel, sample weights, and cross-fold metric, but
estimates the spatial slopes after weighted Frisch--Waugh--Lovell adjustment
for behavior.  The control is a post-outcome diagnostic and is not a
substitute for a design that independently varies position and behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from .arena import positions_on_accessible_support
from .local_linear import LocalMapConfig
from .metrics import cross_metric, pooled_metric
from .pilot import temporal_folds


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


KINEMATIC_COVARIATE_NAMES = (
    "speed_cm_s",
    "heading_cos_1",
    "heading_sin_1",
    "heading_cos_2",
    "heading_sin_2",
)


@dataclass(frozen=True)
class KinematicCovariates:
    """Fixed trajectory-derived nuisance regressors for one session."""

    velocity_cm_s: FloatArray
    speed_cm_s: FloatArray
    design: FloatArray
    names: tuple[str, ...] = KINEMATIC_COVARIATE_NAMES


@dataclass(frozen=True)
class BehaviorAdjustedLocalMap:
    """Local response derivatives after partialling out behavior."""

    jacobian: FloatArray
    effective_n: FloatArray
    original_design_eigenratio: FloatArray
    adjusted_spatial_eigenratio: FloatArray
    nuisance_rank: NDArray[np.int64]
    valid: BoolArray


@dataclass(frozen=True)
class BehaviorAdjustedSessionMetric:
    """Cross-validated metric and diagnostics for one adjusted session."""

    metric: FloatArray
    pooled: FloatArray
    jacobians: FloatArray
    pooled_jacobian: FloatArray
    valid: BoolArray
    effective_n: FloatArray
    original_design_eigenratio: FloatArray
    adjusted_spatial_eigenratio: FloatArray
    nuisance_rank: NDArray[np.int64]
    frames_per_fold: NDArray[np.int64]
    split_pair_reliability: float


def kinematic_covariates(
    position: ArrayLike,
    *,
    frame_rate_hz: float = 30.0,
    half_window_frames: int = 5,
) -> KinematicCovariates:
    """Calculate fixed speed and heading-harmonic regressors.

    Velocity at frame ``t`` is the secant from ``t - 5`` to ``t + 5`` at the
    released 30 Hz sampling rate (a 1/3-second interval). At the first and last
    five frames, the available asymmetric interval is used. Heading harmonics
    are obtained from the unit velocity; all four harmonics are set to zero
    only when speed is exactly zero. Each nuisance column is then centered and
    scaled over the complete session for numerical conditioning. Centering and
    scaling cannot change unpenalized joint-regression spatial slopes because
    an intercept is included locally.
    """

    xy = np.asarray(position, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("position must have shape (frame, 2)")
    if xy.shape[0] < 2:
        raise ValueError("at least two position frames are required")
    if not np.isfinite(xy).all():
        raise ValueError("position must be finite")
    if frame_rate_hz <= 0:
        raise ValueError("frame_rate_hz must be positive")
    if half_window_frames < 1:
        raise ValueError("half_window_frames must be positive")

    frame = np.arange(xy.shape[0], dtype=np.int64)
    low = np.maximum(0, frame - half_window_frames)
    high = np.minimum(xy.shape[0] - 1, frame + half_window_frames)
    elapsed = (high - low) / frame_rate_hz
    velocity = (xy[high] - xy[low]) / elapsed[:, None]
    speed = np.linalg.norm(velocity, axis=1)

    unit = np.zeros_like(velocity)
    moving = speed > 0
    unit[moving] = velocity[moving] / speed[moving, None]
    cosine = unit[:, 0]
    sine = unit[:, 1]
    raw = np.column_stack(
        (
            speed,
            cosine,
            sine,
            cosine**2 - sine**2,
            2.0 * cosine * sine,
        )
    )
    center = raw.mean(axis=0)
    scale = raw.std(axis=0)
    constant_tolerance = np.sqrt(np.finfo(np.float64).eps) * np.maximum(
        1.0, np.abs(center)
    )
    nonconstant = scale > constant_tolerance
    design = np.zeros_like(raw)
    design[:, nonconstant] = (
        raw[:, nonconstant] - center[nonconstant]
    ) / scale[nonconstant]
    return KinematicCovariates(
        velocity_cm_s=velocity,
        speed_cm_s=speed,
        design=design,
    )


def _kernel_weights(radius: FloatArray, kernel: str) -> FloatArray:
    if kernel == "tricube":
        weight = np.zeros_like(radius)
        inside = radius < 1.0
        weight[inside] = (1.0 - radius[inside] ** 3) ** 3
        return weight
    return np.exp(-0.5 * radius**2)


def _eigenratio(gram: FloatArray) -> float:
    eigenvalue = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    if eigenvalue[-1] <= 0:
        return 0.0
    return float(max(0.0, eigenvalue[0]) / eigenvalue[-1])


def fit_local_linear_with_nuisance(
    position: ArrayLike,
    response: ArrayLike,
    nuisance: ArrayLike,
    query: ArrayLike,
    config: LocalMapConfig,
    *,
    sample_weight: ArrayLike | None = None,
) -> BehaviorAdjustedLocalMap:
    """Estimate local spatial slopes jointly with arbitrary nuisance columns.

    At every query, the intercept and nuisance columns are partialled out of
    both the normalized spatial offsets and every neural response using
    weighted Frisch--Waugh--Lovell cross-products. The returned spatial
    coefficient is therefore the exact unpenalized joint-regression
    coefficient, apart from the same tiny relative ridge used for the two
    residual spatial columns.
    """

    xy = np.asarray(position, dtype=np.float64)
    event = np.asarray(response, dtype=np.float64)
    behavior = np.asarray(nuisance, dtype=np.float64)
    queries = np.asarray(query, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("position must have shape (sample, 2)")
    if event.ndim != 2 or event.shape[0] != xy.shape[0]:
        raise ValueError("response must have shape (sample, neuron)")
    if behavior.ndim != 2 or behavior.shape[0] != xy.shape[0]:
        raise ValueError("nuisance must have shape (sample, covariate)")
    if queries.ndim != 2 or queries.shape[1] != 2:
        raise ValueError("query must have shape (query, 2)")
    if not (
        np.isfinite(xy).all()
        and np.isfinite(event).all()
        and np.isfinite(behavior).all()
    ):
        raise ValueError("position, response, and nuisance must be finite")

    if sample_weight is None:
        base_weight = np.ones(xy.shape[0], dtype=np.float64)
    else:
        base_weight = np.asarray(sample_weight, dtype=np.float64)
        if base_weight.shape != (xy.shape[0],):
            raise ValueError("sample_weight must have shape (sample,)")
        if not np.isfinite(base_weight).all() or np.any(base_weight < 0):
            raise ValueError(
                "sample_weight must be finite and non-negative"
            )

    n_query = queries.shape[0]
    jacobian = np.full(
        (n_query, event.shape[1], 2), np.nan, dtype=np.float64
    )
    effective_n = np.zeros(n_query, dtype=np.float64)
    original_ratio = np.zeros(n_query, dtype=np.float64)
    adjusted_ratio = np.zeros(n_query, dtype=np.float64)
    nuisance_rank = np.zeros(n_query, dtype=np.int64)
    valid = np.zeros(n_query, dtype=bool)
    tree = cKDTree(xy)
    radius_limit = (
        config.bandwidth
        if config.kernel == "tricube"
        else 3.0 * config.bandwidth
    )

    for query_index, center in enumerate(queries):
        index = np.asarray(
            tree.query_ball_point(center, radius_limit), dtype=np.int64
        )
        if index.size < 3:
            continue
        offset = (xy[index] - center) / config.bandwidth
        radius = np.linalg.norm(offset, axis=1)
        weight = (
            _kernel_weights(radius, config.kernel) * base_weight[index]
        )
        positive = weight > 0
        if np.count_nonzero(positive) < 3:
            continue
        index = index[positive]
        offset = offset[positive]
        weight = weight[positive]

        total_weight = float(weight.sum())
        effective = total_weight**2 / float(np.square(weight).sum())
        effective_n[query_index] = effective
        if effective < config.min_effective_samples:
            continue

        spatial_design = np.column_stack(
            (np.ones(index.size, dtype=np.float64), offset)
        )
        original_gram = spatial_design.T @ (
            weight[:, None] * spatial_design
        )
        original_ratio[query_index] = _eigenratio(original_gram)
        if original_ratio[query_index] < config.min_design_eigenratio:
            continue

        nuisance_design = np.column_stack(
            (
                np.ones(index.size, dtype=np.float64),
                behavior[index],
            )
        )
        weighted_nuisance = weight[:, None] * nuisance_design
        nuisance_gram = nuisance_design.T @ weighted_nuisance
        nuisance_inverse = np.linalg.pinv(
            nuisance_gram, rcond=1e-10, hermitian=True
        )
        nuisance_rank[query_index] = int(
            np.linalg.matrix_rank(nuisance_gram, hermitian=True)
        )

        weighted_offset = weight[:, None] * offset
        weighted_event = weight[:, None] * event[index]
        xwz = offset.T @ weighted_nuisance
        zwx = xwz.T
        zwy = nuisance_design.T @ weighted_event
        adjusted_xx = (
            offset.T @ weighted_offset
            - xwz @ nuisance_inverse @ zwx
        )
        adjusted_xy = (
            offset.T @ weighted_event
            - xwz @ nuisance_inverse @ zwy
        )
        adjusted_xx = 0.5 * (adjusted_xx + adjusted_xx.T)
        adjusted_ratio[query_index] = _eigenratio(adjusted_xx)
        if adjusted_ratio[query_index] < config.min_design_eigenratio:
            continue

        ridge = (
            config.ridge_relative
            * float(np.trace(adjusted_xx))
            / adjusted_xx.shape[0]
        )
        coefficient = np.linalg.solve(
            adjusted_xx + ridge * np.eye(2), adjusted_xy
        )
        jacobian[query_index] = coefficient.T / config.bandwidth
        valid[query_index] = True

    return BehaviorAdjustedLocalMap(
        jacobian=jacobian,
        effective_n=effective_n,
        original_design_eigenratio=original_ratio,
        adjusted_spatial_eigenratio=adjusted_ratio,
        nuisance_rank=nuisance_rank,
        valid=valid,
    )


def _metric_correlation(first: FloatArray, second: FloatArray) -> float:
    common = np.isfinite(first) & np.isfinite(second)
    x = first[common]
    y = second[common]
    if x.size < 3:
        return float("nan")
    x_scale = max(1.0, float(np.max(np.abs(x))))
    y_scale = max(1.0, float(np.max(np.abs(y))))
    if (
        np.ptp(x) <= np.finfo(float).eps * x_scale
        or np.ptp(y) <= np.finfo(float).eps * y_scale
    ):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def estimate_behavior_adjusted_session_metric(
    position: ArrayLike,
    response: ArrayLike,
    nuisance: ArrayLike,
    query: ArrayLike,
    *,
    common_blocked: ArrayLike,
    config: LocalMapConfig,
    n_fold: int = 4,
    sample_weight: ArrayLike | None = None,
    fold_assignment: ArrayLike | None = None,
) -> BehaviorAdjustedSessionMetric:
    """Apply the behavior-adjusted fit with the frozen support and folds."""

    xy = np.asarray(position, dtype=np.float64)
    event = np.asarray(response, dtype=np.float64)
    behavior = np.asarray(nuisance, dtype=np.float64)
    queries = np.asarray(query, dtype=np.float64)
    if event.ndim != 2 or event.shape[0] != xy.shape[0]:
        raise ValueError("response must have shape (frame, neuron)")
    if behavior.ndim != 2 or behavior.shape[0] != xy.shape[0]:
        raise ValueError("nuisance must have shape (frame, covariate)")
    keep = positions_on_accessible_support(xy, common_blocked)
    keep &= (
        np.isfinite(xy).all(axis=1)
        & np.isfinite(event).all(axis=1)
        & np.isfinite(behavior).all(axis=1)
    )
    if fold_assignment is None:
        fold = temporal_folds(xy.shape[0], n_fold)
    else:
        fold = np.asarray(fold_assignment, dtype=np.int64)
        if fold.shape != (xy.shape[0],):
            raise ValueError("fold_assignment must have shape (frame,)")
        allowed = (fold == -1) | ((fold >= 0) & (fold < n_fold))
        if not np.all(allowed):
            raise ValueError(
                "fold assignments must be -1 or valid fold IDs"
            )
    if sample_weight is None:
        weight = np.ones(xy.shape[0], dtype=np.float64)
    else:
        weight = np.asarray(sample_weight, dtype=np.float64)
        if weight.shape != (xy.shape[0],):
            raise ValueError("sample_weight must have shape (frame,)")

    local: list[BehaviorAdjustedLocalMap] = []
    frames = np.zeros(n_fold, dtype=np.int64)
    for fold_index in range(n_fold):
        selected = keep & (fold == fold_index) & (weight > 0)
        frames[fold_index] = int(selected.sum())
        local.append(
            fit_local_linear_with_nuisance(
                xy[selected],
                event[selected],
                behavior[selected],
                queries,
                config,
                sample_weight=weight[selected],
            )
        )

    jacobian = np.stack([item.jacobian for item in local])
    valid = np.logical_and.reduce([item.valid for item in local])
    metric = cross_metric(jacobian)
    pooled_local = fit_local_linear_with_nuisance(
        xy[keep],
        event[keep],
        behavior[keep],
        queries,
        config,
        sample_weight=weight[keep],
    )
    pooled = pooled_metric(pooled_local.jacobian)
    if n_fold >= 4:
        split = n_fold // 2
        reliability = _metric_correlation(
            cross_metric(jacobian[:split])[valid],
            cross_metric(jacobian[split:])[valid],
        )
    else:
        reliability = float("nan")
    return BehaviorAdjustedSessionMetric(
        metric=metric,
        pooled=pooled,
        jacobians=jacobian,
        pooled_jacobian=pooled_local.jacobian,
        valid=valid,
        effective_n=np.stack([item.effective_n for item in local]),
        original_design_eigenratio=np.stack(
            [item.original_design_eigenratio for item in local]
        ),
        adjusted_spatial_eigenratio=np.stack(
            [item.adjusted_spatial_eigenratio for item in local]
        ),
        nuisance_rank=np.stack([item.nuisance_rank for item in local]),
        frames_per_fold=frames,
        split_pair_reliability=reliability,
    )
