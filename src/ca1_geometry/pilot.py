"""Reusable components for the one-animal biological pilot."""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .arena import positions_on_accessible_support
from .local_linear import LocalMapConfig, fit_local_linear
from .metrics import (
    anisotropy_components,
    cross_metric,
    pooled_metric,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SessionMetric:
    """Cross-validated local metric and estimator diagnostics."""

    metric: FloatArray
    pooled: FloatArray
    jacobians: FloatArray
    pooled_jacobian: FloatArray
    valid: BoolArray
    effective_n: FloatArray
    design_eigenratio: FloatArray
    frames_per_fold: NDArray[np.int64]
    split_pair_reliability: float


@dataclass(frozen=True)
class DirectionalReliability:
    """Repeatability of target tensor components across independent pairs."""

    normal: float
    tangent: float
    contrast: float
    n_query: int


def temporal_folds(n_frame: int, n_fold: int = 4) -> NDArray[np.int64]:
    """Assign equal-duration, contiguous folds without random frame mixing."""

    if n_frame < n_fold or n_fold < 2:
        raise ValueError("need at least one frame per fold and two folds")
    index = np.arange(n_frame, dtype=np.int64)
    return np.minimum(n_fold - 1, index * n_fold // n_frame)


def balanced_block_folds(
    position: ArrayLike,
    keep: ArrayLike,
    *,
    n_fold: int = 4,
    block_frames: int = 1_800,
    guard_frames: int = 60,
    arena_size_cm: float = 75.0,
    bin_size_cm: float = 5.0,
) -> NDArray[np.int64]:
    """Distribute guarded temporal blocks using position-only balance.

    Consecutive groups of ``n_fold`` blocks contribute one block to every
    fold. Within each group, all fold permutations are evaluated and the
    assignment minimizing the accumulated spatial-occupancy imbalance is
    selected deterministically. Frames in a guard interval at both ends of
    every block receive label ``-1``.
    """

    xy = np.asarray(position, dtype=np.float64)
    selected = np.asarray(keep, dtype=bool)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("position must have shape (frame, 2)")
    if selected.shape != (xy.shape[0],):
        raise ValueError("keep must have shape (frame,)")
    if n_fold < 2 or block_frames <= 2 * guard_frames:
        raise ValueError("fold and guarded block sizes are incompatible")
    if guard_frames < 0:
        raise ValueError("guard_frames must be non-negative")
    n_bins_float = arena_size_cm / bin_size_cm
    n_bins = int(round(n_bins_float))
    if not np.isclose(n_bins, n_bins_float):
        raise ValueError("arena size must be divisible by bin size")

    clipped = np.minimum(
        np.floor(xy / bin_size_cm).astype(np.int64), n_bins - 1
    )
    flat_bin = clipped[:, 1] * n_bins + clipped[:, 0]
    n_block = int(np.ceil(xy.shape[0] / block_frames))
    block_occupancy: list[FloatArray] = []
    block_interior: list[NDArray[np.int64]] = []
    for block in range(n_block):
        start = block * block_frames
        end = min(xy.shape[0], (block + 1) * block_frames)
        interior_start = min(end, start + guard_frames)
        interior_end = max(interior_start, end - guard_frames)
        index = np.arange(interior_start, interior_end, dtype=np.int64)
        index = index[selected[index]]
        block_interior.append(index)
        block_occupancy.append(
            np.bincount(flat_bin[index], minlength=n_bins**2).astype(
                np.float64
            )
        )

    assignment = np.full(n_block, -1, dtype=np.int64)
    accumulated = np.zeros((n_fold, n_bins**2), dtype=np.float64)
    block_count = np.zeros(n_fold, dtype=np.int64)
    for group_start in range(0, n_block, n_fold):
        group = np.arange(
            group_start, min(n_block, group_start + n_fold)
        )
        best_score = np.inf
        best_folds: tuple[int, ...] | None = None
        for folds in itertools.permutations(range(n_fold), group.size):
            candidate = accumulated.copy()
            candidate_count = block_count.copy()
            for block, fold in zip(group, folds, strict=True):
                candidate[fold] += block_occupancy[block]
                candidate_count[fold] += 1
            spatial_mean = candidate.mean(axis=0)
            spatial_scale = spatial_mean + 1.0
            spatial_loss = float(
                np.sum(
                    np.square(candidate - spatial_mean[None, :])
                    / spatial_scale[None, :]
                )
            )
            count_loss = float(np.var(candidate_count)) * n_bins**2
            score = spatial_loss + count_loss
            if score < best_score:
                best_score = score
                best_folds = folds
        if best_folds is None:
            raise RuntimeError("failed to assign temporal blocks")
        for block, fold in zip(group, best_folds, strict=True):
            assignment[block] = fold
            accumulated[fold] += block_occupancy[block]
            block_count[fold] += 1

    frame_fold = np.full(xy.shape[0], -1, dtype=np.int64)
    for block, index in enumerate(block_interior):
        frame_fold[index] = assignment[block]
    if np.any(
        [np.count_nonzero(frame_fold == fold) == 0 for fold in range(n_fold)]
    ):
        raise ValueError("at least one balanced fold is empty")
    return frame_fold


def query_balanced_block_folds(
    position: ArrayLike,
    keep: ArrayLike,
    query: ArrayLike,
    *,
    bandwidth: float,
    n_fold: int = 4,
    block_frames: int = 1_800,
    guard_frames: int = 30,
    max_passes: int = 12,
) -> NDArray[np.int64]:
    """Balance guarded temporal blocks on predefined local-design features.

    This position-only assignment targets the exact frozen query support. Each
    consecutive group of ``n_fold`` blocks contributes one block to each fold.
    Coordinate descent over the within-group permutations minimizes imbalance
    in kernel mass and first/second local-design moments.
    """

    xy = np.asarray(position, dtype=np.float64)
    selected = np.asarray(keep, dtype=bool)
    queries = np.asarray(query, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("position must have shape (frame, 2)")
    if selected.shape != (xy.shape[0],):
        raise ValueError("keep must have shape (frame,)")
    if queries.ndim != 2 or queries.shape[1] != 2:
        raise ValueError("query must have shape (query, 2)")
    if bandwidth <= 0 or n_fold < 2:
        raise ValueError("bandwidth and fold count must be positive")
    if block_frames <= 2 * guard_frames or guard_frames < 0:
        raise ValueError("guarded block sizes are incompatible")

    n_block = int(np.ceil(xy.shape[0] / block_frames))
    block_interior: list[NDArray[np.int64]] = []
    block_feature: list[FloatArray] = []
    for block in range(n_block):
        start = block * block_frames
        end = min(xy.shape[0], (block + 1) * block_frames)
        interior_start = min(end, start + guard_frames)
        interior_end = max(interior_start, end - guard_frames)
        index = np.arange(interior_start, interior_end, dtype=np.int64)
        index = index[selected[index]]
        block_interior.append(index)

        offset = (
            xy[index, None, :] - queries[None, :, :]
        ) / bandwidth
        radius = np.linalg.norm(offset, axis=2)
        weight = np.zeros_like(radius)
        inside = radius < 1.0
        weight[inside] = (1.0 - radius[inside] ** 3) ** 3
        ux = offset[:, :, 0]
        uy = offset[:, :, 1]
        feature = np.stack(
            (
                weight.sum(axis=0),
                np.square(weight).sum(axis=0),
                (weight * ux).sum(axis=0),
                (weight * uy).sum(axis=0),
                (weight * ux**2).sum(axis=0),
                (weight * uy**2).sum(axis=0),
                (weight * ux * uy).sum(axis=0),
            ),
            axis=1,
        )
        block_feature.append(feature.ravel())

    feature_array = np.stack(block_feature)
    scale = np.sum(np.abs(feature_array), axis=0)
    informative = scale > np.finfo(float).eps
    feature_array = (
        feature_array[:, informative] / scale[informative][None, :]
    )
    feature_array = np.column_stack(
        (
            feature_array,
            np.ones(n_block, dtype=np.float64) / max(1, n_block),
        )
    )

    groups = [
        np.arange(start, min(n_block, start + n_fold))
        for start in range(0, n_block, n_fold)
    ]
    assignment = np.arange(n_block, dtype=np.int64) % n_fold

    def objective(value: NDArray[np.int64]) -> float:
        total = np.stack(
            [
                feature_array[value == fold].sum(axis=0)
                for fold in range(n_fold)
            ]
        )
        return float(np.sum(np.square(total - total.mean(axis=0))))

    best_loss = objective(assignment)
    for _ in range(max_passes):
        changed = False
        for group in groups:
            current = assignment[group].copy()
            group_best = current
            group_loss = best_loss
            for folds in itertools.permutations(range(n_fold), group.size):
                candidate = assignment.copy()
                candidate[group] = folds
                loss = objective(candidate)
                if loss < group_loss - 1e-15:
                    group_loss = loss
                    group_best = np.asarray(folds, dtype=np.int64)
            if not np.array_equal(group_best, current):
                assignment[group] = group_best
                best_loss = group_loss
                changed = True
        if not changed:
            break

    frame_fold = np.full(xy.shape[0], -1, dtype=np.int64)
    for block, index in enumerate(block_interior):
        frame_fold[index] = assignment[block]
    if np.any(
        [np.count_nonzero(frame_fold == fold) == 0 for fold in range(n_fold)]
    ):
        raise ValueError("at least one query-balanced fold is empty")
    return frame_fold


def occupancy_balance_weights(
    positions: list[ArrayLike],
    keeps: list[ArrayLike],
    *,
    arena_size_cm: float = 75.0,
    bin_size_cm: float = 5.0,
) -> list[FloatArray]:
    """Return sample weights that match coarse occupancy across sessions.

    In each physical bin, the target count is the minimum across sessions.
    Before within-session normalization, each session contributes that target
    weight per mutually visited bin. The final weights are normalized to the
    retained frame count in each session, so sessions share the same relative
    coarse occupancy profile but can have different absolute bin totals.
    """

    if len(positions) != len(keeps) or not positions:
        raise ValueError("positions and keeps must be non-empty equal lists")
    n_bins_float = arena_size_cm / bin_size_cm
    n_bins = int(round(n_bins_float))
    if not np.isclose(n_bins, n_bins_float):
        raise ValueError("arena size must be divisible by bin size")

    bin_index: list[NDArray[np.int64]] = []
    counts: list[FloatArray] = []
    keep_arrays: list[BoolArray] = []
    for position, keep in zip(positions, keeps, strict=True):
        xy = np.asarray(position, dtype=np.float64)
        selected = np.asarray(keep, dtype=bool)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("each position must have shape (frame, 2)")
        if selected.shape != (xy.shape[0],):
            raise ValueError("each keep must have shape (frame,)")
        clipped = np.minimum(
            np.floor(xy / bin_size_cm).astype(np.int64), n_bins - 1
        )
        flat = clipped[:, 1] * n_bins + clipped[:, 0]
        bin_index.append(flat)
        keep_arrays.append(selected)
        counts.append(
            np.bincount(flat[selected], minlength=n_bins**2).astype(
                np.float64
            )
        )

    target = np.min(np.stack(counts), axis=0)
    output: list[FloatArray] = []
    for flat, count, selected in zip(
        bin_index, counts, keep_arrays, strict=True
    ):
        weight = np.zeros(flat.size, dtype=np.float64)
        positive = selected & (count[flat] > 0)
        weight[positive] = target[flat[positive]] / count[flat[positive]]
        if weight[selected].sum() <= 0:
            raise ValueError("sessions have no mutually occupied spatial bins")
        weight *= selected.sum() / weight[selected].sum()
        output.append(weight)
    return output


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


def residual_directional_reliability(
    condition_jacobians: ArrayLike,
    reference_jacobians: list[ArrayLike],
    normal: ArrayLike,
    tangent: ArrayLike,
    *,
    valid: ArrayLike | None = None,
) -> DirectionalReliability:
    """Correlate boundary-residual components across two disjoint fold pairs."""

    condition = np.asarray(condition_jacobians, dtype=np.float64)
    references = [
        np.asarray(value, dtype=np.float64)
        for value in reference_jacobians
    ]
    if condition.ndim != 4 or condition.shape[0] != 4:
        raise ValueError("condition_jacobians must contain exactly four folds")
    if not references or any(value.shape != condition.shape for value in references):
        raise ValueError("reference Jacobians must match the condition")

    condition_pair = (
        cross_metric(condition[:2]),
        cross_metric(condition[2:]),
    )
    reference_pair = []
    for fold_slice in (slice(0, 2), slice(2, 4)):
        reference_pair.append(
            np.mean(
                [
                    cross_metric(reference[fold_slice])
                    for reference in references
                ],
                axis=0,
            )
        )
    component = [
        anisotropy_components(
            condition_pair[index],
            reference_pair[index],
            normal,
            tangent=tangent,
        )
        for index in range(2)
    ]
    if valid is None:
        keep = np.ones(condition.shape[1], dtype=bool)
    else:
        keep = np.asarray(valid, dtype=bool)
        if keep.shape != (condition.shape[1],):
            raise ValueError("valid must have shape (query,)")
    return DirectionalReliability(
        normal=_metric_correlation(
            component[0]["delta_normal"][keep],
            component[1]["delta_normal"][keep],
        ),
        tangent=_metric_correlation(
            component[0]["delta_tangent"][keep],
            component[1]["delta_tangent"][keep],
        ),
        contrast=_metric_correlation(
            component[0]["contrast"][keep],
            component[1]["contrast"][keep],
        ),
        n_query=int(keep.sum()),
    )


def estimate_session_metric(
    position: ArrayLike,
    response: ArrayLike,
    query: ArrayLike,
    *,
    common_blocked: ArrayLike,
    config: LocalMapConfig,
    n_fold: int = 4,
    sample_weight: ArrayLike | None = None,
    fold_assignment: ArrayLike | None = None,
) -> SessionMetric:
    """Estimate a metric after applying a common spatial-support operator."""

    xy = np.asarray(position, dtype=np.float64)
    event = np.asarray(response, dtype=np.float64)
    queries = np.asarray(query, dtype=np.float64)
    if event.ndim != 2 or event.shape[0] != xy.shape[0]:
        raise ValueError("response must have shape (frame, neuron)")
    keep = positions_on_accessible_support(xy, common_blocked)
    keep &= np.isfinite(xy).all(axis=1) & np.isfinite(event).all(axis=1)
    if fold_assignment is None:
        fold = temporal_folds(xy.shape[0], n_fold)
    else:
        fold = np.asarray(fold_assignment, dtype=np.int64)
        if fold.shape != (xy.shape[0],):
            raise ValueError("fold_assignment must have shape (frame,)")
        allowed = (fold == -1) | ((fold >= 0) & (fold < n_fold))
        if not np.all(allowed):
            raise ValueError("fold assignments must be -1 or valid fold IDs")
    if sample_weight is None:
        weight = np.ones(xy.shape[0], dtype=np.float64)
    else:
        weight = np.asarray(sample_weight, dtype=np.float64)
        if weight.shape != (xy.shape[0],):
            raise ValueError("sample_weight must have shape (frame,)")

    local = []
    frames = np.zeros(n_fold, dtype=np.int64)
    for fold_index in range(n_fold):
        selected = keep & (fold == fold_index) & (weight > 0)
        frames[fold_index] = int(selected.sum())
        local.append(
            fit_local_linear(
                xy[selected],
                event[selected],
                queries,
                config,
                sample_weight=weight[selected],
            )
        )

    jacobian = np.stack([item.jacobian for item in local])
    valid = np.logical_and.reduce([item.valid for item in local])
    metric = cross_metric(jacobian)
    pooled_local = fit_local_linear(
        xy[keep],
        event[keep],
        queries,
        config,
        sample_weight=weight[keep],
    )
    pooled = pooled_metric(pooled_local.jacobian)
    if n_fold >= 4:
        split = n_fold // 2
        pair_first = cross_metric(jacobian[:split])
        pair_second = cross_metric(jacobian[split:])
        reliability = _metric_correlation(
            pair_first[valid], pair_second[valid]
        )
    else:
        reliability = float("nan")
    return SessionMetric(
        metric=metric,
        pooled=pooled,
        jacobians=jacobian,
        pooled_jacobian=pooled_local.jacobian,
        valid=valid,
        effective_n=np.stack([item.effective_n for item in local]),
        design_eigenratio=np.stack(
            [item.design_eigenratio for item in local]
        ),
        frames_per_fold=frames,
        split_pair_reliability=reliability,
    )
