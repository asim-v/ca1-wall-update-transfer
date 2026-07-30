"""Behavioral diagnostics for pairs of arena partitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import jensenshannon
from scipy.stats import rankdata


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PartitionBehavior:
    """Downsampled partition labels, local coordinates, and features."""

    partition: IntArray
    local_position: FloatArray
    features: FloatArray


def partition_behavior(position: ArrayLike) -> PartitionBehavior:
    """Build tile-local kinematic features from a 2 Hz trajectory."""

    xy = np.asarray(position, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("position must have shape (sample, 2)")
    if not np.isfinite(xy).all():
        raise ValueError("position must be finite")
    if np.any((xy < 0.0) | (xy > 75.0)):
        raise ValueError("positions must lie within the 75 cm arena")

    column = np.clip(np.floor(xy[:, 0] / 25.0).astype(int), 0, 2)
    physical_y_cell = np.clip(
        np.floor(xy[:, 1] / 25.0).astype(int),
        0,
        2,
    )
    paper_row = 2 - physical_y_cell
    partition = (3 * paper_row + column).astype(np.int64)
    local = np.column_stack(
        (
            xy[:, 0] - 25.0 * column,
            xy[:, 1] - 25.0 * physical_y_cell,
        )
    )

    sample = np.arange(xy.shape[0])
    index = lambda offset: np.clip(  # noqa: E731
        sample + offset,
        0,
        xy.shape[0] - 1,
    )
    velocity = xy[index(1)] - xy[index(-1)]
    acceleration = xy[index(2)] - 2.0 * xy + xy[index(-2)]
    past_displacement = xy - xy[index(-4)]
    future_displacement = xy[index(4)] - xy
    speed = np.linalg.norm(velocity, axis=1)
    safe_speed = np.maximum(speed, np.finfo(np.float64).eps)
    cosine = velocity[:, 0] / safe_speed
    sine = velocity[:, 1] / safe_speed
    stopped = speed <= np.finfo(np.float64).eps
    cosine[stopped] = 0.0
    sine[stopped] = 0.0
    features = np.column_stack(
        (
            local,
            velocity,
            acceleration,
            past_displacement,
            future_displacement,
            speed,
            cosine,
            sine,
            cosine**2 - sine**2,
            2.0 * cosine * sine,
        )
    )
    return PartitionBehavior(
        partition=partition,
        local_position=local,
        features=features,
    )


def _cardinal_side(source: int, target: int) -> int | None:
    source_row, source_column = divmod(source, 3)
    target_row, target_column = divmod(target, 3)
    delta = source_row - target_row, source_column - target_column
    # N, E, S, W
    return {
        (-1, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (0, -1): 3,
    }.get(delta)


def route_counts(
    partition: ArrayLike,
    target: int,
    *,
    minimum_samples: int = 2,
) -> tuple[IntArray, int]:
    """Count ordered entry-side to exit-side visits through one partition."""

    value = np.asarray(partition, dtype=np.int64).ravel()
    if not 0 <= target <= 8:
        raise ValueError("target must lie in [0, 8]")
    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")
    selected = value == target
    transitions = np.diff(
        np.concatenate(([False], selected, [False])).astype(np.int8)
    )
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    counts = np.zeros(16, dtype=np.int64)
    qualifying = 0
    for start, stop in zip(starts, stops, strict=True):
        if (
            stop - start < minimum_samples
            or start == 0
            or stop == value.size
        ):
            continue
        entry = _cardinal_side(int(value[start - 1]), target)
        exit_side = _cardinal_side(int(value[stop]), target)
        if entry is None or exit_side is None:
            continue
        counts[4 * entry + exit_side] += 1
        qualifying += 1
    return counts, qualifying


def route_js_divergence(
    partition: ArrayLike,
    pair: tuple[int, int],
    *,
    minimum_samples: int = 2,
) -> tuple[float, tuple[int, int]]:
    """Return entry-exit route divergence in bits for a partition pair."""

    first, first_visits = route_counts(
        partition,
        pair[0],
        minimum_samples=minimum_samples,
    )
    second, second_visits = route_counts(
        partition,
        pair[1],
        minimum_samples=minimum_samples,
    )
    if first.sum() == 0 or second.sum() == 0:
        return float("nan"), (first_visits, second_visits)
    divergence = jensenshannon(
        first / first.sum(),
        second / second.sum(),
        base=2.0,
    )
    return float(divergence**2), (first_visits, second_visits)


def auc_score(label: ArrayLike, score: ArrayLike) -> float:
    """Compute tie-corrected binary AUC from continuous scores."""

    y = np.asarray(label, dtype=np.int64).ravel()
    value = np.asarray(score, dtype=np.float64).ravel()
    if y.shape != value.shape or not np.isin(y, (0, 1)).all():
        raise ValueError("label and score must be equal binary vectors")
    positive = y == 1
    n_positive = int(np.count_nonzero(positive))
    n_negative = int(y.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(value)
    statistic = ranks[positive].sum() - n_positive * (
        n_positive + 1
    ) / 2.0
    return float(statistic / (n_positive * n_negative))


def blocked_ridge_auc(
    features: ArrayLike,
    label: ArrayLike,
    sample_index: ArrayLike,
    *,
    samples_per_group: int = 120,
    n_folds: int = 4,
    ridge: float = 3.0,
    minimum_class_samples: int = 10,
) -> float:
    """Cross-validated linear twin-identity decoding with held-out blocks."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(label, dtype=np.int64).ravel()
    index = np.asarray(sample_index, dtype=np.int64).ravel()
    if x.ndim != 2 or x.shape[0] != y.size or y.shape != index.shape:
        raise ValueError("features, labels, and indices have incompatible shape")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(x).all():
        raise ValueError("features must be finite and labels binary")
    if (
        samples_per_group <= 0
        or n_folds < 2
        or ridge < 0
        or minimum_class_samples <= 0
    ):
        raise ValueError("invalid cross-validation setting")

    group = index // samples_per_group
    prediction = np.full(y.size, np.nan, dtype=np.float64)
    target = 2.0 * y - 1.0
    for fold in range(n_folds):
        test = group % n_folds == fold
        train = ~test
        if any(
            np.count_nonzero(y[mask] == label_value)
            < minimum_class_samples
            for mask in (train, test)
            for label_value in (0, 1)
        ):
            continue
        mean = np.mean(x[train], axis=0)
        scale = np.std(x[train], axis=0)
        scale[scale <= np.finfo(np.float64).eps] = 1.0
        train_design = np.column_stack(
            (np.ones(np.count_nonzero(train)), (x[train] - mean) / scale)
        )
        test_design = np.column_stack(
            (np.ones(np.count_nonzero(test)), (x[test] - mean) / scale)
        )
        penalty = np.eye(train_design.shape[1]) * ridge
        penalty[0, 0] = 0.0
        beta = np.linalg.pinv(
            train_design.T @ train_design + penalty
        ) @ (train_design.T @ target[train])
        prediction[test] = test_design @ beta
    finite = np.isfinite(prediction)
    return auc_score(y[finite], prediction[finite])


def behavior_twin_auc(
    behavior: PartitionBehavior,
    pair: tuple[int, int],
    *,
    samples_per_group: int = 120,
) -> tuple[float, tuple[int, int]]:
    """Decode pair identity from tile-local trajectory features only."""

    selected = np.isin(behavior.partition, pair)
    label = (behavior.partition[selected] == pair[1]).astype(np.int64)
    counts = (
        int(np.count_nonzero(label == 0)),
        int(np.count_nonzero(label == 1)),
    )
    return (
        blocked_ridge_auc(
            behavior.features[selected],
            label,
            np.flatnonzero(selected),
            samples_per_group=samples_per_group,
        ),
        counts,
    )


def expanded_behavior_features(
    behavior: PartitionBehavior,
    *,
    spatial_bins: int = 5,
) -> FloatArray:
    """Add local-position bins and position-by-heading interactions."""

    if spatial_bins <= 0:
        raise ValueError("spatial_bins must be positive")
    binned = np.floor(
        behavior.local_position * (spatial_bins / 25.0)
    ).astype(int)
    binned = np.clip(binned, 0, spatial_bins - 1)
    flat = binned[:, 1] * spatial_bins + binned[:, 0]
    one_hot = np.eye(spatial_bins**2, dtype=np.float64)[flat]
    # The final four columns are cos(theta), sin(theta), cos(2 theta), and
    # sin(2 theta). Their interaction with position flexibly absorbs
    # direction-dependent sampling within each local bin.
    heading = behavior.features[:, -4:]
    interaction = np.einsum("ni,nj->nij", one_hot, heading).reshape(
        behavior.features.shape[0],
        -1,
    )
    return np.column_stack((behavior.features, one_hot, interaction))


def _training_standardize(
    train: FloatArray,
    test: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _centroid_scores(
    train: FloatArray,
    train_label: IntArray,
    test: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    train_z, test_z = _training_standardize(train, test)
    mean_zero = np.mean(train_z[train_label == 0], axis=0)
    mean_one = np.mean(train_z[train_label == 1], axis=0)
    weight = mean_one - mean_zero
    midpoint = 0.5 * (mean_one + mean_zero)
    denominator = max(
        float(np.linalg.norm(weight)),
        np.finfo(np.float64).eps,
    )
    return (
        (train_z - midpoint) @ weight / denominator,
        (test_z - midpoint) @ weight / denominator,
    )


def blocked_conditional_neural_auc(
    behavior_features: ArrayLike,
    neural_activity: ArrayLike,
    label: ArrayLike,
    sample_index: ArrayLike,
    *,
    samples_per_group: int = 120,
    n_folds: int = 4,
    behavior_ridge: float = 10.0,
    minimum_class_samples: int = 10,
) -> tuple[float, float]:
    """Decode identity from neural activity before and after behavior removal.

    In each held-out temporal fold, a multivariate ridge model predicts every
    neural feature from tile-local position and trajectory. A diagonal
    nearest-centroid decoder is then trained on either the raw population or
    the behavior-residual population. Both scores are strictly out of fold.
    """

    behavior = np.asarray(behavior_features, dtype=np.float64)
    neural = np.asarray(neural_activity, dtype=np.float64)
    y = np.asarray(label, dtype=np.int64).ravel()
    index = np.asarray(sample_index, dtype=np.int64).ravel()
    if (
        behavior.ndim != 2
        or neural.ndim != 2
        or behavior.shape[0] != neural.shape[0]
        or neural.shape[0] != y.size
        or y.shape != index.shape
    ):
        raise ValueError("behavior, neural, label, and index shapes disagree")
    if (
        not np.isfinite(behavior).all()
        or not np.isfinite(neural).all()
        or not np.isin(y, (0, 1)).all()
    ):
        raise ValueError("features must be finite and labels binary")
    if behavior_ridge < 0:
        raise ValueError("behavior_ridge must be non-negative")

    group = index // samples_per_group
    raw_prediction = np.full(y.size, np.nan, dtype=np.float64)
    residual_prediction = np.full(y.size, np.nan, dtype=np.float64)
    for fold in range(n_folds):
        test = group % n_folds == fold
        train = ~test
        if any(
            np.count_nonzero(y[mask] == label_value)
            < minimum_class_samples
            for mask in (train, test)
            for label_value in (0, 1)
        ):
            continue

        _, raw_test_score = _centroid_scores(
            neural[train],
            y[train],
            neural[test],
        )
        raw_prediction[test] = raw_test_score

        behavior_train, behavior_test = _training_standardize(
            behavior[train],
            behavior[test],
        )
        train_design = np.column_stack(
            (np.ones(np.count_nonzero(train)), behavior_train)
        )
        test_design = np.column_stack(
            (np.ones(np.count_nonzero(test)), behavior_test)
        )
        penalty = np.eye(train_design.shape[1]) * behavior_ridge
        penalty[0, 0] = 0.0
        beta = np.linalg.pinv(
            train_design.T @ train_design + penalty
        ) @ (train_design.T @ neural[train])
        train_residual = neural[train] - train_design @ beta
        test_residual = neural[test] - test_design @ beta
        _, residual_test_score = _centroid_scores(
            train_residual,
            y[train],
            test_residual,
        )
        residual_prediction[test] = residual_test_score

    raw_finite = np.isfinite(raw_prediction)
    residual_finite = np.isfinite(residual_prediction)
    return (
        auc_score(y[raw_finite], raw_prediction[raw_finite]),
        auc_score(
            y[residual_finite],
            residual_prediction[residual_finite],
        ),
    )
