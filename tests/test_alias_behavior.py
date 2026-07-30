import numpy as np
import pytest

from ca1_geometry.alias_behavior import (
    auc_score,
    behavior_twin_auc,
    blocked_conditional_neural_auc,
    blocked_ridge_auc,
    expanded_behavior_features,
    partition_behavior,
    route_counts,
    route_js_divergence,
)


def test_partition_behavior_maps_physical_y_to_paper_rows():
    behavior = partition_behavior(
        np.array(
            [
                [1.0, 74.0],
                [26.0, 49.0],
                [74.0, 1.0],
            ]
        )
    )
    assert behavior.partition.tolist() == [0, 4, 8]
    assert np.all((behavior.local_position >= 0) & (behavior.local_position < 25))
    assert behavior.features.shape == (3, 15)


def test_route_counts_entry_to_exit():
    # West neighbor -> center -> east neighbor.
    partition = np.array([3, 4, 4, 5, 5])
    counts, visits = route_counts(partition, 4)
    assert visits == 1
    assert counts[4 * 3 + 1] == 1


def test_route_js_is_zero_for_equal_route_distributions():
    partition = np.array(
        [3, 4, 4, 5, 3, 4, 4, 5, 3, 4, 4, 5]
    )
    value, visits = route_js_divergence(partition, (4, 4))
    assert visits == (3, 3)
    assert value == pytest.approx(0.0)


def test_auc_score():
    assert auc_score([0, 0, 1, 1], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert auc_score([0, 1], [1, 1]) == pytest.approx(0.5)


def test_blocked_ridge_decoder_recovers_a_feature_signal():
    rng = np.random.default_rng(4)
    y = np.tile(np.array([0, 1]), 400)
    x = np.column_stack((2.0 * y + rng.normal(scale=0.2, size=y.size),))
    auc = blocked_ridge_auc(
        x,
        y,
        np.arange(y.size),
        samples_per_group=20,
    )
    assert auc > 0.99


def test_behavior_twin_auc_runs_on_two_tiles():
    first = np.column_stack((np.linspace(1, 24, 300), np.full(300, 60.0)))
    second = np.column_stack((np.linspace(1, 24, 300), np.full(300, 10.0)))
    behavior = partition_behavior(np.vstack((first, second)))
    auc, counts = behavior_twin_auc(behavior, (0, 6), samples_per_group=20)
    assert counts == (300, 300)
    assert np.isfinite(auc)


def test_expanded_behavior_features_adds_local_interactions():
    behavior = partition_behavior(
        np.column_stack((np.linspace(1, 24, 20), np.full(20, 60.0)))
    )
    expanded = expanded_behavior_features(behavior)
    assert expanded.shape == (20, 15 + 25 + 25 * 4)
    assert np.isfinite(expanded).all()


def test_conditional_decoder_retains_label_specific_neural_signal():
    rng = np.random.default_rng(12)
    n_samples = 800
    label = np.tile(np.array([0, 1]), n_samples // 2)
    behavior = rng.normal(size=(n_samples, 6))
    neural = rng.normal(scale=0.3, size=(n_samples, 12))
    neural[:, 0] += 2.0 * label
    raw_auc, conditional_auc = blocked_conditional_neural_auc(
        behavior,
        neural,
        label,
        np.arange(n_samples),
        samples_per_group=20,
    )
    assert raw_auc > 0.95
    assert conditional_auc > 0.95
