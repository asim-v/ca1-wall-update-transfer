import unittest

import numpy as np

from ca1_geometry.pilot import (
    balanced_block_folds,
    occupancy_balance_weights,
    query_balanced_block_folds,
    residual_directional_reliability,
    temporal_folds,
)


class PilotTests(unittest.TestCase):
    def test_temporal_folds_are_contiguous_and_balanced(self) -> None:
        fold = temporal_folds(103, 4)
        self.assertTrue(np.all(np.diff(fold) >= 0))
        self.assertLessEqual(np.bincount(fold).max() - np.bincount(fold).min(), 1)

    def test_occupancy_weights_match_bin_totals(self) -> None:
        first = np.array([[1.0, 1.0]] * 3 + [[6.0, 1.0]] * 2)
        second = np.array([[1.0, 1.0]] * 2 + [[6.0, 1.0]] * 4)
        keep = [np.ones(first.shape[0], bool), np.ones(second.shape[0], bool)]
        weights = occupancy_balance_weights(
            [first, second],
            keep,
            arena_size_cm=10.0,
            bin_size_cm=5.0,
        )
        first_total = [weights[0][:3].sum(), weights[0][3:].sum()]
        second_total = [weights[1][:2].sum(), weights[1][2:].sum()]
        ratio = np.asarray(first_total) / np.asarray(second_total)
        np.testing.assert_allclose(ratio, np.full(2, ratio[0]))

    def test_balanced_blocks_are_guarded_and_distributed(self) -> None:
        # Four different occupancy regimes recur twice. A fold should receive
        # one block from every group, with guards left unassigned.
        position = np.repeat(
            np.array(
                [[1.0, 1.0], [6.0, 1.0], [1.0, 6.0], [6.0, 6.0]]
            ),
            20,
            axis=0,
        )
        position = np.vstack((position, position))
        fold = balanced_block_folds(
            position,
            np.ones(position.shape[0], dtype=bool),
            n_fold=4,
            block_frames=20,
            guard_frames=2,
            arena_size_cm=10.0,
            bin_size_cm=5.0,
        )
        self.assertEqual(int(np.count_nonzero(fold == -1)), 32)
        np.testing.assert_array_equal(
            np.bincount(fold[fold >= 0], minlength=4),
            np.full(4, 32),
        )
        for block in range(8):
            start = block * 20
            self.assertTrue(np.all(fold[start : start + 2] == -1))
            self.assertTrue(np.all(fold[start + 18 : start + 20] == -1))

    def test_directional_residual_reliability_targets_components(self) -> None:
        query_value = np.linspace(0.5, 2.0, 20)
        reference = np.zeros((4, 20, 3, 2))
        condition = reference.copy()
        condition[:, :, 0, 0] = query_value
        condition[:, :, 1, 1] = 0.5
        result = residual_directional_reliability(
            condition,
            [reference],
            normal=np.array([1.0, 0.0]),
            tangent=np.array([0.0, 1.0]),
        )
        self.assertAlmostEqual(result.normal, 1.0)
        self.assertAlmostEqual(result.contrast, 1.0)
        self.assertTrue(np.isnan(result.tangent))

    def test_query_balanced_blocks_preserve_temporal_distribution(self) -> None:
        position = np.column_stack(
            (
                np.tile(np.linspace(0.0, 10.0, 20, endpoint=False), 8),
                np.repeat([2.0, 8.0], 80),
            )
        )
        fold = query_balanced_block_folds(
            position,
            np.ones(position.shape[0], dtype=bool),
            np.array([[2.5, 2.0], [7.5, 8.0]]),
            bandwidth=3.0,
            n_fold=4,
            block_frames=20,
            guard_frames=2,
        )
        for group_start in range(0, 8, 4):
            labels = []
            for block in range(group_start, group_start + 4):
                interior = fold[block * 20 + 2 : block * 20 + 18]
                labels.append(int(interior[0]))
                self.assertTrue(np.all(interior == interior[0]))
            self.assertEqual(set(labels), {0, 1, 2, 3})


if __name__ == "__main__":
    unittest.main()
