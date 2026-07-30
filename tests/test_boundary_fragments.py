import unittest

import numpy as np

from ca1_geometry.boundary_fragments import (
    common_support_bins,
    globally_demeaned_local_cell_rate,
    independent_square_residual_correlation,
    local_cell_rate,
    occupancy_weighted_cell_rate,
    spearman_correlation,
)


class BoundaryFragmentTests(unittest.TestCase):
    def test_common_support_requires_every_session(self) -> None:
        first = np.full((3, 3), 2.0)
        second = np.full((3, 3), 2.0)
        second[1, 1] = 0.25
        result = common_support_bins(
            [first, second],
            ((0, 0), (1, 1), (2, 2)),
            minimum_seconds=0.5,
        )
        self.assertEqual(result, ((0, 0), (2, 2)))

    def test_local_rate_uses_equal_bin_weight(self) -> None:
        maps = np.zeros((2, 2, 2))
        maps[:, 0, 0] = [1.0, 4.0]
        maps[:, 1, 1] = [3.0, 8.0]
        value = local_cell_rate(maps, [0, 1], ((0, 0), (1, 1)))
        np.testing.assert_allclose(value, [2.0, 6.0])

    def test_global_demeaning_removes_session_wide_cell_rate(self) -> None:
        maps = np.array(
            [
                [[1.0, 3.0], [5.0, 7.0]],
                [[2.0, 4.0], [6.0, 8.0]],
            ]
        )
        occupancy = np.array([[1.0, 1.0], [1.0, 3.0]])
        global_rate = occupancy_weighted_cell_rate(
            maps,
            occupancy,
            [0, 1],
        )
        np.testing.assert_allclose(global_rate, [5.0, 6.0])
        demeaned = globally_demeaned_local_cell_rate(
            maps,
            occupancy,
            [0, 1],
            ((0, 0), (0, 1)),
        )
        np.testing.assert_allclose(demeaned, [-3.0, -3.0])

    def test_spearman_is_invariant_to_monotonic_scale(self) -> None:
        x = np.arange(10.0)
        self.assertAlmostEqual(
            spearman_correlation(x, np.exp(x)),
            1.0,
        )

    def test_independent_baselines_recover_shared_edit(self) -> None:
        pre = np.array([1.0, 2.0, 3.0, 4.0])
        post = np.array([4.0, 1.0, 3.0, 2.0])
        edit = np.array([-2.0, -1.0, 1.0, 2.0])
        result = independent_square_residual_correlation(
            pre + edit,
            post + edit,
            pre,
            post,
        )
        self.assertAlmostEqual(result.first_assignment, 1.0)
        # The swapped assignment intentionally exposes baseline drift.
        self.assertTrue(np.isfinite(result.second_assignment))


if __name__ == "__main__":
    unittest.main()
