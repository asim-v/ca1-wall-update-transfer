import unittest

import numpy as np

from ca1_geometry.rate_maps import (
    authors_current_rate_maps,
    fixed_rate_maps,
    released_rate_maps,
)


class RateMapTests(unittest.TestCase):
    def test_fixed_map_uses_y_x_orientation_and_right_edge(self) -> None:
        position = np.array(
            [[1.0, 1.0], [9.0, 1.0], [1.0, 9.0], [10.0, 10.0]]
        )
        response = np.array([[1.0], [0.0], [1.0], [1.0]])
        result = fixed_rate_maps(
            position,
            response,
            arena_size_cm=10.0,
            bin_size_cm=5.0,
            frames_per_second=1.0,
        )
        np.testing.assert_allclose(result.occupancy, np.full((2, 2), 0.25))
        np.testing.assert_allclose(
            result.rate[0], np.array([[1.0, 0.0], [1.0, 1.0]])
        )

    def test_legacy_unsmoothed_matches_direct_loop(self) -> None:
        position = np.array(
            [[0.0, 0.0], [9.0, 1.0], [1.0, 9.0], [9.0, 9.0]]
        )
        response = np.array([[1.0], [0.0], [1.0], [1.0]])
        result = authors_current_rate_maps(
            position,
            response,
            n_bins=2,
            frames_per_second=1.0,
        )
        np.testing.assert_allclose(result.occupancy, np.full((2, 2), 0.25))
        np.testing.assert_allclose(
            result.rate[0], np.array([[1.0, 0.0], [1.0, 1.0]])
        )

    def test_released_convention_crops_upper_edge_and_uses_seconds(
        self,
    ) -> None:
        position = np.array(
            [[1.0, 1.0], [9.0, 1.0], [1.0, 9.0], [10.0, 10.0]]
        )
        response = np.array([[1.0], [0.0], [1.0], [100.0]])
        result = released_rate_maps(
            position,
            response,
            arena_size_cm=10.0,
            bin_size_cm=5.0,
            frames_per_second=2.0,
        )
        self.assertEqual(float(result.occupancy.sum()), 1.5)
        self.assertTrue(np.isnan(result.rate[0, 1, 1]))
        np.testing.assert_allclose(
            result.rate[0, 0], np.array([1.0, 0.0])
        )

    def test_released_smoothing_occurs_before_upper_edge_crop(self) -> None:
        position = np.array([[9.0, 9.0], [10.0, 9.0]])
        response = np.array([[0.0], [1.0]])
        result = released_rate_maps(
            position,
            response,
            arena_size_cm=10.0,
            bin_size_cm=5.0,
            frames_per_second=1.0,
            smoothing_sigma_bins=1.0,
        )
        self.assertGreater(result.rate[0, 1, 1], 0.0)

    def test_fixed_rejects_out_of_arena_positions(self) -> None:
        with self.assertRaises(ValueError):
            fixed_rate_maps(
                np.array([[75.1, 1.0]]),
                np.ones((1, 1)),
            )


if __name__ == "__main__":
    unittest.main()
