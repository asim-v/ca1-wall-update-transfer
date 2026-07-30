import unittest

import numpy as np

from ca1_geometry.seams import (
    OrientedSeam,
    SeamState,
    internal_seams,
    seam_frame,
    seam_state,
    seam_strip_bins,
)


class SeamTests(unittest.TestCase):
    def test_grid_has_twelve_physical_and_twenty_four_oriented_seams(
        self,
    ) -> None:
        physical = internal_seams(both_orientations=False)
        oriented = internal_seams()
        self.assertEqual(len(physical), 12)
        self.assertEqual(len(oriented), 24)
        self.assertEqual(len({item.unordered for item in oriented}), 12)

    def test_oriented_state_distinguishes_wall_direction(self) -> None:
        east = OrientedSeam(4, 5)
        west = OrientedSeam(5, 4)
        self.assertEqual(seam_state([4], east), SeamState.WALL)
        self.assertEqual(seam_state([4], west), SeamState.REVERSE_WALL)
        self.assertEqual(seam_state([], east), SeamState.OPEN)
        self.assertEqual(seam_state([4, 5], east), SeamState.CLOSED)

    def test_center_to_east_frame_and_strip_are_physical(self) -> None:
        seam = OrientedSeam(4, 5)
        start, end, normal = seam_frame(seam)
        np.testing.assert_allclose(normal, [1.0, 0.0])
        np.testing.assert_allclose(start, [50.0, 25.0])
        np.testing.assert_allclose(end, [50.0, 50.0])
        bins = seam_strip_bins(seam)
        self.assertEqual(len(bins), 15)
        self.assertEqual({x for _, x in bins}, {10, 11, 12})
        self.assertEqual({y for y, _ in bins}, {5, 6, 7, 8, 9})

    def test_nonadjacent_partitions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OrientedSeam(0, 8)

    def test_north_strip_uses_released_map_row_order(self) -> None:
        bins = seam_strip_bins(OrientedSeam(4, 1))
        self.assertEqual({row for row, _ in bins}, {2, 3, 4})
        self.assertEqual({column for _, column in bins}, {5, 6, 7, 8, 9})

    def test_north_frame_uses_raw_image_coordinate_direction(self) -> None:
        start, end, normal = seam_frame(OrientedSeam(4, 1))
        np.testing.assert_allclose(normal, [0.0, -1.0])
        self.assertTrue(np.all(np.asarray([start[1], end[1]]) == 25.0))


if __name__ == "__main__":
    unittest.main()
