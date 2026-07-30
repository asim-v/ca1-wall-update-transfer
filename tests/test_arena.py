import unittest

import numpy as np

from ca1_geometry.arena import (
    introduced_boundaries,
    midpoint_boundary_queries,
    partition_accessibility,
    positions_on_accessible_support,
    spatial_accessibility,
)


class ArenaTests(unittest.TestCase):
    def test_center_occlusion_has_four_internal_boundaries(self) -> None:
        segment = introduced_boundaries([4])
        self.assertEqual(len(segment), 4)
        self.assertEqual(
            {item.normal for item in segment},
            {(0.0, 1.0), (0.0, -1.0), (-1.0, 0.0), (1.0, 0.0)},
        )

    def test_square_has_no_introduced_boundaries(self) -> None:
        self.assertEqual(introduced_boundaries([]), [])

    def test_midpoint_queries_extend_into_accessible_space(self) -> None:
        queries = midpoint_boundary_queries(
            introduced_boundaries([4]), [2.5, 7.5]
        )
        self.assertEqual(queries.position.shape, (8, 2))
        np.testing.assert_allclose(
            queries.position
            - queries.distance[:, None] * queries.normal,
            np.repeat(
                np.array(
                    [
                        [37.5, 25.0],
                        [37.5, 50.0],
                        [25.0, 37.5],
                        [50.0, 37.5],
                    ]
                ),
                2,
                axis=0,
            ),
        )

    def test_partition_and_spatial_mask(self) -> None:
        partition = partition_accessibility([0, 8])
        self.assertFalse(partition[0, 0])
        self.assertFalse(partition[2, 2])
        spatial = spatial_accessibility([0, 8])
        self.assertEqual(spatial.shape, (15, 15))
        self.assertEqual(int(spatial.sum()), 175)

    def test_position_support_uses_released_image_y_orientation(self) -> None:
        position = np.array(
            [
                [5.0, 5.0],
                [5.0, 70.0],
                [70.0, 5.0],
                [70.0, 70.0],
                [25.0, 25.0],
            ]
        )
        keep = positions_on_accessible_support(position, [0])
        np.testing.assert_array_equal(
            keep,
            [False, True, True, True, True],
        )


if __name__ == "__main__":
    unittest.main()
