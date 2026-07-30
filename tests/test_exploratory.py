import unittest

import numpy as np

from ca1_geometry.exploratory import exact_segment_label_spin


class ExploratoryTests(unittest.TestCase):
    def test_segment_spin_is_exact_and_shared_across_exposures(self) -> None:
        contrast = np.array([[1.0, 2.0], [3.0, 4.0]])
        scale = np.ones_like(contrast)
        result = exact_segment_label_spin(
            contrast,
            scale,
            np.array([0, 1]),
        )
        self.assertEqual(result["n_exact_labelings"], 4)
        self.assertAlmostEqual(result["observed_statistic"], 2.5)
        self.assertEqual(result["segment_numerator"], [4.0, 6.0])
        np.testing.assert_allclose(
            result["exact_statistics"],
            [-2.5, 0.5, -0.5, 2.5],
        )
        self.assertAlmostEqual(result["two_sided_tail_fraction"], 0.5)

    def test_segment_spin_includes_unsupported_segments_as_zero(self) -> None:
        result = exact_segment_label_spin(
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.array([0, 2]),
            n_segments=3,
        )
        self.assertEqual(result["n_exact_labelings"], 8)
        self.assertEqual(result["segment_numerator"], [2.0, 0.0, 2.0])

    def test_segment_spin_rejects_too_small_segment_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "include"):
            exact_segment_label_spin(
                np.ones((2, 2)),
                np.ones((2, 2)),
                np.array([0, 2]),
                n_segments=2,
            )


if __name__ == "__main__":
    unittest.main()
