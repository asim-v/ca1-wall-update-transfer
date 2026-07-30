import unittest

import numpy as np

from ca1_geometry.synthetic import center_occlusion_normal_warp


class CenterOcclusionWarpTests(unittest.TestCase):
    def test_face_normal_derivative_and_tangent_are_exact(self) -> None:
        amplitude = 0.6
        decay = 8.0
        distance = 4.0
        step = 1e-4
        cases = (
            (np.array([50.0 + distance, 37.5]), np.array([1.0, 0.0])),
            (np.array([25.0 - distance, 37.5]), np.array([-1.0, 0.0])),
            (np.array([37.5, 50.0 + distance]), np.array([0.0, 1.0])),
            (np.array([37.5, 25.0 - distance]), np.array([0.0, -1.0])),
        )
        expected_normal = 1.0 + amplitude * np.exp(-distance / decay)
        for point, normal in cases:
            tangent = np.array([-normal[1], normal[0]])
            plus_normal = center_occlusion_normal_warp(
                [point + step * normal],
                amplitude=amplitude,
                decay=decay,
            )[0]
            minus_normal = center_occlusion_normal_warp(
                [point - step * normal],
                amplitude=amplitude,
                decay=decay,
            )[0]
            derivative_normal = (
                plus_normal - minus_normal
            ) / (2.0 * step)
            plus_tangent = center_occlusion_normal_warp(
                [point + step * tangent],
                amplitude=amplitude,
                decay=decay,
            )[0]
            minus_tangent = center_occlusion_normal_warp(
                [point - step * tangent],
                amplitude=amplitude,
                decay=decay,
            )[0]
            derivative_tangent = (
                plus_tangent - minus_tangent
            ) / (2.0 * step)
            self.assertAlmostEqual(
                float(derivative_normal @ normal),
                expected_normal,
                places=6,
            )
            self.assertAlmostEqual(
                float(derivative_normal @ tangent), 0.0, places=6
            )
            self.assertAlmostEqual(
                float(derivative_tangent @ tangent), 1.0, places=6
            )
            self.assertAlmostEqual(
                float(derivative_tangent @ normal), 0.0, places=6
            )

    def test_corner_sector_is_unchanged(self) -> None:
        corner = np.array([[60.0, 60.0], [15.0, 15.0]])
        warped = center_occlusion_normal_warp(
            corner, amplitude=0.6, decay=8.0
        )
        np.testing.assert_array_equal(warped, corner)

    def test_zero_amplitude_is_identity(self) -> None:
        position = np.array(
            [[55.0, 37.5], [20.0, 37.5], [37.5, 55.0], [37.5, 20.0]]
        )
        warped = center_occlusion_normal_warp(
            position, amplitude=0.0, decay=8.0
        )
        np.testing.assert_array_equal(warped, position)


if __name__ == "__main__":
    unittest.main()
