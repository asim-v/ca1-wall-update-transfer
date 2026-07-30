import unittest

import numpy as np

from ca1_geometry.metrics import (
    anisotropy_components,
    anisotropy_profile,
    cross_metric,
    pooled_metric,
)


class MetricTests(unittest.TestCase):
    def test_cross_metric_is_rotation_invariant(self) -> None:
        rng = np.random.default_rng(21)
        jacobian = rng.normal(size=(4, 9, 16, 2))
        rotation, _ = np.linalg.qr(rng.normal(size=(16, 16)))
        rotated = np.einsum("mn,fqnj->fqmj", rotation, jacobian)
        np.testing.assert_allclose(
            cross_metric(jacobian),
            cross_metric(rotated),
            atol=1e-12,
        )

    def test_pooled_metric_is_permutation_invariant(self) -> None:
        rng = np.random.default_rng(22)
        jacobian = rng.normal(size=(7, 20, 2))
        permutation = rng.permutation(jacobian.shape[1])
        np.testing.assert_allclose(
            pooled_metric(jacobian),
            pooled_metric(jacobian[:, permutation]),
            atol=1e-12,
        )

    def test_cross_metric_removes_independent_noise_bias(self) -> None:
        rng = np.random.default_rng(23)
        # There is no true derivative: every fold contains independent
        # derivative-estimation noise.
        jacobian = rng.normal(size=(12, 40_000, 1, 2))
        plugin_trace = np.trace(
            np.mean(
                [
                    pooled_metric(jacobian[index])
                    for index in range(jacobian.shape[0])
                ],
                axis=0,
            ),
            axis1=1,
            axis2=2,
        ).mean()
        cross_trace = np.trace(
            cross_metric(jacobian), axis1=1, axis2=2
        ).mean()
        self.assertGreater(plugin_trace, 1.8)
        self.assertLess(abs(cross_trace), 0.02)

    def test_anisotropy_requires_normal_magnification(self) -> None:
        reference = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()
        condition = reference.copy()
        condition[:, 0, 0] += 0.5
        condition[:, 1, 1] -= 0.25
        result = anisotropy_components(
            condition, reference, normal=np.array([1.0, 0.0])
        )
        np.testing.assert_allclose(result["delta_normal"], 0.5)
        np.testing.assert_allclose(result["delta_tangent"], -0.25)
        np.testing.assert_allclose(result["contrast"], 0.75)

    def test_profile_can_use_separate_positive_denominator(self) -> None:
        metric = np.array([[[3.0, 0.0], [0.0, 1.0]]])
        reference = np.zeros((1, 2, 2))
        denominator = np.array([[[5.0, 0.0], [0.0, 5.0]]])
        profile = anisotropy_profile(
            metric,
            reference,
            normal=np.array([1.0, 0.0]),
            distance=np.array([1.0]),
            bin_edges=np.array([0.0, 2.0]),
            denominator_metric=denominator,
        )
        np.testing.assert_allclose(profile.anisotropy, [0.2])
        np.testing.assert_allclose(profile.normal_magnification, [0.3])


if __name__ == "__main__":
    unittest.main()
