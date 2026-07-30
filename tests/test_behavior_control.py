import unittest

import numpy as np

from ca1_geometry.behavior_control import (
    estimate_behavior_adjusted_session_metric,
    fit_local_linear_with_nuisance,
    kinematic_covariates,
)
from ca1_geometry.local_linear import LocalMapConfig, fit_local_linear


class BehaviorControlTests(unittest.TestCase):
    def test_fixed_secant_velocity_recovers_constant_motion(self) -> None:
        frame_rate = 30.0
        time = np.arange(61) / frame_rate
        position = np.column_stack((12.0 * time, -5.0 * time))
        result = kinematic_covariates(position)
        np.testing.assert_allclose(
            result.velocity_cm_s,
            np.broadcast_to([12.0, -5.0], position.shape),
            atol=1e-12,
        )
        np.testing.assert_allclose(result.speed_cm_s, 13.0, atol=1e-12)
        self.assertEqual(
            result.names,
            (
                "speed_cm_s",
                "heading_cos_1",
                "heading_sin_1",
                "heading_cos_2",
                "heading_sin_2",
            ),
        )
        np.testing.assert_allclose(result.design, 0.0, atol=1e-12)

    def test_joint_fit_removes_correlated_nuisance_slope(self) -> None:
        rng = np.random.default_rng(341)
        position = rng.uniform(-1.0, 1.0, size=(8_000, 2))
        nuisance = np.column_stack(
            (
                position[:, 0] + 0.25 * rng.normal(size=position.shape[0]),
                position[:, 1] + 0.25 * rng.normal(size=position.shape[0]),
                rng.normal(size=(position.shape[0], 3)),
            )
        )
        loading = rng.normal(size=(9, 2))
        behavior_loading = rng.normal(size=(9, 5))
        response = (
            position @ loading.T + nuisance @ behavior_loading.T
        )
        query = np.array([[0.0, 0.0]])
        config = LocalMapConfig(
            bandwidth=1.5,
            min_effective_samples=40,
            min_design_eigenratio=0.001,
        )
        adjusted = fit_local_linear_with_nuisance(
            position, response, nuisance, query, config
        )
        unadjusted = fit_local_linear(
            position, response, query, config
        )
        self.assertTrue(adjusted.valid[0])
        np.testing.assert_allclose(
            adjusted.jacobian[0], loading, atol=2e-6
        )
        self.assertGreater(
            float(np.linalg.norm(unadjusted.jacobian[0] - loading)),
            0.25,
        )

    def test_weighted_fwl_matches_full_joint_regression(self) -> None:
        rng = np.random.default_rng(530)
        position = rng.uniform(-0.8, 0.8, size=(3_000, 2))
        nuisance = rng.normal(size=(3_000, 5))
        response = rng.normal(size=(3_000, 7))
        sample_weight = rng.uniform(0.2, 2.0, size=3_000)
        bandwidth = 1.25
        config = LocalMapConfig(
            bandwidth=bandwidth,
            min_effective_samples=20,
            min_design_eigenratio=0.001,
            ridge_relative=0.0,
        )
        result = fit_local_linear_with_nuisance(
            position,
            response,
            nuisance,
            np.array([[0.0, 0.0]]),
            config,
            sample_weight=sample_weight,
        )

        offset = position / bandwidth
        radius = np.linalg.norm(offset, axis=1)
        kernel = np.zeros(radius.size)
        inside = radius < 1.0
        kernel[inside] = (1.0 - radius[inside] ** 3) ** 3
        weight = sample_weight * kernel
        positive = weight > 0
        full_design = np.column_stack(
            (
                np.ones(np.count_nonzero(positive)),
                offset[positive],
                nuisance[positive],
            )
        )
        coefficient = np.linalg.lstsq(
            np.sqrt(weight[positive])[:, None] * full_design,
            np.sqrt(weight[positive])[:, None] * response[positive],
            rcond=1e-10,
        )[0]
        expected = coefficient[1:3].T / bandwidth
        self.assertTrue(result.valid[0])
        np.testing.assert_allclose(
            result.jacobian[0], expected, atol=2e-11
        )

    def test_fold_estimator_preserves_assignment_and_shapes(self) -> None:
        rng = np.random.default_rng(916)
        position = rng.uniform(0.0, 75.0, size=(4_000, 2))
        nuisance = rng.normal(size=(4_000, 5))
        loading = rng.normal(size=(6, 2))
        response = position @ loading.T + nuisance @ rng.normal(
            size=(5, 6)
        )
        fold = np.arange(position.shape[0]) % 4
        query = np.array([[25.0, 25.0], [50.0, 50.0]])
        result = estimate_behavior_adjusted_session_metric(
            position,
            response,
            nuisance,
            query,
            common_blocked=(),
            config=LocalMapConfig(
                bandwidth=15.0,
                min_effective_samples=20,
                min_design_eigenratio=0.001,
            ),
            n_fold=4,
            fold_assignment=fold,
        )
        np.testing.assert_array_equal(
            result.frames_per_fold, [1_000, 1_000, 1_000, 1_000]
        )
        self.assertEqual(result.jacobians.shape, (4, 2, 6, 2))
        self.assertEqual(
            result.adjusted_spatial_eigenratio.shape, (4, 2)
        )
        self.assertTrue(result.valid.all())
        for fold_index in range(4):
            np.testing.assert_allclose(
                result.jacobians[fold_index],
                np.broadcast_to(loading, (2, 6, 2)),
                atol=3e-6,
            )


if __name__ == "__main__":
    unittest.main()
