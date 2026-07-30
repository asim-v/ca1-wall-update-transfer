import unittest

import numpy as np

from ca1_geometry.local_linear import LocalMapConfig, fit_local_linear
from ca1_geometry.synthetic import linear_code


class LocalLinearTests(unittest.TestCase):
    def test_recovers_linear_jacobian(self) -> None:
        rng = np.random.default_rng(11)
        position = rng.uniform(-1, 1, size=(5_000, 2))
        loading = rng.normal(size=(12, 2))
        response = linear_code(position, loading, rng.normal(size=12))
        query = np.array([[0.0, 0.0], [0.7, 0.0], [-0.7, 0.4]])
        result = fit_local_linear(
            position,
            response,
            query,
            LocalMapConfig(
                bandwidth=0.35,
                min_effective_samples=30,
                min_design_eigenratio=0.01,
            ),
        )
        self.assertTrue(result.valid.all())
        np.testing.assert_allclose(
            result.jacobian,
            np.broadcast_to(loading, result.jacobian.shape),
            atol=2e-6,
        )

    def test_one_sided_support_recovers_linear_jacobian(self) -> None:
        rng = np.random.default_rng(12)
        position = np.column_stack(
            (rng.uniform(0, 1, 6_000), rng.uniform(-1, 1, 6_000))
        )
        loading = rng.normal(size=(8, 2))
        response = linear_code(position, loading)
        result = fit_local_linear(
            position,
            response,
            np.array([[0.03, 0.0]]),
            LocalMapConfig(
                bandwidth=0.30,
                min_effective_samples=30,
                min_design_eigenratio=0.005,
            ),
        )
        self.assertTrue(result.valid[0])
        np.testing.assert_allclose(result.jacobian[0], loading, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
