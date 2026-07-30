"""Response scaling shared across paired experimental conditions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PopulationScaler:
    """Per-neuron affine scaling fitted once on training data."""

    center: FloatArray
    scale: FloatArray

    @classmethod
    def fit(
        cls, response: ArrayLike, *, scale_floor: float = 1e-8
    ) -> "PopulationScaler":
        value = np.asarray(response, dtype=np.float64)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError("response must be a finite (sample, neuron) array")
        center = value.mean(axis=0)
        scale = value.std(axis=0, ddof=1)
        scale = np.where(scale > scale_floor, scale, 1.0)
        return cls(center=center, scale=scale)

    def transform(self, response: ArrayLike) -> FloatArray:
        value = np.asarray(response, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != self.center.size:
            raise ValueError("response has incompatible neuron dimension")
        return (value - self.center) / self.scale
