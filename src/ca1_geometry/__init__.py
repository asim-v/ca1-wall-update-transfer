"""Tools for cross-validated local geometry of CA1 population codes."""

from .local_linear import LocalMap, LocalMapConfig, fit_local_linear
from .metrics import (
    AnisotropyProfile,
    anisotropy_components,
    anisotropy_profile,
    cross_metric,
    pooled_metric,
)

__all__ = [
    "AnisotropyProfile",
    "LocalMap",
    "LocalMapConfig",
    "anisotropy_components",
    "anisotropy_profile",
    "cross_metric",
    "fit_local_linear",
    "pooled_metric",
]
