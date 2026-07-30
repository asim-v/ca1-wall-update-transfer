"""Small utilities for explicitly exploratory post-outcome diagnostics."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def exact_segment_label_spin(
    contrast: ArrayLike,
    denominator: ArrayLike,
    segment_index: ArrayLike,
    *,
    valid: ArrayLike | None = None,
    n_segments: int | None = None,
) -> dict[str, Any]:
    """Enumerate segment-wise normal/tangent label spins exactly.

    The first two arrays have shape ``(exposure, query)``. A segment receives
    one sign that is preserved across all exposures, corresponding to swapping
    the normal and tangent labels for every query assigned to that segment.
    The returned tail fractions are calibration diagnostics, not randomized-
    experiment p-values.
    """

    value = np.asarray(contrast, dtype=np.float64)
    scale = np.asarray(denominator, dtype=np.float64)
    segment = np.asarray(segment_index, dtype=np.int64)
    if value.ndim != 2 or scale.shape != value.shape:
        raise ValueError(
            "contrast and denominator must have equal (exposure, query) shape"
        )
    if segment.shape != (value.shape[1],):
        raise ValueError("segment_index must have shape (query,)")
    if np.any(segment < 0):
        raise ValueError("segment indices must be non-negative")
    if n_segments is None:
        n_segments = int(np.max(segment)) + 1 if segment.size else 0
    if n_segments < 1 or np.any(segment >= n_segments):
        raise ValueError("n_segments must include every segment index")
    if valid is None:
        keep = np.ones(value.shape[1], dtype=bool)
    else:
        keep = np.asarray(valid, dtype=bool)
        if keep.shape != (value.shape[1],):
            raise ValueError("valid must have shape (query,)")

    finite_query = np.all(
        np.isfinite(value) & np.isfinite(scale) & (scale > 0), axis=0
    )
    keep &= finite_query
    if not np.any(keep):
        raise ValueError("no finite positive-scale queries remain")

    numerator = np.asarray(
        [
            np.sum(value[:, keep & (segment == index)])
            for index in range(n_segments)
        ],
        dtype=np.float64,
    )
    total_scale = float(np.sum(scale[:, keep]))
    if not np.isfinite(total_scale) or total_scale <= 0:
        raise ValueError("total denominator must be finite and positive")

    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=n_segments)),
        dtype=np.float64,
    )
    statistic = signs @ numerator / total_scale
    observed = float(np.sum(numerator) / total_scale)
    tolerance = 16.0 * np.finfo(float).eps * max(1.0, abs(observed))
    two_sided = float(
        np.mean(np.abs(statistic) >= abs(observed) - tolerance)
    )
    positive = float(np.mean(statistic >= observed - tolerance))
    return {
        "status": (
            "exploratory_label_calibration_not_a_randomized_experiment_p_value"
        ),
        "definition": (
            "one normal/tangent swap sign per boundary segment, preserved "
            "across every exposure"
        ),
        "n_exposures": int(value.shape[0]),
        "n_queries": int(keep.sum()),
        "n_segments": int(n_segments),
        "n_exact_labelings": int(signs.shape[0]),
        "observed_statistic": observed,
        "two_sided_tail_fraction": two_sided,
        "positive_tail_fraction": positive,
        "spin_distribution": {
            "minimum": float(np.min(statistic)),
            "q025": float(np.quantile(statistic, 0.025)),
            "median": float(np.median(statistic)),
            "q975": float(np.quantile(statistic, 0.975)),
            "maximum": float(np.max(statistic)),
        },
        "segment_numerator": numerator.tolist(),
        # Retained for exact cohort aggregation; only 2^8 values for `+`.
        "exact_statistics": statistic.tolist(),
    }
