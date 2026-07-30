"""Cross-validated metric tensors and boundary anisotropy summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _normalized_neuron_weight(
    n_neuron: int, neuron_weight: ArrayLike | None
) -> FloatArray:
    if neuron_weight is None:
        return np.full(n_neuron, 1.0 / n_neuron, dtype=np.float64)
    weight = np.asarray(neuron_weight, dtype=np.float64)
    if weight.shape != (n_neuron,):
        raise ValueError("neuron_weight must have shape (neuron,)")
    if not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError("neuron_weight must be finite and non-negative")
    total = float(weight.sum())
    if total <= 0:
        raise ValueError("neuron_weight must have positive sum")
    return weight / total


def pooled_metric(
    jacobian: ArrayLike, neuron_weight: ArrayLike | None = None
) -> FloatArray:
    """Return the ordinary positive-semidefinite pullback metric.

    This estimator is appropriate for visualization. Its derivative-noise bias
    makes it unsuitable as the confirmatory condition contrast.
    """

    j = np.asarray(jacobian, dtype=np.float64)
    if j.ndim != 3 or j.shape[-1] != 2:
        raise ValueError("jacobian must have shape (query, neuron, 2)")
    weight = _normalized_neuron_weight(j.shape[1], neuron_weight)
    return np.einsum("qni,n,qnj->qij", j, weight, j)


def cross_metric(
    jacobians: ArrayLike,
    neuron_weight: ArrayLike | None = None,
) -> FloatArray:
    """Estimate a derivative-noise-unbiased metric from independent folds.

    ``jacobians`` has shape ``(fold, query, neuron, 2)``. All ordered
    cross-fold products are averaged, which makes the result symmetric. Like a
    crossnobis estimator, finite-sample tensors can be locally indefinite and
    must not be projected onto the PSD cone for inference.
    """

    j = np.asarray(jacobians, dtype=np.float64)
    if j.ndim != 4 or j.shape[-1] != 2:
        raise ValueError(
            "jacobians must have shape (fold, query, neuron, 2)"
        )
    n_fold, _, n_neuron, _ = j.shape
    if n_fold < 2:
        raise ValueError("at least two independent folds are required")
    weight = _normalized_neuron_weight(n_neuron, neuron_weight)

    result = np.zeros((j.shape[1], 2, 2), dtype=np.float64)
    n_pair = 0
    for first in range(n_fold):
        for second in range(first + 1, n_fold):
            forward = np.einsum(
                "qni,n,qnj->qij", j[first], weight, j[second]
            )
            result += 0.5 * (forward + np.swapaxes(forward, -1, -2))
            n_pair += 1
    return result / n_pair


def _broadcast_direction(direction: ArrayLike, n_query: int) -> FloatArray:
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape == (2,):
        vector = np.broadcast_to(vector, (n_query, 2)).copy()
    if vector.shape != (n_query, 2):
        raise ValueError("direction must have shape (2,) or (query, 2)")
    length = np.linalg.norm(vector, axis=1)
    if np.any(~np.isfinite(length)) or np.any(length <= 0):
        raise ValueError("directions must be finite and nonzero")
    return vector / length[:, None]


def directional_metric(metric: ArrayLike, direction: ArrayLike) -> FloatArray:
    """Evaluate ``v.T @ g @ v`` at each query."""

    g = np.asarray(metric, dtype=np.float64)
    if g.ndim != 3 or g.shape[1:] != (2, 2):
        raise ValueError("metric must have shape (query, 2, 2)")
    vector = _broadcast_direction(direction, g.shape[0])
    return np.einsum("qi,qij,qj->q", vector, g, vector)


def anisotropy_components(
    metric: ArrayLike,
    reference_metric: ArrayLike,
    normal: ArrayLike,
    tangent: ArrayLike | None = None,
) -> dict[str, FloatArray]:
    """Return normal magnification and normal-minus-tangent contrast."""

    g = np.asarray(metric, dtype=np.float64)
    reference = np.asarray(reference_metric, dtype=np.float64)
    if g.shape != reference.shape:
        raise ValueError("metric and reference_metric must have equal shape")
    normal_vector = _broadcast_direction(normal, g.shape[0])
    if tangent is None:
        tangent_vector = np.column_stack(
            (-normal_vector[:, 1], normal_vector[:, 0])
        )
    else:
        tangent_vector = _broadcast_direction(tangent, g.shape[0])
        dot = np.abs(np.sum(normal_vector * tangent_vector, axis=1))
        if np.any(dot > 1e-6):
            raise ValueError("normal and tangent must be orthogonal")

    delta = g - reference
    delta_normal = directional_metric(delta, normal_vector)
    delta_tangent = directional_metric(delta, tangent_vector)
    trace_reference = np.trace(reference, axis1=1, axis2=2)
    return {
        "delta_normal": delta_normal,
        "delta_tangent": delta_tangent,
        "contrast": delta_normal - delta_tangent,
        "trace_reference": trace_reference,
    }


@dataclass(frozen=True)
class AnisotropyProfile:
    """Strip-aggregated boundary anisotropy results."""

    left: FloatArray
    right: FloatArray
    center: FloatArray
    anisotropy: FloatArray
    normal_magnification: FloatArray
    tangential_change: FloatArray
    n_query: NDArray[np.int64]


def anisotropy_profile(
    metric: ArrayLike,
    reference_metric: ArrayLike,
    normal: ArrayLike,
    distance: ArrayLike,
    bin_edges: ArrayLike,
    *,
    tangent: ArrayLike | None = None,
    area_weight: ArrayLike | None = None,
    valid: ArrayLike | None = None,
    denominator_metric: ArrayLike | None = None,
    denominator_floor: float = 1e-12,
) -> AnisotropyProfile:
    """Aggregate anisotropy as a ratio of spatial sums within distance strips.

    A separately estimated positive reference scale can be supplied through
    ``denominator_metric``. The contrast numerator always uses the two
    cross-validated metrics passed as the first arguments.
    """

    g = np.asarray(metric, dtype=np.float64)
    distance_array = np.asarray(distance, dtype=np.float64)
    edge = np.asarray(bin_edges, dtype=np.float64)
    if distance_array.shape != (g.shape[0],):
        raise ValueError("distance must have shape (query,)")
    if edge.ndim != 1 or edge.size < 2 or np.any(np.diff(edge) <= 0):
        raise ValueError("bin_edges must be a strictly increasing vector")

    if area_weight is None:
        area = np.ones(g.shape[0], dtype=np.float64)
    else:
        area = np.asarray(area_weight, dtype=np.float64)
        if area.shape != (g.shape[0],):
            raise ValueError("area_weight must have shape (query,)")
    if valid is None:
        keep = np.ones(g.shape[0], dtype=bool)
    else:
        keep = np.asarray(valid, dtype=bool)
        if keep.shape != (g.shape[0],):
            raise ValueError("valid must have shape (query,)")

    component = anisotropy_components(
        g, reference_metric, normal, tangent=tangent
    )
    if denominator_metric is not None:
        denominator_value = np.asarray(
            denominator_metric, dtype=np.float64
        )
        if denominator_value.shape != g.shape:
            raise ValueError("denominator_metric must match metric shape")
        component["trace_reference"] = np.trace(
            denominator_value, axis1=1, axis2=2
        )
    finite = np.ones(g.shape[0], dtype=bool)
    for value in component.values():
        finite &= np.isfinite(value)
    keep &= finite & np.isfinite(distance_array) & np.isfinite(area)
    keep &= area > 0

    n_bin = edge.size - 1
    anisotropy = np.full(n_bin, np.nan)
    normal_change = np.full(n_bin, np.nan)
    tangent_change = np.full(n_bin, np.nan)
    count = np.zeros(n_bin, dtype=np.int64)

    for index in range(n_bin):
        member = (
            keep
            & (distance_array >= edge[index])
            & (distance_array < edge[index + 1])
        )
        count[index] = int(member.sum())
        if count[index] == 0:
            continue
        denominator = np.sum(
            area[member] * component["trace_reference"][member]
        )
        if not np.isfinite(denominator) or denominator <= denominator_floor:
            continue
        normal_change[index] = (
            np.sum(area[member] * component["delta_normal"][member])
            / denominator
        )
        tangent_change[index] = (
            np.sum(area[member] * component["delta_tangent"][member])
            / denominator
        )
        anisotropy[index] = normal_change[index] - tangent_change[index]

    return AnisotropyProfile(
        left=edge[:-1],
        right=edge[1:],
        center=0.5 * (edge[:-1] + edge[1:]),
        anisotropy=anisotropy,
        normal_magnification=normal_change,
        tangential_change=tangent_change,
        n_query=count,
    )
