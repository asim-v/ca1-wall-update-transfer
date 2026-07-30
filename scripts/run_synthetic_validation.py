"""Run an end-to-end synthetic sign-recovery demonstration."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ca1_geometry.local_linear import LocalMapConfig, fit_local_linear
from ca1_geometry.metrics import anisotropy_profile, cross_metric
from ca1_geometry.synthetic import (
    boundary_normal_warp,
    gaussian_place_code,
    sample_events,
)


def _fit_folds(
    position: np.ndarray,
    response: np.ndarray,
    query: np.ndarray,
    fold: np.ndarray,
    config: LocalMapConfig,
) -> tuple[np.ndarray, np.ndarray]:
    maps = [
        fit_local_linear(
            position[fold == index],
            response[fold == index],
            query,
            config,
        )
        for index in range(int(fold.max()) + 1)
    ]
    jacobian = np.stack([item.jacobian for item in maps])
    valid = np.logical_and.reduce([item.valid for item in maps])
    return jacobian, valid


def main() -> None:
    rng = np.random.default_rng(20260728)
    n_sample = 12_000
    n_neuron = 100
    n_fold = 4

    # Both conditions occupy the same physical half-arena, but their sampling
    # distributions deliberately differ.
    baseline_position = np.column_stack(
        (rng.beta(1.6, 1.6, n_sample), rng.uniform(-1.0, 1.0, n_sample))
    )
    condition_position = np.column_stack(
        (rng.beta(1.2, 2.0, n_sample), rng.uniform(-1.0, 1.0, n_sample))
    )

    side = int(np.sqrt(n_neuron))
    center_x, center_y = np.meshgrid(
        np.linspace(-0.15, 1.35, side),
        np.linspace(-1.15, 1.15, side),
    )
    centers = np.column_stack((center_x.ravel(), center_y.ravel()))

    baseline_probability = gaussian_place_code(
        baseline_position, centers, width=0.18
    )
    warped_position = boundary_normal_warp(
        condition_position,
        boundary_x=0.0,
        amplitude=0.50,
        decay=0.30,
    )
    condition_probability = gaussian_place_code(
        warped_position, centers, width=0.18
    )
    null_probability = gaussian_place_code(
        condition_position, centers, width=0.18
    )
    baseline_response = sample_events(baseline_probability, rng)
    condition_response = sample_events(condition_probability, rng)
    null_response = sample_events(null_probability, rng)

    # Contiguous independent folds emulate the confirmatory blocked analysis.
    baseline_fold = np.repeat(
        np.arange(n_fold), np.ceil(n_sample / n_fold)
    )[:n_sample]
    condition_fold = baseline_fold.copy()

    query_x, query_y = np.meshgrid(
        np.linspace(0.08, 0.92, 10), np.linspace(-0.80, 0.80, 11)
    )
    query = np.column_stack((query_x.ravel(), query_y.ravel()))
    config = LocalMapConfig(
        bandwidth=0.22,
        min_effective_samples=35,
        min_design_eigenratio=0.02,
    )

    baseline_jacobian, baseline_valid = _fit_folds(
        baseline_position,
        baseline_response,
        query,
        baseline_fold,
        config,
    )
    condition_jacobian, condition_valid = _fit_folds(
        condition_position,
        condition_response,
        query,
        condition_fold,
        config,
    )
    null_jacobian, null_valid = _fit_folds(
        condition_position,
        null_response,
        query,
        condition_fold,
        config,
    )
    baseline_metric = cross_metric(baseline_jacobian)
    condition_metric = cross_metric(condition_jacobian)
    null_metric = cross_metric(null_jacobian)
    valid = baseline_valid & condition_valid
    null_joint_valid = baseline_valid & null_valid

    edges = np.linspace(0.0, 1.0, 6)
    profile = anisotropy_profile(
        condition_metric,
        baseline_metric,
        normal=np.array([1.0, 0.0]),
        distance=query[:, 0],
        bin_edges=edges,
        valid=valid,
    )
    null_profile = anisotropy_profile(
        null_metric,
        baseline_metric,
        normal=np.array([1.0, 0.0]),
        distance=query[:, 0],
        bin_edges=edges,
        valid=null_joint_valid,
    )

    output = Path("results/diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": 20260728,
        "n_sample_per_condition": n_sample,
        "n_neuron": n_neuron,
        "valid_query_fraction": float(valid.mean()),
        "near_anisotropy": float(profile.anisotropy[0]),
        "far_anisotropy": float(profile.anisotropy[-1]),
        "unequal_occupancy_null_near_anisotropy": float(
            null_profile.anisotropy[0]
        ),
        "sign_recovered": bool(
            profile.anisotropy[0] > 0
            and profile.normal_magnification[0] > 0
        ),
        "null_is_smaller": bool(
            abs(null_profile.anisotropy[0])
            < 0.25 * abs(profile.anisotropy[0])
        ),
    }
    (output / "synthetic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.axhline(0.0, color="0.5", linewidth=1)
    axis.plot(
        profile.center,
        profile.anisotropy,
        marker="o",
        label="normal − tangential",
    )
    axis.plot(
        profile.center,
        profile.normal_magnification,
        marker="s",
        label="normal magnification",
    )
    axis.plot(
        null_profile.center,
        null_profile.anisotropy,
        color="0.35",
        linestyle="--",
        marker=".",
        label="unequal-occupancy null",
    )
    axis.set(
        xlabel="Distance from synthetic boundary",
        ylabel="Cross-validated metric change / baseline trace",
        title="Known boundary-normal warp",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "synthetic_anisotropy.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
