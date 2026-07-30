"""Replay known geometry on QLAK-CA1-51's measured square/o trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ca1_geometry.arena import (
    introduced_boundaries,
    positions_on_accessible_support,
    segment_boundary_queries,
)
from ca1_geometry.io import Mat73Animal
from ca1_geometry.local_linear import LocalMapConfig
from ca1_geometry.metrics import anisotropy_profile
from ca1_geometry.pilot import (
    balanced_block_folds,
    estimate_session_metric,
    occupancy_balance_weights,
    residual_directional_reliability,
)
from ca1_geometry.synthetic import (
    center_occlusion_normal_warp,
    gaussian_place_code,
)


SQUARE_SESSION = 0
CENTER_OCCLUSION_SESSION = 1
COMMON_BLOCKED = (4,)
DISTANCES_CM = np.array([2.5, 7.5, 12.5])
DISTANCE_EDGES_CM = np.array([0.0, 5.0, 10.0, 15.0])
BANDWIDTHS_CM = (5.0, 7.5, 10.0)
WARP_AMPLITUDE = 0.55
WARP_DECAY_CM = 8.0
PLACE_FIELD_WIDTH_CM = 9.0
N_FOLD = 4


def _finite_list(value: np.ndarray) -> list[float | None]:
    return [
        float(item) if np.isfinite(item) else None
        for item in np.asarray(value).ravel()
    ]


def _reliability_dict(value: Any) -> dict[str, float | int | None]:
    return {
        "normal": float(value.normal) if np.isfinite(value.normal) else None,
        "tangent": (
            float(value.tangent) if np.isfinite(value.tangent) else None
        ),
        "contrast": (
            float(value.contrast) if np.isfinite(value.contrast) else None
        ),
        "n_query": int(value.n_query),
    }


def _profile_dict(profile: Any) -> dict[str, Any]:
    return {
        "distance_cm": _finite_list(profile.center),
        "anisotropy": _finite_list(profile.anisotropy),
        "normal_magnification": _finite_list(
            profile.normal_magnification
        ),
        "tangential_change": _finite_list(profile.tangential_change),
        "n_query": [int(item) for item in profile.n_query],
    }


def _place_centers() -> np.ndarray:
    """A padded lattice makes the local unwarped metric nearly stationary."""

    coordinate = np.linspace(-15.0, 90.0, 12)
    center_x, center_y = np.meshgrid(coordinate, coordinate)
    return np.column_stack((center_x.ravel(), center_y.ravel()))


def run_replay(
    data_file: Path,
    *,
    fold_scheme: str = "balanced_blocks",
    block_seconds: int = 60,
    guard_seconds: int = 2,
) -> dict[str, Any]:
    """Run the trajectory-matched warp and no-warp controls."""

    with Mat73Animal(data_file) as animal:
        observed = {
            "square": animal.environment(SQUARE_SESSION),
            "center_occlusion": animal.environment(
                CENTER_OCCLUSION_SESSION
            ),
            "center_blocked": animal.blocked(CENTER_OCCLUSION_SESSION),
        }
        if observed != {
            "square": "square",
            "center_occlusion": "o",
            "center_blocked": COMMON_BLOCKED,
        }:
            raise ValueError(f"unexpected first sequence: {observed}")

        square_position = animal.position(SQUARE_SESSION)
        occlusion_position = animal.position(CENTER_OCCLUSION_SESSION)

    square_keep = positions_on_accessible_support(
        square_position, COMMON_BLOCKED
    )
    occlusion_keep = positions_on_accessible_support(
        occlusion_position, COMMON_BLOCKED
    )
    weights = occupancy_balance_weights(
        [square_position, occlusion_position],
        [square_keep, occlusion_keep],
    )
    if fold_scheme == "balanced_blocks":
        fold_assignment = [
            balanced_block_folds(
                position,
                keep,
                n_fold=N_FOLD,
                block_frames=block_seconds * 30,
                guard_frames=guard_seconds * 30,
            )
            for position, keep in (
                (square_position, square_keep),
                (occlusion_position, occlusion_keep),
            )
        ]
    elif fold_scheme == "contiguous_quarters":
        fold_assignment = [None, None]
    else:
        raise ValueError(f"unknown fold scheme: {fold_scheme}")

    centers = _place_centers()
    square_response = gaussian_place_code(
        square_position,
        centers,
        PLACE_FIELD_WIDTH_CM,
        peak_probability=0.30,
        baseline_probability=0.01,
    )
    flat_response = gaussian_place_code(
        occlusion_position,
        centers,
        PLACE_FIELD_WIDTH_CM,
        peak_probability=0.30,
        baseline_probability=0.01,
    )
    warped_position = center_occlusion_normal_warp(
        occlusion_position,
        amplitude=WARP_AMPLITUDE,
        decay=WARP_DECAY_CM,
    )
    warped_response = gaussian_place_code(
        warped_position,
        centers,
        PLACE_FIELD_WIDTH_CM,
        peak_probability=0.30,
        baseline_probability=0.01,
    )

    queries = segment_boundary_queries(
        introduced_boundaries(COMMON_BLOCKED),
        DISTANCES_CM,
        # At h=10 cm these anchors keep every query kernel outside the
        # segment endpoints, where a unique local wall normal breaks down.
        tangential_fractions=np.array([0.40, 0.50, 0.60]),
    )
    bandwidth_results: list[dict[str, Any]] = []
    plotted_profiles: list[tuple[float, Any, Any]] = []
    for bandwidth in BANDWIDTHS_CM:
        config = LocalMapConfig(
            bandwidth=bandwidth,
            min_effective_samples=40.0,
            min_design_eigenratio=0.01,
        )
        square_metric = estimate_session_metric(
            square_position,
            square_response,
            queries.position,
            common_blocked=COMMON_BLOCKED,
            config=config,
            n_fold=N_FOLD,
            sample_weight=weights[0],
            fold_assignment=fold_assignment[0],
        )
        warp_metric = estimate_session_metric(
            occlusion_position,
            warped_response,
            queries.position,
            common_blocked=COMMON_BLOCKED,
            config=config,
            n_fold=N_FOLD,
            sample_weight=weights[1],
            fold_assignment=fold_assignment[1],
        )
        flat_metric = estimate_session_metric(
            occlusion_position,
            flat_response,
            queries.position,
            common_blocked=COMMON_BLOCKED,
            config=config,
            n_fold=N_FOLD,
            sample_weight=weights[1],
            fold_assignment=fold_assignment[1],
        )

        warp_valid = square_metric.valid & warp_metric.valid
        flat_valid = square_metric.valid & flat_metric.valid
        warp_profile = anisotropy_profile(
            warp_metric.metric,
            square_metric.metric,
            queries.normal,
            queries.distance,
            DISTANCE_EDGES_CM,
            tangent=queries.tangent,
            valid=warp_valid,
            denominator_metric=square_metric.pooled,
        )
        flat_profile = anisotropy_profile(
            flat_metric.metric,
            square_metric.metric,
            queries.normal,
            queries.distance,
            DISTANCE_EDGES_CM,
            tangent=queries.tangent,
            valid=flat_valid,
            denominator_metric=square_metric.pooled,
        )
        warp_reliability = residual_directional_reliability(
            warp_metric.jacobians,
            [square_metric.jacobians],
            queries.normal,
            queries.tangent,
            valid=warp_valid,
        )
        flat_reliability = residual_directional_reliability(
            flat_metric.jacobians,
            [square_metric.jacobians],
            queries.normal,
            queries.tangent,
            valid=flat_valid,
        )
        supported = np.flatnonzero(
            (warp_profile.n_query > 0)
            & (flat_profile.n_query > 0)
            & np.isfinite(warp_profile.anisotropy)
            & np.isfinite(warp_profile.normal_magnification)
            & np.isfinite(flat_profile.anisotropy)
        )
        closest_index = int(supported[0]) if supported.size else None
        if closest_index is None:
            closest_distance = None
            closest_warp = float("nan")
            closest_normal = float("nan")
            closest_flat = float("nan")
        else:
            closest_distance = float(warp_profile.center[closest_index])
            closest_warp = float(warp_profile.anisotropy[closest_index])
            closest_normal = float(
                warp_profile.normal_magnification[closest_index]
            )
            closest_flat = float(flat_profile.anisotropy[closest_index])
        bandwidth_results.append(
            {
                "bandwidth_cm": bandwidth,
                "valid_queries_warp": int(warp_valid.sum()),
                "valid_queries_flat": int(flat_valid.sum()),
                "warp": _profile_dict(warp_profile),
                "flat_null": _profile_dict(flat_profile),
                "warp_residual_split_pair_reliability": _reliability_dict(
                    warp_reliability
                ),
                "flat_residual_split_pair_reliability": _reliability_dict(
                    flat_reliability
                ),
                "closest_supported_distance_cm": closest_distance,
                "boundary_adjacent_strip_supported": bool(
                    warp_profile.n_query[0] > 0
                ),
                "closest_supported_sign_recovered": bool(
                    closest_warp > 0.0 and closest_normal > 0.0
                ),
                "closest_supported_flat_smaller_than_warp": bool(
                    abs(closest_flat) < abs(closest_warp)
                ),
                "minimum_effective_n": {
                    "square": float(
                        np.min(
                            square_metric.effective_n[
                                :, square_metric.valid
                            ]
                        )
                    ),
                    "warp": float(
                        np.min(
                            warp_metric.effective_n[:, warp_metric.valid]
                        )
                    ),
                    "flat": float(
                        np.min(
                            flat_metric.effective_n[:, flat_metric.valid]
                        )
                    ),
                },
            }
        )
        plotted_profiles.append((bandwidth, warp_profile, flat_profile))

    summary: dict[str, Any] = {
        "data_file": data_file.name,
        "sessions_zero_based": {
            "square": SQUARE_SESSION,
            "center_occlusion": CENTER_OCCLUSION_SESSION,
        },
        "common_blocked_partitions": list(COMMON_BLOCKED),
        "response_mode": "exact Gaussian place-field probability",
        "n_synthetic_neurons": int(centers.shape[0]),
        "n_folds": N_FOLD,
        "fold_scheme": fold_scheme,
        "block_seconds": (
            block_seconds if fold_scheme == "balanced_blocks" else None
        ),
        "guard_seconds_each_block_edge": (
            guard_seconds if fold_scheme == "balanced_blocks" else None
        ),
        "warp": {
            "amplitude": WARP_AMPLITUDE,
            "decay_cm": WARP_DECAY_CM,
            "normal_derivative_at_boundary": 1.0 + WARP_AMPLITUDE,
        },
        "frames": {
            "square_total": int(square_position.shape[0]),
            "center_occlusion_total": int(occlusion_position.shape[0]),
            "square_common_support": int(square_keep.sum()),
            "center_occlusion_common_support": int(occlusion_keep.sum()),
        },
        "weighted_frame_totals": {
            "square": float(weights[0].sum()),
            "center_occlusion": float(weights[1].sum()),
        },
        "query_count": int(queries.position.shape[0]),
        "bandwidth_results": bandwidth_results,
        "sign_recovered_at_closest_supported_strip_all_bandwidths": all(
            item["closest_supported_sign_recovered"]
            for item in bandwidth_results
        ),
        "flat_null_smaller_at_closest_supported_strip_all_bandwidths": all(
            item["closest_supported_flat_smaller_than_warp"]
            for item in bandwidth_results
        ),
        "boundary_adjacent_strip_supported_all_bandwidths": all(
            item["boundary_adjacent_strip_supported"]
            for item in bandwidth_results
        ),
    }
    summary["_plotted_profiles"] = plotted_profiles
    return summary


def _save_figure(
    profiles: list[tuple[float, Any, Any]], output_path: Path
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(9.6, 4.1),
        sharey=True,
        layout="constrained",
    )
    for bandwidth, warp, flat in profiles:
        label = f"h = {bandwidth:g} cm"
        axes[0].plot(
            warp.center,
            warp.anisotropy,
            marker="o",
            label=label,
        )
        axes[1].plot(
            flat.center,
            flat.anisotropy,
            marker="o",
            label=label,
        )
    for axis in axes:
        axis.axhline(0.0, color="0.5", linewidth=1)
        axis.set_xlabel("Distance from center boundary (cm)")
    axes[0].set_ylabel("Normalized normal - tangential metric change")
    axes[0].set_title("Known normal warp")
    axes[1].set_title("No-warp trajectory null")
    axes[0].legend(frameon=False)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_file",
        nargs="?",
        type=Path,
        default=Path("data/raw/QLAK-CA1-51.complete.mat"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/diagnostics"),
    )
    parser.add_argument(
        "--fold-scheme",
        choices=("balanced_blocks", "contiguous_quarters"),
        default="balanced_blocks",
    )
    parser.add_argument("--block-seconds", type=int, default=60)
    parser.add_argument("--guard-seconds", type=int, default=2)
    args = parser.parse_args()

    summary = run_replay(
        args.data_file,
        fold_scheme=args.fold_scheme,
        block_seconds=args.block_seconds,
        guard_seconds=args.guard_seconds,
    )
    profiles = summary.pop("_plotted_profiles")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"QLAK-CA1-51_trajectory_replay_{args.fold_scheme}"
    if args.fold_scheme == "balanced_blocks":
        stem += f"_b{args.block_seconds}_g{args.guard_seconds}"
    json_path = args.output_dir / f"{stem}.json"
    figure_path = args.output_dir / f"{stem}.png"
    json_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _save_figure(profiles, figure_path)
    print(json.dumps(summary, indent=2))

    if not summary[
        "sign_recovered_at_closest_supported_strip_all_bandwidths"
    ]:
        raise SystemExit("known normal-warp sign was not recovered")
    if not summary[
        "flat_null_smaller_at_closest_supported_strip_all_bandwidths"
    ]:
        raise SystemExit("trajectory null exceeded known warp")


if __name__ == "__main__":
    main()
