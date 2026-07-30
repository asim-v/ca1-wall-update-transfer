"""Quantify reproduction of released maps from raw position and events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ca1_geometry.io import Mat73Animal
from ca1_geometry.rate_maps import (
    authors_current_rate_maps,
    fixed_rate_maps,
    released_rate_maps,
)


def _agreement(candidate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    common = np.isfinite(candidate) & np.isfinite(target)
    first = candidate[common]
    second = target[common]
    if first.size < 2:
        return {
            "finite_values": int(first.size),
            "pearson_r": float("nan"),
            "rmse": float("nan"),
            "max_absolute_error": float("nan"),
        }
    slope, intercept = np.polyfit(first, second, 1)
    return {
        "finite_values": int(first.size),
        "pearson_r": float(np.corrcoef(first, second)[0, 1]),
        "rmse": float(np.sqrt(np.mean(np.square(first - second)))),
        "max_absolute_error": float(np.max(np.abs(first - second))),
        "target_on_candidate_slope": float(slope),
        "target_on_candidate_intercept": float(intercept),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("animal_file", type=Path)
    parser.add_argument("--session", type=int, default=1, help="one-based")
    parser.add_argument("--output", type=Path, required=True)
    argument = parser.parse_args()
    session = argument.session - 1

    with Mat73Animal(argument.animal_file) as animal:
        cell = np.flatnonzero(animal.registered_cells(session))
        position = animal.position(session)
        response = animal.trace(session, cell)
        stored_unsmoothed = animal.stored_rate_maps(
            session, smoothed=False
        )[cell]
        stored_smoothed = animal.stored_rate_maps(
            session, smoothed=True
        )[cell]
        stored_sampling = animal.sampling_map(session)

        current_helper_raw = authors_current_rate_maps(position, response)
        fixed_raw = fixed_rate_maps(position, response)
        released_raw = released_rate_maps(position, response)
        released_smooth = released_rate_maps(
            position,
            response,
            smoothing_sigma_bins=1.0,
        )
        smooth_comparison: dict[str, dict[str, float]] = {}
        sampling_comparison: dict[str, dict[str, float]] = {}
        fixed_smooth_comparison: dict[str, dict[str, float]] = {}
        for sigma in (0.5, 1.0, 1.5, 2.0):
            current_helper_smooth = authors_current_rate_maps(
                position,
                response,
                smoothing_sigma_bins=sigma,
            )
            fixed_smooth = fixed_rate_maps(
                position,
                response,
                smoothing_sigma_bins=sigma,
            )
            label = f"{sigma:g}_bins"
            smooth_comparison[label] = _agreement(
                current_helper_smooth.rate, stored_smoothed
            )
            sampling_comparison[label] = _agreement(
                current_helper_smooth.occupancy, stored_sampling
            )
            fixed_smooth_comparison[label] = _agreement(
                fixed_smooth.rate / 30.0, stored_smoothed
            )

        result = {
            "animal": argument.animal_file.stem.replace(".complete", ""),
            "session": argument.session,
            "environment": animal.environment(session),
            "registered_cells": int(cell.size),
            "frames": int(position.shape[0]),
            "stored_unsmoothed_vs_current_public_helper": _agreement(
                current_helper_raw.rate, stored_unsmoothed
            ),
            "stored_unsmoothed_vs_fixed_edges": _agreement(
                fixed_raw.rate, stored_unsmoothed
            ),
            "stored_unsmoothed_vs_fixed_event_probability": _agreement(
                fixed_raw.rate / 30.0, stored_unsmoothed
            ),
            "stored_sampling_vs_fixed_dwell_seconds": _agreement(
                fixed_raw.occupancy * position.shape[0] / 30.0,
                stored_sampling,
            ),
            "stored_unsmoothed_vs_released_reproducer": _agreement(
                released_raw.rate, stored_unsmoothed
            ),
            "stored_sampling_vs_released_reproducer": _agreement(
                released_raw.occupancy, stored_sampling
            ),
            "stored_smoothed_vs_released_reproducer": _agreement(
                released_smooth.rate, stored_smoothed
            ),
            "stored_smoothed_vs_current_public_helper_by_sigma": (
                smooth_comparison
            ),
            "stored_smoothed_vs_fixed_event_probability_by_sigma": (
                fixed_smooth_comparison
            ),
            "stored_sampling_vs_current_public_helper_by_sigma": (
                sampling_comparison
            ),
            "conventions": {
                "current_helper_edges": "observed session maxima plus 1e-5",
                "fixed_edges": "0 to 75 cm in 5 cm bins",
                "released_upper_edge": (
                    "temporary bin 15, smoothed before 15 by 15 crop"
                ),
                "released_rate_unit": "event probability per frame",
                "released_sampling_unit": "seconds",
                "released_smoothing": (
                    "sigma 1 bin (5 cm), radius 2 bins, constant-zero padding"
                ),
                "array_orientation": "cell, y, x",
            },
        }

    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
