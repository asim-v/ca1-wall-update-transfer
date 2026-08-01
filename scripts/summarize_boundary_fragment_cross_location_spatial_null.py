"""Calibrate spatially matched cross-location controls at the animal level.

The underlying audits retain the observed CA1 maps and compare the reported
same-signed-normal transfer with two empirical spatial controls:

1. source seams at the same 25 cm midpoint lag but with the wrong signed
   normal; and
2. a reflected open seam in the same held-out session, with source distance,
   translation axis, relative bins, occupancy support, and registered cells
   matched to the wall target.

This script does not reshuffle cells, seams, or repeated queries as though they
were independent. It takes the animal-level paired control differences and
enumerates every possible sign assignment. The resulting tail fractions are
descriptive exact sign-flip calibrations for the observed cohort, not
confirmatory population p-values.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RATE_MODES = ("raw_local_rate", "global_rate_demeaned")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spatial-controls",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "boundary_fragment_cross_location_spatial_controls.json"
        ),
    )
    parser.add_argument(
        "--mirror-open",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "boundary_fragment_cross_location_mirror_open.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "boundary_fragment_cross_location_spatial_null.json"
        ),
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_sign_flip(values_by_animal: dict[str, float]) -> dict[str, Any]:
    """Enumerate all animal-level sign assignments for a paired contrast."""

    if not values_by_animal:
        raise ValueError("sign-flip calibration requires at least one animal")
    animals = sorted(values_by_animal)
    values = np.asarray(
        [values_by_animal[animal] for animal in animals],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("sign-flip inputs must be finite")
    observed = float(np.mean(values))
    null_means = np.asarray(
        [
            float(np.mean(values * np.asarray(signs, dtype=np.float64)))
            for signs in product((-1.0, 1.0), repeat=values.size)
        ],
        dtype=np.float64,
    )
    tolerance = 1e-15
    leave_one_out = {
        animal: float(np.mean(np.delete(values, index)))
        for index, animal in enumerate(animals)
    }
    return {
        "animals": int(values.size),
        "positive_animals": int(np.count_nonzero(values > 0)),
        "animal_values": {
            animal: float(value)
            for animal, value in zip(animals, values, strict=True)
        },
        "observed_animal_mean": observed,
        "sign_assignments": int(null_means.size),
        "one_sided_tail_fraction": float(
            np.mean(null_means >= observed - tolerance)
        ),
        "two_sided_tail_fraction": float(
            np.mean(np.abs(null_means) >= abs(observed) - tolerance)
        ),
        "null_mean_minimum": float(np.min(null_means)),
        "null_mean_maximum": float(np.max(null_means)),
        "leave_one_animal_out_means": leave_one_out,
        "leave_one_animal_out_minimum": float(min(leave_one_out.values())),
    }


def _spatial_values(
    spatial: dict[str, Any],
    *,
    mode: str,
    key: str,
) -> dict[str, float]:
    return spatial["cohort_descriptive"][mode][key]["animal_values"]


def _mirror_values(
    mirror: dict[str, Any],
    *,
    mode: str,
) -> dict[str, float]:
    return mirror["cohort_descriptive"]["modes"][mode][
        "wall_minus_mirror_open_correlation_advantage"
    ]["values_by_animal"]


def summarize(
    spatial: dict[str, Any],
    mirror: dict[str, Any],
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in RATE_MODES:
        modes[mode] = {
            "exact_midpoint_wrong_orientation_null": exact_sign_flip(
                _spatial_values(
                    spatial,
                    mode=mode,
                    key="same_all_minus_opposite_all_transfer",
                )
            ),
            "tangential_five_cm_strip_lag_null": exact_sign_flip(
                _spatial_values(
                    spatial,
                    mode=mode,
                    key=(
                        "same_tangential_minus_opposite_tangential_"
                        "transfer"
                    ),
                )
            ),
            "normal_axis_wrong_orientation_null": exact_sign_flip(
                _spatial_values(
                    spatial,
                    mode=mode,
                    key="same_normal_minus_opposite_normal_transfer",
                )
            ),
            "same_session_reflected_open_null": exact_sign_flip(
                _mirror_values(mirror, mode=mode)
            ),
        }
    return {
        "status": "cross_location_spatial_null_calibration",
        "question": (
            "Does the observed wall-related transfer exceed empirical "
            "controls that retain spatially smooth CA1 maps and the observed "
            "source-target lag while breaking the correct wall relation?"
        ),
        "design": {
            "independent_unit": "animal",
            "calibration": (
                "complete enumeration of all 2^N animal-level sign "
                "assignments for each paired spatial-control contrast"
            ),
            "exact_midpoint_wrong_orientation_null": {
                "preserved": [
                    "observed neural maps and place-field smoothness",
                    "held-out target response and registered-cell identities",
                    "25 cm source-target seam-midpoint distance",
                    "query construction and within-animal aggregation",
                ],
                "broken": (
                    "correct signed relation between source and target wall"
                ),
                "limitation": (
                    "pooled normal and tangential strata differ in "
                    "nearest-strip distance and overlap"
                ),
            },
            "tangential_five_cm_strip_lag_null": {
                "preserved": [
                    "observed neural maps and place-field smoothness",
                    "held-out target response and registered-cell identities",
                    "tangential translation axis",
                    "25 cm midpoint distance and 5 cm nearest-strip lag",
                ],
                "broken": (
                    "correct signed relation between source and target wall"
                ),
                "limitation": "only six animals have eligible paired support",
            },
            "same_session_reflected_open_null": {
                "preserved": [
                    "source predictor",
                    "held-out target session",
                    "25 cm source distance and translation axis",
                    "one-to-one relative bins and occupancy support",
                    "registered cells",
                ],
                "broken": "wall versus reflected open target state",
                "limitation": (
                    "only demeaned tangential translations in five animals "
                    "support the strict result"
                ),
            },
            "interpretation": (
                "tail fractions are descriptive cohort calibrations; the "
                "study was exploratory and repeated queries are not independent"
            ),
        },
        "modes": modes,
    }


def main() -> None:
    argument = parse_arguments()
    report = summarize(
        _load(argument.spatial_controls),
        _load(argument.mirror_open),
    )
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["modes"], indent=2))


if __name__ == "__main__":
    main()
