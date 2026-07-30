"""Test a focal seam while matching the rest of its local boundary context.

This is a stricter version of the target-rate-held-out exact-location
analysis.  Training wall and open environments must have exactly the same
blocked/accessibility pattern as the target around every other grid neighbor
of the accessible target partition.  The target geometry's neural rates
remain excluded from profile fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_boundary_component_validation import (  # noqa: E402
    RATE_MODES,
    analyze_animal,
    cohort_summary,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "boundary_fragment_context_matched.json"
        ),
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-bins", type=int, default=6)
    parser.add_argument("--minimum-cells", type=int, default=20)
    parser.add_argument(
        "--depths-cm",
        type=float,
        nargs="+",
        default=(2.5, 7.5, 12.5),
    )
    return parser.parse_args()


def _orientation_cohort(
    animals: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for orientation in ("vertical_wall", "horizontal_wall"):
        animal_values = {
            animal["animal"]: (
                animal["summaries"][mode]["orientation"][orientation][
                    "mean_wall_minus_open"
                ]
            )
            for animal in animals
        }
        values = np.asarray(
            [
                value
                for value in animal_values.values()
                if value is not None and np.isfinite(value)
            ],
            dtype=np.float64,
        )
        output[orientation] = {
            "animals": int(values.size),
            "positive_animals": int(
                np.count_nonzero(values > 0)
            ),
            "animal_mean_wall_minus_open": (
                float(np.mean(values)) if values.size else None
            ),
            "animal_values": animal_values,
        }
    return output


def _support_summary(animals: list[dict[str, Any]]) -> dict[str, Any]:
    records = [
        record
        for animal in animals
        for record in animal["records"]
    ]
    return {
        "eligible_local_predictions": len(records),
        "animal_values": {
            animal["animal"]: len(animal["records"])
            for animal in animals
        },
        "median_matching_wall_environments": float(
            np.median(
                [
                    len(record["matching_wall_environments"])
                    for record in records
                ]
            )
        ),
        "median_matching_open_environments": float(
            np.median(
                [
                    len(record["matching_open_environments"])
                    for record in records
                ]
            )
        ),
        "target_environments": sorted(
            {record["target_environment"] for record in records}
        ),
    }


def main() -> None:
    argument = parse_arguments()
    paths = sorted(
        argument.data_dir.glob("QLAK-CA1-*.complete.mat")
    )
    if not paths:
        raise FileNotFoundError(
            f"no complete animal files found beneath {argument.data_dir}"
        )

    animals: list[dict[str, Any]] = []
    for path in paths:
        result = analyze_animal(
            path,
            minimum_seconds=argument.minimum_seconds,
            minimum_bins=argument.minimum_bins,
            minimum_cells=argument.minimum_cells,
            match_nonfocal_context=True,
            strip_depths_cm=tuple(argument.depths_cm),
        )
        animals.append(result)
        print(
            result["animal"],
            {
                mode: round(
                    result["summaries"][mode]["wall_minus_open_mean"],
                    4,
                )
                for mode in RATE_MODES
            },
            len(result["records"]),
        )

    cohort = cohort_summary(animals)
    for mode in RATE_MODES:
        cohort[mode]["orientation"] = _orientation_cohort(
            animals,
            mode=mode,
        )

    report = {
        "status": (
            "exploratory_context_matched_cross_exposure_holdout"
        ),
        "question": (
            "After exactly matching every other grid-neighbor state around "
            "the accessible target partition, does the focal exact-seam wall "
            "profile better match the held-out local CA1 vector than the "
            "focal-seam open profile?"
        ),
        "design": {
            "target_neural_rates_excluded_from_template_fitting": True,
            "target_wall_label_used_to_select_test_queries": True,
            "target_occupancy_and_registration_used_for_support": True,
            "context_match": (
                "exact blocked/accessibility state of every other internal "
                "grid neighbor of the accessible target partition; focal "
                "source partition excluded"
            ),
            "focal_location_and_orientation_fixed": True,
            "global_shape_matched": False,
            "inaccessible_area_elsewhere_matched": False,
            "training_and_test_exposures_nonoverlapping": True,
            "training_baseline": "square_before_training cycle",
            "test_baseline": "square after test cycle",
            "predicted_object": (
                "one square-residual local strip rate per registered cell"
            ),
            "rate_modes": list(RATE_MODES),
            "inferential_unit": "animal",
        },
        "settings": {
            "minimum_seconds_per_common_bin": argument.minimum_seconds,
            "minimum_common_bins": argument.minimum_bins,
            "minimum_common_cells": argument.minimum_cells,
            "strip_depths_cm": list(argument.depths_cm),
        },
        "support": _support_summary(animals),
        "cohort": cohort,
        "animals": animals,
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"support": report["support"], "cohort": cohort}, indent=2))


if __name__ == "__main__":
    main()
