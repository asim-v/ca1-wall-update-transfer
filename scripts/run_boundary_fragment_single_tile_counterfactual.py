"""Test transfer from a training pair differing by one blocked partition.

The target geometry's neural rates are held out.  Training wall and open
profiles must have identical blocked-partition vectors after removing the
focal source partition, and their remaining target-neighbor context must
match the target geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_boundary_component_validation import (  # noqa: E402
    RATE_MODES,
    analyze_animal,
    cohort_summary,
)
from run_boundary_fragment_context_matched import (  # noqa: E402
    _orientation_cohort,
    _support_summary,
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
            / "boundary_fragment_single_tile_counterfactual.json"
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
            match_global_counterfactual=True,
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
            "exploratory_single_tile_counterfactual_cross_exposure_holdout"
        ),
        "question": (
            "Does a focal wall profile learned from two training geometries "
            "that differ only by one blocked partition transfer to the same "
            "wall in a different held-out global geometry?"
        ),
        "design": {
            "target_neural_rates_excluded_from_template_fitting": True,
            "target_wall_label_used_to_select_test_queries": True,
            "target_occupancy_and_registration_used_for_support": True,
            "training_pair_match": (
                "blocked-partition vectors identical after removing the "
                "focal source partition"
            ),
            "target_local_context_match": (
                "all other grid-neighbor states of the accessible target "
                "partition exactly match both training profiles"
            ),
            "training_pair_observed": (
                "u wall versus o open, differing only by blocked partition 5"
            ),
            "target_geometry_excluded_from_training": True,
            "training_and_test_exposures_nonoverlapping": True,
            "fixed_environment_order_confounded": True,
            "wall_and_inaccessible_tile_confounded": True,
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
