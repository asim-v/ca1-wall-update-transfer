"""Re-aggregate the held-out local wall prediction records.

This script does not refit a model or create an independent validation set.
It asks whether the animal-level wall-minus-open advantage changes materially
when records are equalized at several plausible grouping levels.  Mice remain
the biological units in every cohort summary.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Hashable


ROOT = Path(__file__).resolve().parents[1]
RATE_MODES = ("raw_local_rate", "global_rate_demeaned")
RECONSTRUCTION_TOLERANCE = 1e-12


def _prediction_record_key(
    record: dict[str, Any],
    record_index: int,
) -> tuple[str, int]:
    del record
    return ("record", record_index)


def _exposure_pair_key(
    record: dict[str, Any],
    record_index: int,
) -> tuple[str, int, int]:
    del record_index
    return (
        "exposure_pair",
        int(record["training_exposure"]),
        int(record["test_exposure"]),
    )


def _target_environment_key(
    record: dict[str, Any],
    record_index: int,
) -> tuple[str, str]:
    del record_index
    return ("target_environment", str(record["target_environment"]))


def _exact_oriented_seam_key(
    record: dict[str, Any],
    record_index: int,
) -> tuple[str, str, int, int]:
    del record_index
    seam = record["seam"]
    if not isinstance(seam, list) or len(seam) != 2:
        raise ValueError(f"expected a two-entry ordered seam, got {seam!r}")
    return (
        "exact_oriented_seam",
        str(record["orientation"]),
        int(seam[0]),
        int(seam[1]),
    )


GROUPERS: dict[
    str,
    Callable[[dict[str, Any], int], Hashable],
] = {
    "prediction_record_equal": _prediction_record_key,
    "exposure_pair_equal": _exposure_pair_key,
    "target_environment_equal": _target_environment_key,
    "exact_oriented_seam_equal": _exact_oriented_seam_key,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "boundary_component_validation.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "boundary_fragment_aggregation_sensitivity.json"
        ),
    )
    return parser.parse_args()


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _direction(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _support_summary(counts: list[int]) -> dict[str, Any]:
    if not counts:
        return {
            "minimum": None,
            "median": None,
            "maximum": None,
            "counts_sorted": [],
        }
    return {
        "minimum": min(counts),
        "median": float(median(counts)),
        "maximum": max(counts),
        "counts_sorted": sorted(counts),
    }


def aggregate_records(
    records: list[dict[str, Any]],
    *,
    mode: str,
    scheme: str,
) -> dict[str, Any]:
    """Return an equal-group animal estimate and transparent support counts."""

    try:
        grouper = GROUPERS[scheme]
    except KeyError as error:
        raise ValueError(f"unknown aggregation scheme: {scheme}") from error

    total_by_group: dict[Hashable, int] = defaultdict(int)
    finite_by_group: dict[Hashable, list[float]] = defaultdict(list)
    for record_index, record in enumerate(records):
        key = grouper(record, record_index)
        total_by_group[key] += 1
        correlations = record.get("correlations")
        if not isinstance(correlations, dict):
            continue
        mode_result = correlations.get(mode)
        if not isinstance(mode_result, dict):
            continue
        value = _finite_float(mode_result.get("wall_minus_open"))
        if value is not None:
            finite_by_group[key].append(value)

    group_means = [
        mean(values)
        for key, values in finite_by_group.items()
        if key in total_by_group and values
    ]
    if not group_means:
        return {
            "wall_minus_open": None,
            "direction": None,
            "finite_records": 0,
            "total_records": len(records),
            "finite_groups": 0,
            "total_groups": len(total_by_group),
            "finite_records_per_finite_group": _support_summary([]),
        }

    value = float(mean(group_means))
    counts = [len(values) for values in finite_by_group.values() if values]
    return {
        "wall_minus_open": value,
        "direction": _direction(value),
        "finite_records": sum(counts),
        "total_records": len(records),
        "finite_groups": len(group_means),
        "total_groups": len(total_by_group),
        "finite_records_per_finite_group": _support_summary(counts),
    }


def _animal_summary(
    animal: dict[str, Any],
) -> dict[str, Any]:
    records = animal.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{animal.get('animal')}: records is not a list")
    modes: dict[str, Any] = {}
    for mode in RATE_MODES:
        schemes = {
            scheme: aggregate_records(
                records,
                mode=mode,
                scheme=scheme,
            )
            for scheme in GROUPERS
        }
        finite_directions = {
            item["direction"]
            for item in schemes.values()
            if item["direction"] is not None
        }
        modes[mode] = {
            "direction_consistent_across_schemes": (
                len(finite_directions) <= 1
            ),
            "schemes": schemes,
        }
    return {
        "animal": str(animal["animal"]),
        "modes": modes,
    }


def _cohort_summary(
    animals: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in RATE_MODES:
        mode_result: dict[str, Any] = {}
        for scheme in GROUPERS:
            animal_values = {
                animal["animal"]: animal["modes"][mode]["schemes"][scheme][
                    "wall_minus_open"
                ]
                for animal in animals
                if animal["modes"][mode]["schemes"][scheme][
                    "wall_minus_open"
                ]
                is not None
            }
            values = list(animal_values.values())
            directions = [_direction(value) for value in values]
            mode_result[scheme] = {
                "animals": len(values),
                "positive_animals": directions.count("positive"),
                "negative_animals": directions.count("negative"),
                "zero_animals": directions.count("zero"),
                "equal_animal_mean_wall_minus_open": (
                    float(mean(values)) if values else None
                ),
                "animal_median_wall_minus_open": (
                    float(median(values)) if values else None
                ),
                "animal_values": animal_values,
            }
        result[mode] = mode_result
    return result


def _reconstruction_check(
    source: dict[str, Any],
    animals: list[dict[str, Any]],
) -> dict[str, Any]:
    source_animals = {
        str(animal["animal"]): animal for animal in source["animals"]
    }
    per_animal: dict[str, Any] = {}
    errors: list[float] = []
    cohort_errors: list[float] = []
    cohort_summary: dict[str, Any] = {}
    for animal in animals:
        animal_name = animal["animal"]
        source_animal = source_animals[animal_name]
        per_mode: dict[str, Any] = {}
        for mode in RATE_MODES:
            recomputed = animal["modes"][mode]["schemes"][
                "prediction_record_equal"
            ]["wall_minus_open"]
            stored_animal = _finite_float(
                source_animal["summaries"][mode]["wall_minus_open_mean"]
            )
            stored_cohort = _finite_float(
                source["cohort"][mode]["animal_values"][animal_name]
            )
            if recomputed is None or stored_animal is None or stored_cohort is None:
                raise ValueError(
                    f"{animal_name}/{mode}: missing stored reconstruction value"
                )
            animal_error = abs(recomputed - stored_animal)
            cohort_error = abs(recomputed - stored_cohort)
            errors.append(animal_error)
            cohort_errors.append(cohort_error)
            per_mode[mode] = {
                "recomputed_prediction_record_equal": recomputed,
                "stored_animal_summary": stored_animal,
                "stored_cohort_animal_value": stored_cohort,
                "absolute_error_vs_animal_summary": animal_error,
                "absolute_error_vs_cohort_animal_value": cohort_error,
            }
        per_animal[animal_name] = per_mode

    cohort_statistic_errors: list[float] = []
    for mode in RATE_MODES:
        recomputed_values = [
            animal["modes"][mode]["schemes"][
                "prediction_record_equal"
            ]["wall_minus_open"]
            for animal in animals
        ]
        if any(value is None for value in recomputed_values):
            raise ValueError(f"{mode}: missing recomputed animal value")
        finite_values = [
            float(value)
            for value in recomputed_values
            if value is not None
        ]
        recomputed_mean = float(mean(finite_values))
        recomputed_median = float(median(finite_values))
        stored_mean = _finite_float(
            source["cohort"][mode]["animal_mean_wall_minus_open"]
        )
        stored_median = _finite_float(
            source["cohort"][mode]["animal_median_wall_minus_open"]
        )
        if stored_mean is None or stored_median is None:
            raise ValueError(f"{mode}: missing stored cohort statistic")
        mean_error = abs(recomputed_mean - stored_mean)
        median_error = abs(recomputed_median - stored_median)
        cohort_statistic_errors.extend([mean_error, median_error])
        cohort_summary[mode] = {
            "recomputed_equal_animal_mean": recomputed_mean,
            "stored_animal_mean": stored_mean,
            "absolute_mean_error": mean_error,
            "recomputed_animal_median": recomputed_median,
            "stored_animal_median": stored_median,
            "absolute_median_error": median_error,
        }

    max_animal_error = max(errors, default=0.0)
    max_cohort_error = max(cohort_errors, default=0.0)
    max_cohort_statistic_error = max(cohort_statistic_errors, default=0.0)
    passed = (
        max_animal_error <= RECONSTRUCTION_TOLERANCE
        and max_cohort_error <= RECONSTRUCTION_TOLERANCE
        and max_cohort_statistic_error <= RECONSTRUCTION_TOLERANCE
    )
    return {
        "tolerance": RECONSTRUCTION_TOLERANCE,
        "passed": passed,
        "maximum_absolute_error_vs_animal_summary": max_animal_error,
        "maximum_absolute_error_vs_cohort_animal_value": max_cohort_error,
        "maximum_absolute_error_vs_cohort_summary": (
            max_cohort_statistic_error
        ),
        "cohort_summary": cohort_summary,
        "per_animal": per_animal,
    }


def build_report(
    source: dict[str, Any],
    *,
    source_path: Path,
) -> dict[str, Any]:
    source_animals = source.get("animals")
    if not isinstance(source_animals, list):
        raise ValueError("source animals is not a list")
    animals = [_animal_summary(animal) for animal in source_animals]
    reconstruction = _reconstruction_check(source, animals)
    if not reconstruction["passed"]:
        raise RuntimeError(
            "prediction-record weighting did not reconstruct stored values"
        )
    resolved_source = source_path.resolve()
    try:
        source_label = resolved_source.relative_to(ROOT).as_posix()
    except ValueError:
        source_label = source_path.name

    return {
        "status": "aggregation_robustness_not_independent_validation",
        "question": (
            "Does the animal-level exact-wall minus open-template local "
            "prediction advantage retain its direction under equal weighting "
            "of records, exposure pairs, target environments, or exact "
            "oriented seams?"
        ),
        "scope": {
            "analysis_type": (
                "descriptive re-aggregation of existing target-rate-held-out "
                "local prediction records"
            ),
            "not_an_independent_validation": True,
            "biological_unit": "animal",
            "cohort_weighting": "equal weight per animal",
            "p_values_reported": False,
            "rate_modes": list(RATE_MODES),
        },
        "aggregation_schemes": {
            "prediction_record_equal": (
                "current estimator: equal weight for every finite prediction "
                "record within an animal"
            ),
            "exposure_pair_equal": (
                "mean records within each training-to-test exposure pair, "
                "then give each finite exposure pair equal weight"
            ),
            "target_environment_equal": (
                "mean records for each target-environment label across "
                "exposure pairs, then give each finite environment equal "
                "weight"
            ),
            "exact_oriented_seam_equal": (
                "mean records for each ordered physical seam plus orientation "
                "label across targets and exposure pairs, then give each "
                "finite exact oriented seam equal weight"
            ),
        },
        "schema_interpretation": {
            "target_environment": (
                "the same target-environment label is one group even when it "
                "appears in more than one exposure pair"
            ),
            "exact_oriented_seam": (
                "group key is orientation plus the ordered two-entry seam; "
                "reversing the seam order creates a different group"
            ),
            "schema_ambiguities_detected": [],
        },
        "source": {
            "portable_path": source_label,
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
        "current_weighting_reconstruction": reconstruction,
        "cohort": _cohort_summary(animals),
        "animals": animals,
    }


def main() -> None:
    argument = parse_arguments()
    source = json.loads(argument.input.read_text(encoding="utf-8"))
    report = build_report(source, source_path=argument.input)
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    compact_cohort = {
        mode: {
            scheme: {
                "positive_animals": summary["positive_animals"],
                "animals": summary["animals"],
                "equal_animal_mean_wall_minus_open": summary[
                    "equal_animal_mean_wall_minus_open"
                ],
            }
            for scheme, summary in schemes.items()
        }
        for mode, schemes in report["cohort"].items()
    }
    print(
        json.dumps(
            {
                "status": report["status"],
                "reconstruction_passed": report[
                    "current_weighting_reconstruction"
                ]["passed"],
                "cohort": compact_cohort,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
