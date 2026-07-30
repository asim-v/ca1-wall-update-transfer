"""Screen for experience-dependent resolution of local boundary aliases.

This is an explicitly post-outcome exploratory discovery analysis.  It asks
whether the same registered CA1 cells become less spatially similar across
two partitions that have identical immediate allocentric N/E/S/W wall
contexts, relative to non-alias partition pairs at the same grid distance.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ca1_geometry.aliases import (  # noqa: E402
    cellwise_partition_correlations,
    common_relative_support,
    distance_matched_control_pairs,
    exact_alias_pairs,
    mean_fisher_correlation,
)
from ca1_geometry.io import Mat73Animal  # noqa: E402


METRICS = ("mean_r", "fisher_r", "median_r")
MAP_MODES = {
    "unsmoothed": False,
    "smoothed": True,
}


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
            / "alias_resolution_screen.json"
        ),
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-bins", type=int, default=10)
    parser.add_argument("--minimum-stable-cells", type=int, default=10)
    return parser.parse_args()


def _correlation_summary(
    correlations: list[np.ndarray],
    *,
    minimum_stable_cells: int,
) -> dict[str, Any] | None:
    """Summarize cells with a defined pair correlation on every exposure."""

    matrix = np.stack(correlations, axis=0)
    stable = np.all(np.isfinite(matrix), axis=0)
    if np.count_nonzero(stable) < minimum_stable_cells:
        return None
    values = matrix[:, stable]
    exposure = []
    for index, item in enumerate(values):
        exposure.append(
            {
                "exposure": index + 1,
                "mean_r": float(np.mean(item)),
                "fisher_r": mean_fisher_correlation(item),
                "median_r": float(np.median(item)),
            }
        )
    return {
        "stable_cells": int(np.count_nonzero(stable)),
        "exposures": exposure,
    }


def _pair_mode_record(
    *,
    rate_maps: list[np.ndarray],
    occupancy_maps: list[np.ndarray],
    pair: tuple[int, int],
    controls: tuple[tuple[int, int], ...],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_stable_cells: int,
) -> dict[str, Any] | None:
    target_support = common_relative_support(
        occupancy_maps,
        pair,
        minimum_seconds=minimum_seconds,
    )
    if np.count_nonzero(target_support) < minimum_bins:
        return None
    target = _correlation_summary(
        [
            cellwise_partition_correlations(rate, pair, target_support)
            for rate in rate_maps
        ],
        minimum_stable_cells=minimum_stable_cells,
    )
    if target is None:
        return None

    control_records = []
    for control_pair in controls:
        support = common_relative_support(
            occupancy_maps,
            control_pair,
            minimum_seconds=minimum_seconds,
        )
        if np.count_nonzero(support) < minimum_bins:
            continue
        summary = _correlation_summary(
            [
                cellwise_partition_correlations(
                    rate,
                    control_pair,
                    support,
                )
                for rate in rate_maps
            ],
            minimum_stable_cells=minimum_stable_cells,
        )
        if summary is None:
            continue
        control_records.append(
            {
                "pair": list(control_pair),
                "common_support_bins": int(np.count_nonzero(support)),
                **summary,
            }
        )
    if not control_records:
        return None

    exposure_records = []
    for exposure_index, target_exposure in enumerate(target["exposures"]):
        record: dict[str, Any] = {"exposure": exposure_index + 1}
        for metric in METRICS:
            alias_value = target_exposure[metric]
            control_value = float(
                np.mean(
                    [
                        control["exposures"][exposure_index][metric]
                        for control in control_records
                    ]
                )
            )
            record[f"alias_{metric}"] = alias_value
            record[f"control_{metric}"] = control_value
            record[f"advantage_{metric}"] = alias_value - control_value
        exposure_records.append(record)

    change = {}
    for prefix in ("alias", "control", "advantage"):
        for metric in METRICS:
            key = f"{prefix}_{metric}"
            change[key] = (
                exposure_records[-1][key] - exposure_records[0][key]
            )
    return {
        "target_common_support_bins": int(
            np.count_nonzero(target_support)
        ),
        "target_stable_cells": target["stable_cells"],
        "eligible_controls": len(control_records),
        "controls": control_records,
        "exposures": exposure_records,
        "last_minus_first": change,
    }


def _aggregate_pair_modes(
    pairs: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any] | None:
    eligible = [item["modes"][mode] for item in pairs if mode in item["modes"]]
    if not eligible:
        return None
    n_exposures = len(eligible[0]["exposures"])
    exposure_records = []
    for exposure_index in range(n_exposures):
        record: dict[str, Any] = {"exposure": exposure_index + 1}
        for prefix in ("alias", "control", "advantage"):
            for metric in METRICS:
                key = f"{prefix}_{metric}"
                record[key] = float(
                    np.mean(
                        [
                            item["exposures"][exposure_index][key]
                            for item in eligible
                        ]
                    )
                )
        exposure_records.append(record)
    change = {
        key: exposure_records[-1][key] - exposure_records[0][key]
        for key in exposure_records[0]
        if key != "exposure"
    }
    return {
        "eligible_alias_pairs": len(eligible),
        "exposures": exposure_records,
        "last_minus_first": change,
    }


def analyze_environment(
    animal: Mat73Animal,
    environment: str,
    sessions: list[int],
    *,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_stable_cells: int,
) -> dict[str, Any] | None:
    blocked = animal.blocked(sessions[0])
    if any(animal.blocked(session) != blocked for session in sessions):
        raise ValueError("one environment label maps to multiple geometries")
    aliases = exact_alias_pairs(blocked)
    if not aliases:
        return None

    common_cells = animal.common_registered_cells(*sessions)
    occupancy_maps = [animal.sampling_map(session) for session in sessions]
    rates: dict[str, list[np.ndarray]] = {}
    for mode, smoothed in MAP_MODES.items():
        rates[mode] = [
            animal.stored_rate_maps(
                session,
                smoothed=smoothed,
            )[common_cells]
            for session in sessions
        ]

    pairs = []
    for pair in aliases:
        controls = distance_matched_control_pairs(blocked, pair)
        modes = {}
        for mode in MAP_MODES:
            record = _pair_mode_record(
                rate_maps=rates[mode],
                occupancy_maps=occupancy_maps,
                pair=pair,
                controls=controls,
                minimum_seconds=minimum_seconds,
                minimum_bins=minimum_bins,
                minimum_stable_cells=minimum_stable_cells,
            )
            if record is not None:
                modes[mode] = record
        pairs.append(
            {
                "alias_pair": list(pair),
                "distance_matched_control_pairs": [
                    list(item) for item in controls
                ],
                "modes": modes,
            }
        )

    aggregate = {}
    for mode in MAP_MODES:
        result = _aggregate_pair_modes(pairs, mode)
        if result is not None:
            aggregate[mode] = result
    if not aggregate:
        return None
    return {
        "environment": environment,
        "blocked_partitions": list(blocked),
        "sessions": [session + 1 for session in sessions],
        "registered_cells_common_to_all_exposures": int(common_cells.size),
        "pairs": pairs,
        "aggregate": aggregate,
    }


def _aggregate_environments(
    environments: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any] | None:
    eligible = [
        item["aggregate"][mode]
        for item in environments
        if mode in item["aggregate"]
    ]
    if not eligible:
        return None
    n_exposures = len(eligible[0]["exposures"])
    exposure_records = []
    for exposure_index in range(n_exposures):
        record: dict[str, Any] = {"exposure": exposure_index + 1}
        for prefix in ("alias", "control", "advantage"):
            for metric in METRICS:
                key = f"{prefix}_{metric}"
                record[key] = float(
                    np.mean(
                        [
                            item["exposures"][exposure_index][key]
                            for item in eligible
                        ]
                    )
                )
        exposure_records.append(record)
    return {
        "eligible_environments": len(eligible),
        "exposures": exposure_records,
        "last_minus_first": {
            key: exposure_records[-1][key] - exposure_records[0][key]
            for key in exposure_records[0]
            if key != "exposure"
        },
    }


def analyze_animal(
    path: Path,
    *,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_stable_cells: int,
) -> dict[str, Any]:
    with Mat73Animal(path) as animal:
        sessions_by_environment: dict[str, list[int]] = defaultdict(list)
        for session in range(animal.n_sessions):
            sessions_by_environment[animal.environment(session)].append(
                session
            )
        environments = []
        for environment, sessions in sorted(sessions_by_environment.items()):
            result = analyze_environment(
                animal,
                environment,
                sessions,
                minimum_seconds=minimum_seconds,
                minimum_bins=minimum_bins,
                minimum_stable_cells=minimum_stable_cells,
            )
            if result is not None:
                environments.append(result)

    aggregate = {}
    for mode in MAP_MODES:
        result = _aggregate_environments(environments, mode)
        if result is not None:
            aggregate[mode] = result
    return {
        "animal": path.name.removesuffix(".complete.mat"),
        "alias_environments": environments,
        "aggregate": aggregate,
    }


def cohort_summary(animals: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in MAP_MODES:
        eligible = [
            animal["aggregate"][mode]
            for animal in animals
            if mode in animal["aggregate"]
        ]
        mode_result: dict[str, Any] = {
            "eligible_animals": len(eligible),
            "last_minus_first": {},
        }
        for prefix in ("alias", "control", "advantage"):
            for metric in METRICS:
                key = f"{prefix}_{metric}"
                values = np.asarray(
                    [
                        animal["last_minus_first"][key]
                        for animal in eligible
                    ],
                    dtype=np.float64,
                )
                mode_result["last_minus_first"][key] = {
                    "animal_mean": float(np.mean(values)),
                    "animal_median": float(np.median(values)),
                    "negative_animals": int(np.count_nonzero(values < 0)),
                    "values": values.tolist(),
                }

        by_exposure: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for animal in eligible:
            for exposure in animal["exposures"]:
                by_exposure[exposure["exposure"]].append(exposure)
        mode_result["by_exposure"] = []
        for ordinal, exposures in sorted(by_exposure.items()):
            record: dict[str, Any] = {
                "exposure": ordinal,
                "animals": len(exposures),
            }
            for prefix in ("alias", "control", "advantage"):
                for metric in METRICS:
                    key = f"{prefix}_{metric}"
                    record[key] = float(
                        np.mean([item[key] for item in exposures])
                    )
            mode_result["by_exposure"].append(record)
        result[mode] = mode_result
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    argument = parse_arguments()
    paths = sorted(
        argument.data_dir.glob("QLAK-CA1-*.complete.mat")
    )
    if not paths:
        raise FileNotFoundError(
            f"no complete animal files found beneath {argument.data_dir}"
        )

    animals = []
    for path in paths:
        result = analyze_animal(
            path,
            minimum_seconds=argument.minimum_seconds,
            minimum_bins=argument.minimum_bins,
            minimum_stable_cells=argument.minimum_stable_cells,
        )
        animals.append(result)
        primary = result["aggregate"].get("unsmoothed")
        print(
            result["animal"],
            None
            if primary is None
            else primary["last_minus_first"]["advantage_mean_r"],
        )

    report = {
        "status": (
            "post_outcome_exploratory_screen_not_confirmatory_inference"
        ),
        "question": (
            "Do longitudinally registered CA1 cells progressively resolve "
            "within-environment partitions with identical immediate "
            "allocentric N/E/S/W boundary context?"
        ),
        "settings": {
            "spatial_grid": "15x15 bins; 5x5 bins per 25 cm partition",
            "common_support": (
                "relative bins meeting the dwell threshold in both members "
                "of a pair on every exposure"
            ),
            "minimum_seconds_per_pair_member_per_exposure": (
                argument.minimum_seconds
            ),
            "minimum_common_relative_bins": argument.minimum_bins,
            "minimum_cells_with_defined_correlation_on_every_exposure": (
                argument.minimum_stable_cells
            ),
            "cell_registration": (
                "intersection of registered cells across all exposures of "
                "one environment"
            ),
            "primary_metric": (
                "arithmetic mean of per-cell Pearson spatial correlations"
            ),
            "sensitivity_metrics": (
                "Fisher-z mean and median per-cell Pearson correlation"
            ),
            "control": (
                "accessible non-alias partition pairs at identical "
                "partition-grid Euclidean distance"
            ),
            "inferential_unit": "animal",
        },
        "cohort": cohort_summary(animals),
        "animals": animals,
    }
    report = _json_safe(report)
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["cohort"], indent=2))


if __name__ == "__main__":
    main()
