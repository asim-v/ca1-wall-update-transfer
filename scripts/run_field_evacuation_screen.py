"""Screen for coordinate-specific relocation of newly inaccessible fields.

For a cell whose square-session peak lies in a partition that is blocked in a
deformation, the peak coordinate is reflected across the nearest accessible
wall face.  The outcome is deformation-induced rate gain at that projected
bin relative to same-depth tangential bins along the same wall.  Bracketing
square sessions provide an independent baseline and reinstatement criterion.
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

from ca1_geometry.field_evacuation import (  # noqa: E402
    map_bin_partition,
    nearest_accessible_reflections,
    reflection_contrast,
)
from ca1_geometry.io import Mat73Animal  # noqa: E402


SELECTIONS = ("pre_only", "same_partition", "stable_peak_10_cm")
MAP_MODES = {"unsmoothed": False, "smoothed": True}


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
            / "field_evacuation_screen.json"
        ),
    )
    parser.add_argument("--minimum-cells", type=int, default=8)
    return parser.parse_args()


def _finite_peak(rate_map: np.ndarray) -> tuple[int, int] | None:
    finite = np.isfinite(rate_map)
    if (
        np.count_nonzero(finite) < 3
        or np.nanmax(rate_map) <= 0
        or np.nanstd(rate_map) <= np.finfo(np.float64).eps
    ):
        return None
    flat = int(np.nanargmax(rate_map))
    return tuple(int(value) for value in np.unravel_index(flat, (15, 15)))


def _cell_selection(
    pre_peak: tuple[int, int],
    post_peak: tuple[int, int] | None,
    blocked: tuple[int, ...],
) -> tuple[str, ...]:
    pre_partition = map_bin_partition(*pre_peak)
    if pre_partition not in blocked:
        return ()
    result = ["pre_only"]
    if (
        post_peak is not None
        and map_bin_partition(*post_peak) == pre_partition
    ):
        result.append("same_partition")
        distance = np.linalg.norm(
            np.asarray(pre_peak, dtype=float)
            - np.asarray(post_peak, dtype=float)
        )
        if distance <= 2.0:
            result.append("stable_peak_10_cm")
    return tuple(result)


def analyze_target(
    animal: Mat73Animal,
    pre_square: int,
    target: int,
    post_square: int,
    *,
    exposure: int,
    minimum_cells: int,
) -> dict[str, Any]:
    blocked = animal.blocked(target)
    cells = animal.common_registered_cells(
        pre_square,
        target,
        post_square,
    )
    selection_maps = {
        "pre": animal.stored_rate_maps(pre_square, smoothed=True)[cells],
        "post": animal.stored_rate_maps(post_square, smoothed=True)[cells],
    }
    outcome_maps = {
        mode: {
            "pre": animal.stored_rate_maps(
                pre_square,
                smoothed=smoothed,
            )[cells],
            "target": animal.stored_rate_maps(
                target,
                smoothed=smoothed,
            )[cells],
            "post": animal.stored_rate_maps(
                post_square,
                smoothed=smoothed,
            )[cells],
        }
        for mode, smoothed in MAP_MODES.items()
    }

    records: dict[str, dict[str, list[float]]] = {
        selection: {
            mode: [] for mode in MAP_MODES
        }
        for selection in SELECTIONS
    }
    target_contrasts: dict[str, dict[str, list[float]]] = {
        selection: {
            mode: [] for mode in MAP_MODES
        }
        for selection in SELECTIONS
    }
    for local_cell in range(cells.size):
        pre_peak = _finite_peak(selection_maps["pre"][local_cell])
        post_peak = _finite_peak(selection_maps["post"][local_cell])
        if pre_peak is None:
            continue
        selected = _cell_selection(pre_peak, post_peak, blocked)
        if not selected:
            continue
        queries = nearest_accessible_reflections(blocked, pre_peak)
        if not queries:
            continue
        for mode in MAP_MODES:
            target_contrast = reflection_contrast(
                outcome_maps[mode]["target"][local_cell],
                queries,
            )
            pre_contrast = reflection_contrast(
                outcome_maps[mode]["pre"][local_cell],
                queries,
            )
            post_contrast = reflection_contrast(
                outcome_maps[mode]["post"][local_cell],
                queries,
            )
            if not np.all(
                np.isfinite(
                    [target_contrast, pre_contrast, post_contrast]
                )
            ):
                continue
            gain = target_contrast - 0.5 * (
                pre_contrast + post_contrast
            )
            for selection in selected:
                records[selection][mode].append(float(gain))
                target_contrasts[selection][mode].append(
                    float(target_contrast)
                )

    summaries = {}
    for selection in SELECTIONS:
        modes = {}
        for mode in MAP_MODES:
            value = np.asarray(records[selection][mode])
            target_value = np.asarray(
                target_contrasts[selection][mode]
            )
            if value.size < minimum_cells:
                continue
            modes[mode] = {
                "cells": int(value.size),
                "projection_gain_mean": float(np.mean(value)),
                "projection_gain_median": float(np.median(value)),
                "target_projection_contrast_mean": float(
                    np.mean(target_value)
                ),
                "positive_cell_fraction": float(np.mean(value > 0)),
            }
        if modes:
            summaries[selection] = modes
    return {
        "exposure": exposure,
        "environment": animal.environment(target),
        "session": target + 1,
        "blocked_partitions": list(blocked),
        "common_registered_cells": int(cells.size),
        "selections": summaries,
    }


def _aggregate_targets(
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for selection in SELECTIONS:
        selection_result = {}
        for mode in MAP_MODES:
            eligible = [
                target["selections"][selection][mode]
                for target in targets
                if selection in target["selections"]
                and mode in target["selections"][selection]
            ]
            if not eligible:
                continue
            selection_result[mode] = {
                "eligible_targets": len(eligible),
                "median_cells_per_target": float(
                    np.median([item["cells"] for item in eligible])
                ),
                "projection_gain_mean": float(
                    np.mean(
                        [item["projection_gain_mean"] for item in eligible]
                    )
                ),
                "target_projection_contrast_mean": float(
                    np.mean(
                        [
                            item["target_projection_contrast_mean"]
                            for item in eligible
                        ]
                    )
                ),
                "positive_target_fraction": float(
                    np.mean(
                        [
                            item["projection_gain_mean"] > 0
                            for item in eligible
                        ]
                    )
                ),
            }
        if selection_result:
            result[selection] = selection_result
    return result


def analyze_animal(
    path: Path,
    *,
    minimum_cells: int,
) -> dict[str, Any]:
    with Mat73Animal(path) as animal:
        square = [
            session
            for session in range(animal.n_sessions)
            if animal.environment(session) == "square"
        ]
        targets = []
        for exposure, (pre, post) in enumerate(
            zip(square[:-1], square[1:], strict=True),
            start=1,
        ):
            for target in range(pre + 1, post):
                targets.append(
                    analyze_target(
                        animal,
                        pre,
                        target,
                        post,
                        exposure=exposure,
                        minimum_cells=minimum_cells,
                    )
                )
    return {
        "animal": path.name.removesuffix(".complete.mat"),
        "targets": targets,
        "aggregate": _aggregate_targets(targets),
        "by_exposure": {
            str(exposure): _aggregate_targets(
                [
                    target
                    for target in targets
                    if target["exposure"] == exposure
                ]
            )
            for exposure in sorted(
                {target["exposure"] for target in targets}
            )
        },
    }


def cohort_summary(animals: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for selection in SELECTIONS:
        selection_result = {}
        for mode in MAP_MODES:
            values = [
                animal["aggregate"][selection][mode][
                    "projection_gain_mean"
                ]
                for animal in animals
                if selection in animal["aggregate"]
                and mode in animal["aggregate"][selection]
            ]
            if not values:
                continue
            array = np.asarray(values)
            selection_result[mode] = {
                "eligible_animals": len(values),
                "animal_mean": float(np.mean(array)),
                "animal_median": float(np.median(array)),
                "positive_animals": int(np.count_nonzero(array > 0)),
                "values": array.tolist(),
            }
        if selection_result:
            result[selection] = selection_result
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
        raise FileNotFoundError("no complete animal files found")
    animals = []
    for path in paths:
        result = analyze_animal(
            path,
            minimum_cells=argument.minimum_cells,
        )
        animals.append(result)
        strict = (
            result["aggregate"]
            .get("stable_peak_10_cm", {})
            .get("unsmoothed")
        )
        print(
            result["animal"],
            None if strict is None else strict["projection_gain_mean"],
        )

    report = {
        "status": (
            "post_outcome_exploratory_field_evacuation_screen_not_confirmatory"
        ),
        "question": (
            "When a stable square-session field becomes inaccessible, does "
            "activity relocate to the coordinate reflected across the nearest "
            "reachable new wall?"
        ),
        "settings": {
            "source_peak": (
                "global peak of the pre-deformation square smoothed map"
            ),
            "reinstatement_sensitivity": (
                "post-square peak in the same blocked partition and within "
                "10 cm of the pre-square peak"
            ),
            "projection": (
                "bin reflected across the nearest blocked-accessible face"
            ),
            "within_wall_control": (
                "other tangential bins at the same reflected normal depth"
            ),
            "outcome": (
                "target projection contrast minus the mean contrast in "
                "independent bracketing square sessions"
            ),
            "primary_map": "released unsmoothed event probability per frame",
            "sensitivity_map": "released smoothed map",
            "minimum_cells_per_target": argument.minimum_cells,
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
