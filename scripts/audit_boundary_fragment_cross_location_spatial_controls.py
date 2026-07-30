"""Spatial-overlap and translation controls for cross-location transfer.

Exact 25 cm seam-midpoint distance does not imply identical strip proximity.
This audit classifies source seams by signed-normal relation and by whether
the 25 cm displacement is tangential or normal to the target wall. It records
5 cm-bin overlap/minimum distance and recomputes neural transfer with the
query-common registered-cell mask used by the validation branch.

Open-place correlations and source-effect-to-target-open correlations are
included to assess generic nearby-place-field autocorrelation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_boundary_fragment_cross_location_transfer as transfer  # noqa: E402
from ca1_geometry.io import Mat73Animal  # noqa: E402
from ca1_geometry.seams import (  # noqa: E402
    OrientedSeam,
    SeamState,
    internal_seams,
    seam_frame,
    seam_state,
    seam_strip_bins,
)


CATEGORIES = (
    "same_tangential",
    "same_normal",
    "opposite_tangential",
    "opposite_normal_facing_overlap",
    "opposite_normal_away",
)
DERIVED_GROUPS = (
    "same_all",
    "opposite_all",
    "opposite_normal_all",
)
METRICS = (
    "source_effect_r_to_target_residual",
    "source_effect_r_to_target_open_profile",
    "source_open_r_to_target_open_profile",
    "source_open_r_to_target_residual",
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
            / "boundary_fragment_cross_location_spatial_controls.json"
        ),
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-bins", type=int, default=6)
    parser.add_argument("--minimum-cells", type=int, default=20)
    return parser.parse_args()


def _midpoint(seam: OrientedSeam) -> np.ndarray:
    start, end, _normal = seam_frame(seam)
    return (start + end) / 2.0


def _strip_minimum_distance_cm(
    first: OrientedSeam,
    second: OrientedSeam,
) -> float:
    first_bins = seam_strip_bins(first)
    second_bins = seam_strip_bins(second)
    return float(
        min(
            np.linalg.norm(np.subtract(a, b)) * 5.0
            for a in first_bins
            for b in second_bins
        )
    )


def spatial_relation(
    target: OrientedSeam,
    source: OrientedSeam,
) -> dict[str, Any] | None:
    """Classify a parallel exact-25-cm source and its strip proximity."""

    if source.unordered == target.unordered:
        return None
    if not np.isclose(
        transfer._midpoint_distance_cm(target, source),
        25.0,
    ):
        return None
    relation = transfer._orientation_relation(target, source)
    if relation not in ("same_signed_normal", "opposite_normal"):
        return None
    _start, _end, normal = seam_frame(target)
    tangent = np.asarray([-normal[1], normal[0]])
    displacement = _midpoint(source) - _midpoint(target)
    normal_cm = float(abs(displacement @ normal))
    tangential_cm = float(abs(displacement @ tangent))
    if np.isclose(normal_cm, 25.0) and np.isclose(tangential_cm, 0.0):
        translation = "normal"
    elif np.isclose(normal_cm, 0.0) and np.isclose(
        tangential_cm,
        25.0,
    ):
        translation = "tangential"
    else:
        raise AssertionError("exact grid step is not axis aligned")
    overlap = len(
        set(seam_strip_bins(target))
        & set(seam_strip_bins(source))
    )
    if relation == "same_signed_normal":
        category = f"same_{translation}"
    elif translation == "tangential":
        category = "opposite_tangential"
    elif overlap:
        category = "opposite_normal_facing_overlap"
    else:
        category = "opposite_normal_away"
    return {
        "category": category,
        "orientation_relation": relation,
        "translation_axis": translation,
        "normal_displacement_cm": normal_cm,
        "tangential_displacement_cm": tangential_cm,
        "strip_overlap_bins": overlap,
        "strip_minimum_bin_center_distance_cm": (
            _strip_minimum_distance_cm(target, source)
        ),
    }


def geometry_census() -> dict[str, Any]:
    counts: Counter[str] = Counter()
    detail: dict[str, dict[str, Any]] = {}
    for target in internal_seams():
        for source in internal_seams():
            relation = spatial_relation(target, source)
            if relation is None:
                continue
            category = relation["category"]
            counts[category] += 1
            if category not in detail:
                detail[category] = {
                    key: value
                    for key, value in relation.items()
                    if key != "category"
                }
            else:
                for key, value in relation.items():
                    if key == "category":
                        continue
                    if detail[category][key] != value:
                        raise AssertionError(
                            f"{category} has nonconstant {key}"
                        )
    return {
        category: {
            "directed_seam_pairs": counts[category],
            **detail[category],
        }
        for category in CATEGORIES
    }


def _mean_records(
    records: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any] | None:
    if not records:
        return None
    return {
        "source_pairs": len(records),
        **{
            metric: float(
                np.mean(
                    [
                        record["metrics"][mode][metric]
                        for record in records
                    ]
                )
            )
            for metric in METRICS
        },
    }


def _combine_categories(
    records: list[dict[str, Any]],
    categories: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["spatial"]["category"] in categories
    ]


def _query_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not records:
        return None
    groups = {
        category: [
            record
            for record in records
            if record["spatial"]["category"] == category
        ]
        for category in CATEGORIES
    }
    groups.update(
        {
            "same_all": _combine_categories(
                records,
                ("same_tangential", "same_normal"),
            ),
            "opposite_all": _combine_categories(
                records,
                (
                    "opposite_tangential",
                    "opposite_normal_facing_overlap",
                    "opposite_normal_away",
                ),
            ),
            "opposite_normal_all": _combine_categories(
                records,
                (
                    "opposite_normal_facing_overlap",
                    "opposite_normal_away",
                ),
            ),
        }
    )
    if not groups["same_all"]:
        return None
    by_mode = {}
    for mode in transfer.RATE_MODES:
        summary = {
            label: _mean_records(values, mode=mode)
            for label, values in groups.items()
        }

        def difference(
            first: str,
            second: str,
            metric: str,
        ) -> float | None:
            if summary[first] is None or summary[second] is None:
                return None
            return float(
                summary[first][metric] - summary[second][metric]
            )

        by_mode[mode] = {
            "groups": summary,
            "contrasts": {
                "same_tangential_minus_opposite_tangential_transfer": (
                    difference(
                        "same_tangential",
                        "opposite_tangential",
                        "source_effect_r_to_target_residual",
                    )
                ),
                "same_normal_minus_opposite_normal_transfer": difference(
                    "same_normal",
                    "opposite_normal_all",
                    "source_effect_r_to_target_residual",
                ),
                (
                    "same_normal_minus_opposite_normal_facing_overlap_"
                    "transfer"
                ): difference(
                    "same_normal",
                    "opposite_normal_facing_overlap",
                    "source_effect_r_to_target_residual",
                ),
                "same_normal_minus_opposite_normal_away_transfer": (
                    difference(
                        "same_normal",
                        "opposite_normal_away",
                        "source_effect_r_to_target_residual",
                    )
                ),
                "same_tangential_minus_same_normal_transfer": difference(
                    "same_tangential",
                    "same_normal",
                    "source_effect_r_to_target_residual",
                ),
                "same_all_minus_opposite_all_transfer": difference(
                    "same_all",
                    "opposite_all",
                    "source_effect_r_to_target_residual",
                ),
                "same_all_effect_specificity_over_target_open": difference(
                    "same_all",
                    "same_all",
                    "source_effect_r_to_target_residual",
                ),
                "same_minus_opposite_open_place_autocorrelation": difference(
                    "same_all",
                    "opposite_all",
                    "source_open_r_to_target_open_profile",
                ),
            },
        }
        same = summary["same_all"]
        assert same is not None
        by_mode[mode]["contrasts"][
            "same_all_effect_specificity_over_target_open"
        ] = float(
            same["source_effect_r_to_target_residual"]
            - same["source_effect_r_to_target_open_profile"]
        )
    return {"modes": by_mode}


def _animal_summary(
    queries: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    labels = tuple(
        queries[0]["modes"][mode]["contrasts"]
    )
    contrasts = {}
    for label in labels:
        values = [
            query["modes"][mode]["contrasts"][label]
            for query in queries
            if query["modes"][mode]["contrasts"][label] is not None
        ]
        contrasts[label] = {
            "target_queries": len(values),
            "mean": float(np.mean(values)) if values else None,
            "positive_target_queries": int(
                np.count_nonzero(np.asarray(values) > 0)
            ),
        }
    primary = [
        query["modes"][mode]["groups"]["same_all"][
            "source_effect_r_to_target_residual"
        ]
        for query in queries
        if query["modes"][mode]["groups"]["same_all"] is not None
    ]
    category_transfer = {}
    for category in CATEGORIES:
        values = [
            query["modes"][mode]["groups"][category][
                "source_effect_r_to_target_residual"
            ]
            for query in queries
            if query["modes"][mode]["groups"][category] is not None
        ]
        category_transfer[category] = {
            "target_queries": len(values),
            "mean": float(np.mean(values)) if values else None,
            "positive_target_queries": int(
                np.count_nonzero(np.asarray(values) > 0)
            ),
        }
    return {
        "eligible_target_queries": len(queries),
        "same_all_transfer": {
            "target_queries": len(primary),
            "mean": float(np.mean(primary)),
            "positive_target_queries": int(
                np.count_nonzero(np.asarray(primary) > 0)
            ),
        },
        "category_transfer": category_transfer,
        "contrasts": contrasts,
    }


def analyze_animal(
    path: Path,
    *,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> dict[str, Any]:
    seams = internal_seams()
    strips = {seam: seam_strip_bins(seam) for seam in seams}
    with Mat73Animal(path) as animal:
        environment = [
            animal.environment(session)
            for session in range(animal.n_sessions)
        ]
        blocked = [
            animal.blocked(session)
            for session in range(animal.n_sessions)
        ]
        rate = {
            session: animal.stored_rate_maps(
                session,
                smoothed=False,
            )
            for session in range(animal.n_sessions)
        }
        occupancy = {
            session: animal.sampling_map(session)
            for session in range(animal.n_sessions)
        }
        registered = {
            session: animal.registered_cells(session)
            for session in range(animal.n_sessions)
        }
        repetitions = (animal.n_sessions - 1) // 10

    queries = []
    for training_exposure in range(repetitions - 1):
        training_start = training_exposure * 10
        test_start = (training_exposure + 1) * 10
        for target_offset in range(1, 10):
            target_test = test_start + target_offset
            for target_seam in seams:
                if (
                    seam_state(blocked[target_test], target_seam)
                    is not SeamState.WALL
                ):
                    continue
                query_cells = transfer._query_common_cells(
                    training_start=training_start,
                    test_start=test_start,
                    target_offset=target_offset,
                    registered=registered,
                )
                if query_cells.size < minimum_cells:
                    continue
                records = []
                for source_seam in seams:
                    spatial = spatial_relation(
                        target_seam,
                        source_seam,
                    )
                    if spatial is None:
                        continue
                    record = transfer._source_record(
                        training_start=training_start,
                        test_start=test_start,
                        target_offset=target_offset,
                        target_seam=target_seam,
                        source_seam=source_seam,
                        blocked=blocked,
                        rate=rate,
                        occupancy=occupancy,
                        registered=registered,
                        query_cells=query_cells,
                        strips=strips,
                        minimum_seconds=minimum_seconds,
                        minimum_bins=minimum_bins,
                        minimum_cells=minimum_cells,
                    )
                    if record is not None:
                        records.append(
                            {**record, "spatial": spatial}
                        )
                summary = _query_summary(records)
                if summary is None:
                    continue
                queries.append(
                    {
                        "training_exposure": training_exposure + 1,
                        "test_exposure": training_exposure + 2,
                        "target_environment": environment[target_test],
                        "target_seam": [
                            target_seam.source,
                            target_seam.target,
                        ],
                        "query_common_cells": int(query_cells.size),
                        **summary,
                    }
                )
    if not queries:
        raise ValueError(f"{path.name} has no eligible spatial queries")
    return {
        "animal": path.name.removesuffix(".complete.mat"),
        "modes": {
            mode: _animal_summary(queries, mode=mode)
            for mode in transfer.RATE_MODES
        },
        "queries": queries,
    }


def _cohort_summary(
    animals: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for mode in transfer.RATE_MODES:
        labels = tuple(
            animals[0]["modes"][mode]["contrasts"]
        )
        mode_result = {}
        primary_values = {
            animal["animal"]: animal["modes"][mode][
                "same_all_transfer"
            ]["mean"]
            for animal in animals
        }
        mode_result["same_all_transfer"] = {
            "animals": len(primary_values),
            "positive_animals": int(
                np.count_nonzero(
                    np.asarray(list(primary_values.values())) > 0
                )
            ),
            "animal_mean": float(
                np.mean(list(primary_values.values()))
            ),
            "animal_values": primary_values,
        }
        for category in CATEGORIES:
            category_values = {
                animal["animal"]: animal["modes"][mode][
                    "category_transfer"
                ][category]["mean"]
                for animal in animals
                if animal["modes"][mode]["category_transfer"][
                    category
                ]["mean"]
                is not None
            }
            category_array = np.asarray(
                list(category_values.values())
            )
            mode_result[f"{category}_transfer"] = {
                "animals": len(category_values),
                "positive_animals": int(
                    np.count_nonzero(category_array > 0)
                ),
                "animal_mean": (
                    float(np.mean(category_array))
                    if category_array.size
                    else None
                ),
                "animal_values": category_values,
            }
        for label in labels:
            values = {
                animal["animal"]: animal["modes"][mode][
                    "contrasts"
                ][label]["mean"]
                for animal in animals
                if animal["modes"][mode]["contrasts"][label]["mean"]
                is not None
            }
            array = np.asarray(list(values.values()))
            mode_result[label] = {
                "animals": len(values),
                "positive_animals": int(
                    np.count_nonzero(array > 0)
                ),
                "animal_mean": (
                    float(np.mean(array)) if array.size else None
                ),
                "animal_values": values,
            }
        result[mode] = mode_result
    return result


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
        animal = analyze_animal(
            path,
            minimum_seconds=argument.minimum_seconds,
            minimum_bins=argument.minimum_bins,
            minimum_cells=argument.minimum_cells,
        )
        animals.append(animal)
        print(
            animal["animal"],
            {
                mode: animal["modes"][mode]["contrasts"][
                    "same_tangential_minus_opposite_tangential_transfer"
                ]["mean"]
                for mode in transfer.RATE_MODES
            },
        )
    report = {
        "status": "cross_location_spatial_overlap_control",
        "question": (
            "Can exact-25-cm cross-location transfer be explained by strip "
            "overlap, translation axis, or generic nearby-place profiles?"
        ),
        "design": {
            "query_common_registered_cells": True,
            "target_training_geometry_neural_rates_excluded": True,
            "predictor_evaluation_neural_sessions_disjoint": True,
            "exact_midpoint_distance_cm": 25.0,
            "translation_split": (
                "midpoint displacement resolved into target-wall normal "
                "and tangent components"
            ),
            "generic_place_controls": [
                "source effect versus target open profile",
                "source open versus target open profile",
                "source open versus held-out target residual",
            ],
            "aggregation": (
                "source seams within target query, target queries within "
                "animal, animals descriptively across cohort"
            ),
        },
        "settings": {
            "minimum_seconds_per_common_bin": argument.minimum_seconds,
            "minimum_common_bins": argument.minimum_bins,
            "minimum_common_cells": argument.minimum_cells,
        },
        "geometry_census": geometry_census(),
        "cohort_descriptive": _cohort_summary(animals),
        "animals": animals,
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["cohort_descriptive"], indent=2))


if __name__ == "__main__":
    main()
