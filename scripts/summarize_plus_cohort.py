"""Create a descriptive animal-level summary of the frozen ``+`` analyses.

The three repeated square--plus--square sequences are retained as repeated
exposures.  Cohort-facing summaries first average within animal, so sequences,
queries, wall segments, and cells are never counted as independent biological
replicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ANIMALS = ("QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-56")
ROLE = {
    "QLAK-CA1-08": {
        "label": "exploratory",
        "detail": "exploratory condition-selection animal",
    },
    "QLAK-CA1-30": {
        "label": "preregistered",
        "detail": "untouched confirmatory animal preregistered at 443d038",
    },
    "QLAK-CA1-50": {
        "label": "preregistered",
        "detail": "untouched confirmatory animal preregistered at 443d038",
    },
    "QLAK-CA1-56": {
        "label": "implementation-correction",
        "detail": "replication with documented post-view implementation correction",
    },
}


@dataclass(frozen=True)
class AnalysisSpec:
    key: str
    weighting: str
    bandwidth_key: str
    bandwidth_cm: float
    label: str


SPECS = (
    AnalysisSpec(
        "primary_occupancy_balanced_10cm",
        "occupancy_balanced",
        "10_cm",
        10.0,
        "Primary\nocc.-balanced, 10 cm",
    ),
    AnalysisSpec(
        "localization_sensitivity_occupancy_balanced_7p5cm",
        "occupancy_balanced",
        "7.5_cm",
        7.5,
        "Localization\nocc.-balanced, 7.5 cm",
    ),
    AnalysisSpec(
        "weighting_sensitivity_unweighted_10cm",
        "unweighted",
        "10_cm",
        10.0,
        "Weighting\nunweighted, 10 cm",
    ),
)

GATE_KEYS = (
    "near_query_support",
    "near_segment_support",
    "positive_normal_magnification",
    "positive_normal_minus_tangent",
    "contrast_reliability_above_0_3",
    "target_larger_than_square_null",
    "passes_all",
)

METRIC_KEYS = (
    "target_normal_magnification",
    "target_tangential_change",
    "target_contrast",
    "square_null_contrast",
    "signed_target_minus_null",
    "absolute_null_margin",
    "all_query_contrast_reliability",
    "near_query_contrast_reliability",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def _median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.median(finite)) if finite else None


def _delta(last: float | None, first: float | None) -> float | None:
    if last is None or first is None:
        return None
    return float(last - first)


def _sequence_number(name: str) -> int:
    match = re.fullmatch(r"sequence_(\d+)", name)
    if match is None:
        raise ValueError(f"unrecognized sequence key: {name!r}")
    return int(match.group(1))


def _extract_sequence_row(
    animal: str,
    role: dict[str, str],
    source_role: str,
    stable_cell_count: int,
    sequence_name: str,
    sequence: dict[str, Any],
    spec: AnalysisSpec,
) -> dict[str, Any]:
    result = sequence["weight_modes"][spec.weighting][spec.bandwidth_key]
    distances = np.asarray(result["target"]["distance_cm"], dtype=float)
    near_index = int(np.nanargmin(distances))
    near_distance = float(distances[near_index])
    if not np.isclose(near_distance, 2.5):
        raise ValueError(
            f"{animal} {sequence_name} {spec.key}: expected 2.5 cm near bin, "
            f"found {near_distance:g} cm"
        )

    target_normal = _number(
        result["target"]["normal_magnification"][near_index]
    )
    target_tangent = _number(
        result["target"]["tangential_change"][near_index]
    )
    target_contrast = _number(result["target"]["anisotropy"][near_index])
    null_contrast = _number(
        result["square_pseudo_wall_null"]["anisotropy"][near_index]
    )
    signed_difference = (
        None
        if target_contrast is None or null_contrast is None
        else target_contrast - null_contrast
    )
    absolute_margin = (
        None
        if target_contrast is None or null_contrast is None
        else abs(target_contrast) - abs(null_contrast)
    )
    gate = result["gate"]

    row: dict[str, Any] = {
        "row_unit": "repeated_exposure_descriptive",
        "animal": animal,
        "cohort_role": role["label"],
        "cohort_role_detail": role["detail"],
        "source_selection_provenance": source_role,
        "stable_cell_count": int(stable_cell_count),
        "analysis_spec": spec.key,
        "weighting": spec.weighting,
        "bandwidth_cm": spec.bandwidth_cm,
        "sequence": sequence_name,
        "exposure_index": _sequence_number(sequence_name),
        "sessions_one_based": [
            int(value) for value in sequence["sessions_one_based"]
        ],
        "near_distance_cm": near_distance,
        "target_normal_magnification": target_normal,
        "target_tangential_change": target_tangent,
        "target_contrast": target_contrast,
        "square_null_contrast": null_contrast,
        "signed_target_minus_null": signed_difference,
        "absolute_null_margin": absolute_margin,
        "all_query_contrast_reliability": _number(
            result["target_residual_reliability"]["contrast"]
        ),
        "near_query_contrast_reliability": _number(
            result["near_target_residual_reliability"]["contrast"]
        ),
        "near_valid_queries": int(result["near_valid_queries"]),
        "near_valid_segments": int(result["near_valid_segments"]),
        "minimum_near_queries": int(result["minimum_near_queries"]),
        "minimum_near_segments": int(result["minimum_near_segments"]),
    }
    for key in GATE_KEYS:
        row[f"gate_{key}"] = bool(gate[key])

    expected_absolute_gate = (
        absolute_margin is not None and absolute_margin > 0.0
    )
    if row["gate_target_larger_than_square_null"] != expected_absolute_gate:
        raise ValueError(
            f"{animal} {sequence_name} {spec.key}: pseudo-wall gate does "
            "not match |target| - |null|"
        )
    return row


def _animal_summary(
    animal: str,
    role: dict[str, str],
    source_role: str,
    stable_cell_count: int,
    spec: AnalysisSpec,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["exposure_index"])
    if [row["exposure_index"] for row in ordered] != [1, 2, 3]:
        raise ValueError(f"{animal} {spec.key}: expected exposures 1, 2, 3")

    summary: dict[str, Any] = {
        "row_unit": "animal_summary",
        "animal": animal,
        "cohort_role": role["label"],
        "cohort_role_detail": role["detail"],
        "source_selection_provenance": source_role,
        "stable_cell_count": int(stable_cell_count),
        "analysis_spec": spec.key,
        "weighting": spec.weighting,
        "bandwidth_cm": spec.bandwidth_cm,
        "n_repeated_exposures": len(ordered),
        "sequence_gates_passed": sum(
            bool(row["gate_passes_all"]) for row in ordered
        ),
        "all_sequence_gates_pass": all(
            bool(row["gate_passes_all"]) for row in ordered
        ),
        "mean_near_valid_queries": _mean(
            row["near_valid_queries"] for row in ordered
        ),
        "mean_near_valid_segments": _mean(
            row["near_valid_segments"] for row in ordered
        ),
    }
    for key in METRIC_KEYS:
        summary[f"mean_{key}"] = _mean(row[key] for row in ordered)
        summary[f"endpoint_change_{key}"] = _delta(
            ordered[-1][key], ordered[0][key]
        )
    return summary


def _cohort_descriptives(
    animal_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    strata = (
        ("all_roles_descriptive", lambda row: True),
        (
            "preregistered_only_descriptive",
            lambda row: row["cohort_role"] == "preregistered",
        ),
    )
    for spec in SPECS:
        spec_rows = [
            row
            for row in animal_summaries
            if row["analysis_spec"] == spec.key
        ]
        for stratum, include in strata:
            rows = [row for row in spec_rows if include(row)]
            item: dict[str, Any] = {
                "stratum": stratum,
                "analysis_spec": spec.key,
                "n_animals": len(rows),
                "animals": [row["animal"] for row in rows],
                "unit": "animal_mean",
                "interpretation": (
                    "descriptive aggregation of equal-weight animal means; "
                    "no confidence interval or hypothesis test"
                ),
            }
            for key in METRIC_KEYS:
                values = [row[f"mean_{key}"] for row in rows]
                item[f"mean_of_animal_means_{key}"] = _mean(values)
                item[f"median_of_animal_means_{key}"] = _median(values)
            output.append(item)
    return output


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return "/".join(str(item) for item in value)
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return ""
    return value


def _write_csv(
    path: Path,
    sequence_rows: list[dict[str, Any]],
    animal_summaries: list[dict[str, Any]],
) -> None:
    sequence_columns = [
        "row_unit",
        "animal",
        "cohort_role",
        "cohort_role_detail",
        "source_selection_provenance",
        "stable_cell_count",
        "analysis_spec",
        "weighting",
        "bandwidth_cm",
        "sequence",
        "exposure_index",
        "sessions_one_based",
        "near_distance_cm",
        *METRIC_KEYS,
        "near_valid_queries",
        "near_valid_segments",
        "minimum_near_queries",
        "minimum_near_segments",
        *(f"gate_{key}" for key in GATE_KEYS),
    ]
    summary_columns = [
        "n_repeated_exposures",
        "sequence_gates_passed",
        "all_sequence_gates_pass",
        "mean_near_valid_queries",
        "mean_near_valid_segments",
        *(f"mean_{key}" for key in METRIC_KEYS),
        *(f"endpoint_change_{key}" for key in METRIC_KEYS),
    ]
    columns = list(dict.fromkeys([*sequence_columns, *summary_columns]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in [*sequence_rows, *animal_summaries]:
            writer.writerow(
                {column: _csv_value(row.get(column)) for column in columns}
            )


def _summary_lookup(
    summaries: list[dict[str, Any]], animal: str, spec: str
) -> dict[str, Any]:
    matches = [
        row
        for row in summaries
        if row["animal"] == animal and row["analysis_spec"] == spec
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one summary for {animal} / {spec}")
    return matches[0]


def _plot(
    path: Path,
    sequence_rows: list[dict[str, Any]],
    animal_summaries: list[dict[str, Any]],
) -> None:
    colors = {
        "exploratory": "#D55E00",
        "preregistered": "#0072B2",
        "implementation-correction": "#7B2CBF",
    }
    markers = {
        "QLAK-CA1-08": "o",
        "QLAK-CA1-30": "s",
        "QLAK-CA1-50": "^",
        "QLAK-CA1-56": "D",
    }
    primary_key = SPECS[0].key
    primary_rows = [
        row for row in sequence_rows if row["analysis_spec"] == primary_key
    ]

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    contrast_axis, margin_axis, sensitivity_axis, endpoint_axis = axes.ravel()

    for animal in ANIMALS:
        rows = sorted(
            [row for row in primary_rows if row["animal"] == animal],
            key=lambda row: row["exposure_index"],
        )
        role = rows[0]["cohort_role"]
        color = colors[role]
        marker = markers[animal]
        exposure = np.array([row["exposure_index"] for row in rows])

        for axis, metric in (
            (contrast_axis, "target_contrast"),
            (margin_axis, "absolute_null_margin"),
        ):
            values = np.array([row[metric] for row in rows], dtype=float)
            axis.plot(
                exposure,
                values,
                color=color,
                linewidth=1.4,
                alpha=0.8,
                label=f"{animal} ({role})" if axis is contrast_axis else None,
            )
            for x_value, y_value, row in zip(
                exposure, values, rows, strict=True
            ):
                axis.scatter(
                    x_value,
                    y_value,
                    marker=marker,
                    s=54,
                    edgecolor=color,
                    facecolor=(
                        color if row["gate_passes_all"] else "white"
                    ),
                    linewidth=1.4,
                    zorder=3,
                )

        spec_x = np.arange(len(SPECS))
        spec_values = [
            _summary_lookup(animal_summaries, animal, spec.key)[
                "mean_absolute_null_margin"
            ]
            for spec in SPECS
        ]
        sensitivity_axis.plot(
            spec_x,
            spec_values,
            marker=marker,
            color=color,
            linewidth=1.4,
            markersize=6,
            alpha=0.85,
        )

    for axis in (contrast_axis, margin_axis, sensitivity_axis):
        axis.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
        axis.grid(axis="y", alpha=0.2)

    contrast_axis.set_title("A  Primary target normal − tangent")
    contrast_axis.set_xlabel("Repeated exposure (within animal)")
    contrast_axis.set_ylabel("Cross-fold contrast")
    contrast_axis.set_xticks([1, 2, 3])
    contrast_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=colors[ROLE[animal]["label"]],
                marker=markers[animal],
                markerfacecolor="white",
                markeredgecolor=colors[ROLE[animal]["label"]],
                linewidth=1.4,
                label=f"{animal} ({ROLE[animal]['label']})",
            )
            for animal in ANIMALS
        ],
        fontsize=8,
        frameon=False,
        loc="best",
    )

    margin_axis.set_title("B  Primary pseudo-wall gate margin")
    margin_axis.set_xlabel("Repeated exposure (within animal)")
    margin_axis.set_ylabel("|target contrast| − |square null contrast|")
    margin_axis.set_xticks([1, 2, 3])

    sensitivity_axis.set_title("C  Animal-mean margin by specification")
    sensitivity_axis.set_ylabel("Mean within animal")
    sensitivity_axis.set_xticks(
        np.arange(len(SPECS)), [spec.label for spec in SPECS], fontsize=8
    )

    delta_metrics = (
        ("target_contrast", "Target contrast", "o", "#333333"),
        ("absolute_null_margin", "Gate margin", "s", "#009E73"),
        (
            "all_query_contrast_reliability",
            "Cross-fold reliability",
            "^",
            "#CC79A7",
        ),
    )
    y_base = np.arange(len(ANIMALS), dtype=float)
    offsets = (-0.18, 0.0, 0.18)
    for offset, (metric, label, marker, color) in zip(
        offsets, delta_metrics, strict=True
    ):
        values = [
            _summary_lookup(animal_summaries, animal, primary_key)[
                f"endpoint_change_{metric}"
            ]
            for animal in ANIMALS
        ]
        endpoint_axis.scatter(
            values,
            y_base + offset,
            marker=marker,
            color=color,
            s=46,
            label=label,
        )
    endpoint_axis.axvline(
        0.0, color="#777777", linewidth=0.8, linestyle="--"
    )
    endpoint_axis.grid(axis="x", alpha=0.2)
    endpoint_axis.set_title("D  Primary endpoint change (exposure 3 − 1)")
    endpoint_axis.set_xlabel("Within-animal change")
    endpoint_axis.set_yticks(y_base, ANIMALS, fontsize=8)
    endpoint_axis.invert_yaxis()
    endpoint_axis.legend(fontsize=8, frameon=False)

    figure.suptitle(
        "CA1 + boundary: descriptive animal-level cohort summary", fontsize=14
    )
    figure.text(
        0.5,
        0.015,
        (
            "Lines and sensitivity points are animals; repeated exposures are "
            "not independent replicates. Filled A/B markers passed every "
            "frozen sequence gate. Descriptive only; no pooled inference."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.0, 0.045, 1.0, 0.96))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _format(value: float | None) -> str:
    return "NA" if value is None else f"{value:+.3f}"


def _print_primary_table(summaries: list[dict[str, Any]]) -> None:
    primary = [
        _summary_lookup(summaries, animal, SPECS[0].key)
        for animal in ANIMALS
    ]
    print(
        "Animal-level descriptive summary; repeated sequences are not "
        "independent cohort units."
    )
    print(
        "animal       role                       mean contrast  "
        "mean abs margin  gates  delta contrast  delta reliability"
    )
    for row in primary:
        print(
            f"{row['animal']:<12} "
            f"{row['cohort_role']:<26} "
            f"{_format(row['mean_target_contrast']):>13}  "
            f"{_format(row['mean_absolute_null_margin']):>12}  "
            f"{row['sequence_gates_passed']}/"
            f"{row['n_repeated_exposures']:<3}  "
            f"{_format(row['endpoint_change_target_contrast']):>14}  "
            f"{_format(row['endpoint_change_all_query_contrast_reliability']):>17}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=Path("results/diagnostics"),
        help="directory containing QLAK-CA1-*_plus_boundary.json",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/diagnostics/plus_cohort_summary"),
        help="output stem; .json, .csv, and .png are appended",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_rows: list[dict[str, Any]] = []
    animal_summaries: list[dict[str, Any]] = []
    animals_output = []
    source_files = []

    for animal in ANIMALS:
        path = args.diagnostics_dir / f"{animal}_plus_boundary.json"
        with path.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        if source["animal"] != animal:
            raise ValueError(
                f"{path}: expected animal {animal}, found {source['animal']}"
            )
        if source["condition_environment"] != "+":
            raise ValueError(f"{path}: expected '+' condition")

        source_role = str(source["selection_provenance"])
        role = ROLE[animal]
        stable_cell_count = int(source["stable_cell_count"])
        sequence_items = sorted(
            source["sequences"].items(),
            key=lambda item: _sequence_number(item[0]),
        )
        animal_rows = []
        animal_summary_rows = []
        for spec in SPECS:
            spec_rows = [
                _extract_sequence_row(
                    animal,
                    role,
                    source_role,
                    stable_cell_count,
                    sequence_name,
                    sequence,
                    spec,
                )
                for sequence_name, sequence in sequence_items
            ]
            animal_rows.extend(spec_rows)
            animal_summary_rows.append(
                _animal_summary(
                    animal,
                    role,
                    source_role,
                    stable_cell_count,
                    spec,
                    spec_rows,
                )
            )

        primary_rows = [
            row
            for row in animal_rows
            if row["analysis_spec"] == SPECS[0].key
        ]
        extracted_passes = sum(row["gate_passes_all"] for row in primary_rows)
        if extracted_passes != int(source["primary_gate_sequences_passed"]):
            raise ValueError(f"{path}: top-level primary gate count mismatch")
        if all(row["gate_passes_all"] for row in primary_rows) != bool(
            source["all_primary_sequence_gates_pass"]
        ):
            raise ValueError(f"{path}: top-level all-gates flag mismatch")

        sequence_rows.extend(animal_rows)
        animal_summaries.extend(animal_summary_rows)
        source_files.append(
            {
                "animal": animal,
                "path": path.as_posix(),
                "sha256": _sha256(path),
            }
        )
        animals_output.append(
            {
                "animal": animal,
                "cohort_role": role["label"],
                "cohort_role_detail": role["detail"],
                "source_selection_provenance": source_role,
                "stable_cell_count": stable_cell_count,
                "repeated_exposures": [
                    {
                        "sequence": name,
                        "sessions_one_based": value["sessions_one_based"],
                    }
                    for name, value in sequence_items
                ],
                "sequence_results": animal_rows,
                "animal_summaries": animal_summary_rows,
            }
        )

    output = {
        "status": "descriptive_animal_level_summary_no_inferential_pooling",
        "generated_by": "scripts/summarize_plus_cohort.py",
        "independent_biological_unit": "animal",
        "nonindependent_within_animal_units": [
            "repeated sequences",
            "queries",
            "wall segments",
            "registered cells",
        ],
        "endpoint_change_definition": "exposure_3_minus_exposure_1",
        "near_distance_cm": 2.5,
        "analysis_specs": [
            {
                "key": spec.key,
                "weighting": spec.weighting,
                "bandwidth_cm": spec.bandwidth_cm,
            }
            for spec in SPECS
        ],
        "source_files": source_files,
        "animals": animals_output,
        "cohort_descriptives": _cohort_descriptives(animal_summaries),
    }

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    figure_path = prefix.with_suffix(".png")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    _write_csv(csv_path, sequence_rows, animal_summaries)
    _plot(figure_path, sequence_rows, animal_summaries)

    _print_primary_table(animal_summaries)
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"PNG:  {figure_path}")


if __name__ == "__main__":
    main()
