"""Plot the evidence and claim boundary for local CA1 wall-update reuse."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "source_data"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "figures"
            / "local_wall_update_transfer.png"
        ),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=SOURCE / "local_wall_update_transfer_figure.csv",
    )
    return parser.parse_args()


def _read(name: str) -> dict[str, Any]:
    return json.loads((SOURCE / name).read_text(encoding="utf-8"))


def _animal_values(
    artifact: dict[str, Any],
    *keys: str,
) -> dict[str, float]:
    value: Any = artifact
    for key in keys:
        value = value[key]
    return {animal: float(number) for animal, number in value.items()}


def _plot_paired_panel(
    axis: plt.Axes,
    values: np.ndarray,
    labels: tuple[str, ...],
    animals: list[str],
    colors: list[tuple[float, float, float, float]],
    *,
    title: str,
    annotations: tuple[str, ...],
) -> None:
    x = np.arange(values.shape[1], dtype=np.float64)
    axis.axhline(0.0, color="#9ca3aa", linewidth=1.0, zorder=0)
    for row, color in zip(values, colors):
        axis.plot(
            x,
            row,
            color=color,
            alpha=0.32,
            linewidth=1.0,
            zorder=1,
        )
        axis.scatter(
            x,
            row,
            s=35,
            facecolor=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
    means = np.mean(values, axis=0)
    for x_value, mean in zip(x, means):
        axis.plot(
            [x_value - 0.18, x_value + 0.18],
            [mean, mean],
            color="#111111",
            linewidth=2.3,
            solid_capstyle="round",
            zorder=4,
        )
    for x_value, label in zip(x, annotations):
        axis.text(
            x_value,
            0.565,
            label,
            ha="center",
            va="top",
            fontsize=7.3,
            color="#444b52",
        )
    axis.set_xticks(x, labels, fontsize=8)
    axis.set_xlim(-0.42, len(labels) - 0.58)
    axis.set_ylim(-0.25, 0.60)
    axis.set_title(title, fontsize=10.5, fontweight="bold", pad=10)
    axis.grid(axis="y", color="#e5e7e9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    argument = parse_arguments()
    context = _read("boundary_fragment_context_matched.json")
    single = _read("boundary_fragment_single_tile_counterfactual.json")
    behavior = _read(
        "boundary_fragment_single_tile_counterfactual_behavior_adjusted.json"
    )
    near = _read(
        "boundary_fragment_single_tile_counterfactual_near.json"
    )
    far = _read(
        "boundary_fragment_single_tile_counterfactual_far.json"
    )
    transfer = _read("boundary_fragment_cross_location_transfer.json")
    transfer_behavior = _read(
        "boundary_fragment_cross_location_behavior_adjusted.json"
    )
    mirror = _read("boundary_fragment_cross_location_mirror_open.json")
    spatial = _read(
        "boundary_fragment_cross_location_spatial_controls.json"
    )

    context_values = _animal_values(
        context,
        "cohort",
        "global_rate_demeaned",
        "animal_values",
    )
    single_values = _animal_values(
        single,
        "cohort",
        "global_rate_demeaned",
        "animal_values",
    )
    behavior_values = _animal_values(
        behavior,
        "cohort",
        "modes",
        "speed_movement_direction_time",
        "animal_values",
    )
    near_values = _animal_values(
        near,
        "cohort",
        "global_rate_demeaned",
        "animal_values",
    )
    far_values = _animal_values(
        far,
        "cohort",
        "global_rate_demeaned",
        "animal_values",
    )
    translated_values = _animal_values(
        transfer,
        "cohort_descriptive",
        "global_rate_demeaned",
        "one_grid_step_same_direction_source_effect_r_to_target_residual",
        "animal_values",
    )
    translated_minus_exact = _animal_values(
        transfer,
        "cohort_descriptive",
        "global_rate_demeaned",
        (
            "one_grid_step_same_direction_minus_exact_location_"
            "effect_r_to_target_residual"
        ),
        "animal_values",
    )
    translated_behavior_values = _animal_values(
        transfer_behavior,
        "cohort_descriptive",
        "speed_movement_direction_time",
        "source_effect_r_to_target_residual",
        "animal_values",
    )
    specificity = _animal_values(
        spatial,
        "cohort_descriptive",
        "global_rate_demeaned",
        "same_all_effect_specificity_over_target_open",
        "animal_values",
    )
    mirror_mode = mirror["cohort_descriptive"]["modes"][
        "global_rate_demeaned"
    ]
    mirror_wall_values = {
        animal: float(value)
        for animal, value in mirror_mode[
            "source_effect_r_to_wall_target"
        ]["values_by_animal"].items()
    }
    mirror_open_values = {
        animal: float(value)
        for animal, value in mirror_mode[
            "source_effect_r_to_mirror_open"
        ]["values_by_animal"].items()
    }
    mirror_advantage_values = {
        animal: float(value)
        for animal, value in mirror_mode[
            "wall_minus_mirror_open_correlation_advantage"
        ]["values_by_animal"].items()
    }

    animals = list(single_values)
    expected = set(animals)
    sources = (
        context_values,
        behavior_values,
        near_values,
        far_values,
        translated_values,
        translated_minus_exact,
        translated_behavior_values,
        specificity,
    )
    if any(set(values) != expected for values in sources):
        raise ValueError("animal sets differ across source artifacts")
    mirror_animals = [
        animal for animal in animals if animal in mirror_wall_values
    ]
    mirror_sets = (
        set(mirror_wall_values),
        set(mirror_open_values),
        set(mirror_advantage_values),
    )
    if (
        len({frozenset(values) for values in mirror_sets}) != 1
        or not mirror_sets[0].issubset(expected)
    ):
        raise ValueError("mirror-control animal sets are inconsistent")

    rows: list[dict[str, Any]] = []
    for animal in animals:
        translated = translated_values[animal]
        exact = translated - translated_minus_exact[animal]
        target_open = translated - specificity[animal]
        rows.append(
            {
                "animal": animal,
                "context_matched_wall_minus_open_delta_r": (
                    context_values[animal]
                ),
                "single_tile_wall_minus_open_delta_r": single_values[animal],
                "single_tile_behavior_adjusted_delta_r": (
                    behavior_values[animal]
                ),
                "single_tile_near_delta_r": near_values[animal],
                "single_tile_far_delta_r": far_values[animal],
                "exact_location_effect_r_to_target_residual": exact,
                "translated_effect_r_to_target_residual": translated,
                "translated_behavior_adjusted_r_to_target_residual": (
                    translated_behavior_values[animal]
                ),
                "translated_effect_r_to_target_open_profile": target_open,
                "translated_effect_specificity_delta_r": specificity[animal],
                "mirror_control_wall_r": mirror_wall_values.get(animal, ""),
                "mirror_control_open_r": mirror_open_values.get(animal, ""),
                "mirror_control_advantage_delta_r": (
                    mirror_advantage_values.get(animal, "")
                ),
            }
        )

    argument.table.parent.mkdir(parents=True, exist_ok=True)
    with argument.table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    first = np.asarray(
        [
            [
                row["context_matched_wall_minus_open_delta_r"],
                row["single_tile_wall_minus_open_delta_r"],
                row["single_tile_behavior_adjusted_delta_r"],
            ]
            for row in rows
        ]
    )
    second = np.asarray(
        [
            [
                row["exact_location_effect_r_to_target_residual"],
                row["translated_effect_r_to_target_residual"],
                row[
                    "translated_behavior_adjusted_r_to_target_residual"
                ],
                row["translated_effect_r_to_target_open_profile"],
            ]
            for row in rows
        ]
    )
    third = np.asarray(
        [
            [
                mirror_wall_values[animal],
                mirror_open_values[animal],
                mirror_advantage_values[animal],
            ]
            for animal in mirror_animals
        ]
    )
    fourth = np.asarray(
        [
            [
                row["single_tile_near_delta_r"],
                row["single_tile_far_delta_r"],
            ]
            for row in rows
        ]
    )

    colors = list(plt.get_cmap("tab10").colors[: len(animals)])
    color_by_animal = dict(zip(animals, colors))
    mirror_colors = [color_by_animal[animal] for animal in mirror_animals]
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 7.4),
        constrained_layout=True,
    )
    axes = axes.ravel()
    _plot_paired_panel(
        axes[0],
        first,
        ("Local context\nmatched", "One-tile\nedit", "+ behavior\nadjustment"),
        animals,
        colors,
        title="A  Held-out local reuse",
        annotations=("7/7 > 0", "7/7 > 0", "7/7 > 0"),
    )
    _plot_paired_panel(
        axes[1],
        second,
        (
            "Exact\nlocation",
            "Shifted\none step",
            "Shifted +\nbehavior adj.",
            "Effect vs\ntrained open",
        ),
        animals,
        colors,
        title="B  Partial spatial transfer",
        annotations=(
            "benchmark",
            "7/7 > 0",
            "7/7 > 0",
            "target-open ctl.",
        ),
    )
    _plot_paired_panel(
        axes[2],
        third,
        ("Wall target", "Mirror open", "Wall - open"),
        mirror_animals,
        mirror_colors,
        title="C  Same-session mirror control (tangential)",
        annotations=("4/5 > 0", "0/5 > 0", "advantage 5/5"),
    )
    _plot_paired_panel(
        axes[3],
        fourth,
        ("2.5--7.5 cm", "17.5--22.5 cm"),
        animals,
        colors,
        title="D  Localization",
        annotations=("7/7 > 0", "descriptive near > far 7/7"),
    )
    axes[0].set_ylabel("Spearman correlation or correlation advantage")
    axes[2].set_ylabel("Spearman correlation or correlation advantage")
    figure.suptitle(
        "CA1 wall-related remapping: local reuse, limited transport",
        fontsize=13.5,
        fontweight="bold",
    )
    figure.supxlabel(
        (
            "Points are animals; black bars are equal-weight animal means. "
            "General direction selectivity and additive composition did not "
            "survive their decisive controls."
        ),
        fontsize=8,
        color="#4f565d",
    )
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        argument.output,
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(figure)
    print(argument.output)
    print(argument.table)


if __name__ == "__main__":
    main()
