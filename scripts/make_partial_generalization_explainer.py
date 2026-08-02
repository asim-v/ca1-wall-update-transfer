"""Create an intuitive, data-linked figure for partial generalization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "source_data"
NAVY = "#183B56"
TEAL = "#147D92"
CORAL = "#D8645D"
GOLD = "#C88A28"
GRID = "#DDE3E6"
TEXT = "#20262B"
MUTED = "#626B72"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "figures"
            / "partial_generalization_explainer.png"
        ),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=SOURCE / "partial_generalization_explainer.csv",
    )
    return parser.parse_args()


def _read(name: str) -> dict[str, Any]:
    return json.loads((SOURCE / name).read_text(encoding="utf-8"))


def _mapping(value: Any) -> dict[str, float]:
    return {animal: float(number) for animal, number in value.items()}


def _clean_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.75)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=8.4, colors=TEXT)


def _draw_schematic(axis: plt.Axes) -> None:
    axis.set_title(
        "A  What cross-location prediction means",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=9,
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    left, bottom, size = 0.03, 0.18, 0.53
    axis.add_patch(
        Rectangle(
            (left, bottom),
            size,
            size,
            facecolor="#F7FAFA",
            edgecolor=NAVY,
            linewidth=1.6,
        )
    )
    for fraction in (1 / 3, 2 / 3):
        axis.plot(
            [left + fraction * size] * 2,
            [bottom, bottom + size],
            color=GRID,
            linewidth=0.9,
        )
        axis.plot(
            [left, left + size],
            [bottom + fraction * size] * 2,
            color=GRID,
            linewidth=0.9,
        )

    source_x = left + size / 3
    target_x = left + 2 * size / 3
    wall_y0 = bottom + size / 3
    wall_y1 = bottom + 2 * size / 3
    axis.plot(
        [source_x, source_x],
        [wall_y0, wall_y1],
        color=TEAL,
        linewidth=7,
        solid_capstyle="round",
    )
    axis.plot(
        [target_x, target_x],
        [wall_y0, wall_y1],
        color=CORAL,
        linewidth=7,
        solid_capstyle="round",
    )
    axis.annotate(
        "",
        xy=(target_x - 0.015, bottom + 0.72 * size),
        xytext=(source_x + 0.015, bottom + 0.72 * size),
        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.4},
    )
    axis.text(
        (source_x + target_x) / 2,
        bottom + 0.76 * size,
        "25 cm",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color=MUTED,
    )
    axis.text(
        source_x,
        wall_y0 - 0.065,
        "SOURCE",
        ha="center",
        fontsize=8.5,
        color=TEAL,
        fontweight="bold",
    )
    axis.text(
        target_x,
        wall_y0 - 0.065,
        "TARGET",
        ha="center",
        fontsize=8.5,
        color=CORAL,
        fontweight="bold",
    )

    pattern_x = np.linspace(0.66, 0.95, 7)
    signs = np.array([1, -1, 1, 1, -1, -1, 1], dtype=float)
    for x_value, sign in zip(pattern_x, signs):
        axis.add_patch(
            Circle(
                (x_value, 0.62),
                0.022,
                facecolor=TEAL if sign > 0 else "white",
                edgecolor=TEAL,
                linewidth=1.2,
            )
        )
        axis.add_patch(
            Circle(
                (x_value, 0.40),
                0.022,
                facecolor=CORAL if sign > 0 else "white",
                edgecolor=CORAL,
                linewidth=1.2,
            )
        )
        axis.plot(
            [x_value, x_value],
            [0.43, 0.59],
            color=GRID,
            linewidth=0.8,
        )
    axis.text(
        0.805,
        0.69,
        "source cell pattern",
        ha="center",
        fontsize=8.5,
        color=TEAL,
        fontweight="bold",
    )
    axis.text(
        0.805,
        0.31,
        "same cells at target",
        ha="center",
        fontsize=8.5,
        color=CORAL,
        fontweight="bold",
    )
    axis.text(
        0.805,
        0.22,
        "schematic: matching signs, weaker prediction",
        ha="center",
        fontsize=7.7,
        color=MUTED,
    )


def _paired_plot(
    axis: plt.Axes,
    left_values: dict[str, float],
    right_values: dict[str, float],
    animals: list[str],
    colors: dict[str, Any],
    *,
    labels: tuple[str, str],
    title: str,
    annotation: str,
    y_limits: tuple[float, float],
) -> None:
    for animal in animals:
        values = [left_values[animal], right_values[animal]]
        axis.plot(
            [0, 1],
            values,
            color=colors[animal],
            alpha=0.42,
            linewidth=1.25,
            zorder=1,
        )
        axis.scatter(
            [0, 1],
            values,
            s=38,
            facecolor=colors[animal],
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
    means = [
        np.mean([left_values[animal] for animal in animals]),
        np.mean([right_values[animal] for animal in animals]),
    ]
    axis.plot([0, 1], means, color=TEXT, linewidth=2.2, zorder=2)
    axis.scatter(
        [0, 1],
        means,
        s=66,
        facecolor="white",
        edgecolor=TEXT,
        linewidth=2.0,
        zorder=4,
    )
    axis.set_xticks([0, 1], labels)
    axis.set_xlim(-0.35, 1.35)
    axis.set_ylim(*y_limits)
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=9)
    axis.text(
        0.5,
        y_limits[1] - 0.015,
        annotation,
        ha="center",
        va="top",
        fontsize=8.2,
        color=MUTED,
    )
    _clean_axis(axis)


def _difference_plot(
    axis: plt.Axes,
    pooled: dict[str, float],
    strict: dict[str, float],
    animals: list[str],
    colors: dict[str, Any],
) -> None:
    groups = [pooled, strict]
    for x_value, group in enumerate(groups):
        eligible = [animal for animal in animals if animal in group]
        offsets = np.linspace(-0.095, 0.095, len(eligible))
        for offset, animal in zip(offsets, eligible):
            axis.scatter(
                x_value + offset,
                group[animal],
                s=38,
                facecolor=colors[animal],
                edgecolor="white",
                linewidth=0.65,
                zorder=3,
            )
        mean = float(np.mean([group[animal] for animal in eligible]))
        axis.plot(
            [x_value - 0.18, x_value + 0.18],
            [mean, mean],
            color=TEXT,
            linewidth=2.5,
            solid_capstyle="round",
            zorder=4,
        )
    axis.axhline(0, color="#8D969D", linewidth=1.0, zorder=0)
    axis.set_xticks(
        [0, 1],
        ["25 cm midpoint\nmatch", "Tangential +\n5 cm strip match"],
    )
    axis.set_xlim(-0.45, 1.45)
    axis.set_ylim(-0.10, 0.28)
    axis.set_title(
        "C  Correct versus wrong wall orientation",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=9,
    )
    axis.text(0, 0.262, "mean +0.101; 7/7 > 0", ha="center", fontsize=8.2)
    axis.text(1, 0.262, "mean +0.040; 3/6 > 0", ha="center", fontsize=8.2)
    axis.set_ylabel("correct minus wrong correlation", fontsize=8.8)
    _clean_axis(axis)


def main() -> None:
    argument = parse_arguments()
    transfer = _read("boundary_fragment_cross_location_transfer.json")
    spatial = _read("boundary_fragment_cross_location_spatial_controls.json")
    mirror = _read("boundary_fragment_cross_location_mirror_open.json")

    transfer_mode = transfer["cohort_descriptive"]["global_rate_demeaned"]
    cross = _mapping(
        transfer_mode[
            "one_grid_step_same_direction_source_effect_r_to_target_residual"
        ]["animal_values"]
    )
    cross_minus_exact = _mapping(
        transfer_mode[
            "one_grid_step_same_direction_minus_exact_location_"
            "effect_r_to_target_residual"
        ]["animal_values"]
    )
    exact = {
        animal: cross[animal] - cross_minus_exact[animal]
        for animal in cross
    }

    spatial_mode = spatial["cohort_descriptive"]["global_rate_demeaned"]
    pooled = _mapping(
        spatial_mode["same_all_minus_opposite_all_transfer"][
            "animal_values"
        ]
    )
    strict = _mapping(
        spatial_mode[
            "same_tangential_minus_opposite_tangential_transfer"
        ]["animal_values"]
    )

    mirror_mode = mirror["cohort_descriptive"]["modes"][
        "global_rate_demeaned"
    ]
    mirror_wall = _mapping(
        mirror_mode["source_effect_r_to_wall_target"]["values_by_animal"]
    )
    mirror_open = _mapping(
        mirror_mode["source_effect_r_to_mirror_open"]["values_by_animal"]
    )

    animals = list(cross)
    palette = plt.get_cmap("tab10")
    colors = {animal: palette(index) for index, animal in enumerate(animals)}

    rows: list[dict[str, str | float]] = []
    for animal in animals:
        rows.append(
            {
                "animal": animal,
                "exact_location_r": exact[animal],
                "cross_location_r": cross[animal],
                "pooled_correct_minus_wrong_r": pooled[animal],
                "strict_tangential_correct_minus_wrong_r": strict.get(
                    animal, ""
                ),
                "mirror_wall_r": mirror_wall.get(animal, ""),
                "mirror_open_r": mirror_open.get(animal, ""),
            }
        )
    argument.table.parent.mkdir(parents=True, exist_ok=True)
    with argument.table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(2, 2, figsize=(12.2, 7.6), dpi=220)
    figure.patch.set_facecolor("white")
    _draw_schematic(axes[0, 0])
    _paired_plot(
        axes[0, 1],
        exact,
        cross,
        animals,
        colors,
        labels=("Same seam", "Different seam\n25 cm away"),
        title="B  Generalization is reliable but smaller",
        annotation="mean 0.230 to 0.148; difference -0.082",
        y_limits=(-0.05, 0.43),
    )
    axes[0, 1].set_ylabel("correlation with later wall response", fontsize=8.8)
    _difference_plot(axes[1, 0], pooled, strict, animals, colors)
    mirror_animals = [animal for animal in animals if animal in mirror_wall]
    _paired_plot(
        axes[1, 1],
        mirror_open,
        mirror_wall,
        mirror_animals,
        colors,
        labels=("Matched open", "Wall target"),
        title="D  Same-session wall versus open target",
        annotation="wall advantage +0.114; 5/5 > 0",
        y_limits=(-0.12, 0.34),
    )
    axes[1, 1].set_ylabel("source-pattern correlation", fontsize=8.8)

    figure.suptitle(
        "Partial transfer: the pattern crosses locations but remains place-bound",
        fontsize=14,
        fontweight="bold",
        color=TEXT,
        y=0.995,
    )
    figure.text(
        0.5,
        0.012,
        "Colored points denote animals; open black circles and black bars denote equal-weight animal means.",
        ha="center",
        fontsize=8.0,
        color=MUTED,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.09,
        top=0.91,
        wspace=0.30,
        hspace=0.42,
    )
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(argument.output, dpi=220, facecolor="white")
    plt.close(figure)
    print(argument.output)
    print(argument.table)


if __name__ == "__main__":
    main()
