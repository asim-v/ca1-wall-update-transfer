"""Plot the frozen empirical spatial-alternative adjudication.

The figure reads only the tracked machine-readable result and exports the
animal-level plotting table used for the manuscript.  It does not rerun the
neural analysis.
"""

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
RESULT = ROOT / "results" / "positive_model_spatial_rank_v1.json"
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
            / "positive_model_spatial_rank_v1.png"
        ),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "positive_model_spatial_rank_v1_figure.csv"
        ),
    )
    return parser.parse_args()


def _metric(
    data: dict[str, Any],
    animal: str,
    tier: str,
    metric: str,
) -> float | None:
    result = data["cohort"][tier]["global_rate_demeaned"]
    values = result[metric]["animal_values"]
    if animal not in values:
        return None
    return float(values[animal])


def _rows(data: dict[str, Any]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for animal_result in data["animals"]:
        animal = str(animal_result["animal"])
        tier1 = animal_result["summaries"]["tier1_exact_25cm"][
            "global_rate_demeaned"
        ]
        tier2 = animal_result["summaries"]["tier2_tangential"][
            "global_rate_demeaned"
        ]
        centered_rank = _metric(
            data,
            animal,
            "tier1_exact_25cm",
            "centered_percentile_rank",
        )
        assert centered_rank is not None
        rows.append(
            {
                "animal": animal,
                "tier1_percentile_rank": centered_rank + 0.5,
                "tier1_correct_minus_mean_alternative_r": _metric(
                    data,
                    animal,
                    "tier1_exact_25cm",
                    "correct_minus_mean_alternative",
                ),
                "tier1_correct_minus_best_alternative_r": _metric(
                    data,
                    animal,
                    "tier1_exact_25cm",
                    "correct_minus_best_alternative",
                ),
                "tier1_scored_queries": int(tier1["eligible_queries"]),
                "tier1_exposure_pairs": int(
                    tier1["eligible_exposure_pairs"]
                ),
                "tier2_centered_percentile_rank": _metric(
                    data,
                    animal,
                    "tier2_tangential",
                    "centered_percentile_rank",
                ),
                "tier2_scored_queries": (
                    "" if tier2 is None else int(tier2["eligible_queries"])
                ),
                "tier2_exposure_pairs": (
                    ""
                    if tier2 is None
                    else int(tier2["eligible_exposure_pairs"])
                ),
            }
        )
    return rows


def _clean(axis: plt.Axes, *, xgrid: bool = False) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(
        axis="x" if xgrid else "y",
        color=GRID,
        linewidth=0.75,
    )
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=8.2, colors=TEXT)


def _panel_title(axis: plt.Axes, title: str) -> None:
    axis.set_title(
        title,
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        color=TEXT,
        pad=9,
    )


def _draw_enumeration(axis: plt.Axes) -> None:
    _panel_title(axis, "A  Every admissible source was scored")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    left, bottom, size = 0.05, 0.17, 0.55
    axis.add_patch(
        Rectangle(
            (left, bottom),
            size,
            size,
            facecolor="#F7FAFA",
            edgecolor=NAVY,
            linewidth=1.5,
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

    target = (left + 0.50 * size, bottom + 0.50 * size)
    axis.add_patch(
        Circle(target, 0.035, facecolor=CORAL, edgecolor="white", linewidth=1)
    )
    axis.text(
        target[0],
        target[1] - 0.095,
        "target",
        ha="center",
        fontsize=8,
        color=CORAL,
        fontweight="bold",
    )

    positions = [
        (left + 0.17 * size, bottom + 0.50 * size),
        (left + 0.83 * size, bottom + 0.50 * size),
        (left + 0.50 * size, bottom + 0.17 * size),
        (left + 0.50 * size, bottom + 0.83 * size),
        (left + 0.17 * size, bottom + 0.17 * size),
        (left + 0.83 * size, bottom + 0.83 * size),
    ]
    correct = {1, 3}
    for index, point in enumerate(positions):
        color = TEAL if index in correct else MUTED
        marker = "o" if index in correct else "s"
        axis.scatter(
            [point[0]],
            [point[1]],
            s=58,
            marker=marker,
            facecolor=color if index in correct else "white",
            edgecolor=color,
            linewidth=1.2,
            zorder=3,
        )
        axis.plot(
            [target[0], point[0]],
            [target[1], point[1]],
            color=GRID,
            linewidth=0.8,
            zorder=1,
        )

    axis.text(
        0.67,
        0.69,
        "Correct relation",
        color=TEAL,
        fontsize=8.6,
        fontweight="bold",
    )
    axis.scatter(
        [0.64], [0.70], s=42, marker="o", facecolor=TEAL, edgecolor=TEAL
    )
    axis.text(0.67, 0.58, "Other exact-distance\nsources", color=MUTED, fontsize=8.6)
    axis.scatter(
        [0.64], [0.61], s=42, marker="s", facecolor="white", edgecolor=MUTED
    )
    axis.text(
        0.64,
        0.37,
        "Matched geometry + support\nNo neural selection",
        color=TEXT,
        fontsize=8.3,
        va="top",
    )
    axis.text(
        0.05,
        0.06,
        "415 enumerated queries | 405 primary queries | 7 animals",
        color=MUTED,
        fontsize=8.1,
    )


def _draw_rank(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    _panel_title(axis, "B  Correct-relation percentile rank")
    values = np.asarray([row["tier1_percentile_rank"] for row in rows])
    x = np.arange(values.size)
    axis.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
    axis.bar(x, values - 0.5, bottom=0.5, color=TEAL, width=0.62)
    axis.scatter(x, values, s=23, color=NAVY, zorder=3)
    axis.axhline(values.mean(), color=CORAL, linewidth=1.4)
    axis.text(
        values.size - 0.55,
        values.mean() + 0.012,
        f"mean {values.mean():.3f}",
        ha="right",
        fontsize=8.2,
        color=CORAL,
        fontweight="bold",
    )
    axis.set_ylim(0.45, 0.82)
    axis.set_ylabel("percentile among admissible sources", fontsize=8.5)
    axis.set_xticks(x, [str(index + 1) for index in x])
    axis.set_xlabel("animal", fontsize=8.5)
    axis.text(
        0.02,
        0.97,
        "7/7 above 0.5\nexact tail = 1/128",
        transform=axis.transAxes,
        va="top",
        fontsize=8.1,
        color=TEXT,
    )
    _clean(axis)


def _draw_alternatives(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    _panel_title(axis, "C  Average versus strongest alternative")
    mean_alt = np.asarray(
        [row["tier1_correct_minus_mean_alternative_r"] for row in rows]
    )
    best_alt = np.asarray(
        [row["tier1_correct_minus_best_alternative_r"] for row in rows]
    )
    for first, second in zip(mean_alt, best_alt):
        axis.plot([0, 1], [first, second], color=GRID, linewidth=1.2, zorder=1)
    axis.scatter(
        np.zeros(mean_alt.size),
        mean_alt,
        s=32,
        color=TEAL,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )
    axis.scatter(
        np.ones(best_alt.size),
        best_alt,
        s=32,
        marker="s",
        color=CORAL,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )
    axis.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    axis.plot([-0.09, 0.09], [mean_alt.mean()] * 2, color=NAVY, linewidth=2.4)
    axis.plot([0.91, 1.09], [best_alt.mean()] * 2, color=NAVY, linewidth=2.4)
    axis.text(-0.12, mean_alt.mean() + 0.018, "7/7", fontsize=8.2, color=TEAL)
    axis.text(0.88, best_alt.mean() + 0.018, "5/7", fontsize=8.2, color=CORAL)
    axis.set_xlim(-0.35, 1.35)
    axis.set_ylim(-0.20, 0.30)
    axis.set_xticks([0, 1], ["correct -\nmean alternative", "correct -\nbest alternative"])
    axis.set_ylabel("difference in prediction correlation", fontsize=8.5)
    _clean(axis)


def _draw_tiers(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    _panel_title(axis, "D  Strict tangential subset remains limited")
    full = np.asarray([row["tier1_percentile_rank"] - 0.5 for row in rows])
    paired = [
        (float(row["tier1_percentile_rank"]) - 0.5, float(row["tier2_centered_percentile_rank"]))
        for row in rows
        if row["tier2_centered_percentile_rank"] not in (None, "")
    ]
    strict = np.asarray([value[1] for value in paired])
    full_paired = np.asarray([value[0] for value in paired])
    for first, second in paired:
        axis.plot([0, 1], [first, second], color=GRID, linewidth=1.2)
    axis.scatter(np.zeros(full_paired.size), full_paired, s=31, color=TEAL, zorder=2)
    axis.scatter(np.ones(strict.size), strict, s=31, marker="s", color=GOLD, zorder=2)
    axis.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    axis.plot([-0.09, 0.09], [full.mean()] * 2, color=NAVY, linewidth=2.4)
    axis.plot([0.91, 1.09], [strict.mean()] * 2, color=NAVY, linewidth=2.4)
    axis.text(-0.12, full.mean() + 0.018, "7/7", fontsize=8.2, color=TEAL)
    axis.text(0.88, strict.mean() + 0.018, "4/6", fontsize=8.2, color=GOLD)
    axis.set_xlim(-0.35, 1.35)
    axis.set_ylim(-0.20, 0.30)
    axis.set_xticks([0, 1], ["all exact-distance\nsources", "tangential\nsources only"])
    axis.set_ylabel("percentile rank minus 0.5", fontsize=8.5)
    _clean(axis)


def main() -> None:
    argument = parse_arguments()
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    rows = _rows(data)
    if len(rows) != 7:
        raise AssertionError("expected seven animal-level rows")

    argument.table.parent.mkdir(parents=True, exist_ok=True)
    with argument.table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.3))
    _draw_enumeration(axes[0, 0])
    _draw_rank(axes[0, 1], rows)
    _draw_alternatives(axes[1, 0], rows)
    _draw_tiers(axes[1, 1], rows)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.95, bottom=0.10, hspace=0.42, wspace=0.30)

    argument.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(argument.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
