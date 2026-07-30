"""Make the evidence chain for exact-location CA1 wall-conditioned profiles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fragment",
        type=Path,
        default=(
            ROOT / "results" / "source_data" / "boundary_fragment_screen.json"
        ),
    )
    parser.add_argument(
        "--heldout",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "boundary_component_validation.json"
        ),
    )
    parser.add_argument(
        "--raw-split",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "boundary_fragment_raw_split.json"
        ),
    )
    parser.add_argument(
        "--behavior",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "boundary_fragment_behavior_adjusted.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "figures"
            / "boundary_component_discovery.png"
        ),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "boundary_component_figure.csv"
        ),
    )
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _animal_lookup(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {record["animal"]: record for record in records}


def _short(animal: str) -> str:
    return animal.rsplit("-", maxsplit=1)[-1]


def _plot_panel(
    axis: plt.Axes,
    values: np.ndarray,
    eligible: np.ndarray,
    animals: list[str],
    colors: list[tuple[float, float, float, float]],
    *,
    title: str,
    subtitle: str,
) -> None:
    jitter = np.linspace(-0.22, 0.22, len(animals))
    axis.axhline(0.0, color="#a8adb4", linewidth=1.0, zorder=0)
    for index, (x_value, y_value) in enumerate(zip(jitter, values)):
        face = colors[index] if eligible[index] else "white"
        axis.scatter(
            x_value,
            y_value,
            s=54,
            facecolor=face,
            edgecolor=colors[index],
            linewidth=1.5,
            zorder=3,
        )
        axis.annotate(
            _short(animals[index]),
            (x_value, y_value),
            xytext=(0, 7 if y_value >= 0 else -11),
            textcoords="offset points",
            ha="center",
            va="bottom" if y_value >= 0 else "top",
            fontsize=7,
            color="#31363b",
        )
    finite_eligible = np.isfinite(values) & eligible
    if np.any(finite_eligible):
        mean = float(np.mean(values[finite_eligible]))
        axis.plot(
            [-0.29, 0.29],
            [mean, mean],
            color="#111111",
            linewidth=2.2,
            solid_capstyle="round",
            zorder=2,
        )
    axis.set_xlim(-0.34, 0.34)
    axis.set_xticks([])
    axis.set_title(title, fontsize=10.5, fontweight="bold", pad=13)
    axis.text(
        0.5,
        1.01,
        subtitle,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#596168",
    )
    axis.spines[["top", "right", "bottom"]].set_visible(False)
    axis.grid(axis="y", color="#e6e8eb", linewidth=0.7)
    axis.set_axisbelow(True)


def main() -> None:
    argument = parse_arguments()
    fragment = _read(argument.fragment)
    heldout = _read(argument.heldout)
    raw_split = _read(argument.raw_split)
    behavior = _read(argument.behavior)

    fragment_by_animal = _animal_lookup(fragment["animals"])
    raw_by_animal = _animal_lookup(raw_split["animals"])
    behavior_by_animal = _animal_lookup(behavior["animals"])
    animals = list(fragment_by_animal)

    full_value = np.asarray(
        [fragment_by_animal[item]["all_sequence_mean"] for item in animals],
        dtype=np.float64,
    )
    full_eligible = np.asarray(
        [
            fragment_by_animal[item]["coverage_eligible_sequences"] > 0
            for item in animals
        ],
        dtype=bool,
    )
    heldout_values = heldout["cohort"]["global_rate_demeaned"][
        "animal_values"
    ]
    heldout_value = np.asarray(
        [heldout_values[item] for item in animals],
        dtype=np.float64,
    )
    raw_value = np.asarray(
        [raw_by_animal[item]["primary"]["all_sequence_mean"] for item in animals],
        dtype=np.float64,
    )
    raw_eligible = np.asarray(
        [
            raw_by_animal[item]["primary"]["eligible_sequences"] > 0
            for item in animals
        ],
        dtype=bool,
    )
    behavior_primary = behavior["cohort"]["modes"][
        "speed_movement_direction_time"
    ][
        "animal_values"
    ]
    behavior_no_time = behavior["cohort"]["modes"][
        "speed_movement_direction"
    ][
        "animal_values"
    ]
    behavior_value = np.asarray(
        [behavior_primary[item] for item in animals],
        dtype=np.float64,
    )

    table_rows = []
    for index, animal in enumerate(animals):
        table_rows.append(
            {
                "animal": animal,
                "same_cycle_full_session_strip_delta_r": full_value[index],
                "same_cycle_full_session_strip_coverage_eligible": bool(
                    full_eligible[index]
                ),
                "target_rate_heldout_global_rate_demeaned_delta_r": (
                    heldout_value[index]
                ),
                "same_cycle_raw_split_delta_r": raw_value[index],
                "same_cycle_raw_split_coverage_eligible": bool(
                    raw_eligible[index]
                ),
                "target_rate_heldout_behavior_adjusted_delta_r": (
                    behavior_value[index]
                ),
                "target_rate_heldout_behavior_adjusted_no_time_delta_r": (
                    behavior_no_time[animal]
                ),
            }
        )
    argument.table.parent.mkdir(parents=True, exist_ok=True)
    with argument.table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    colors = list(plt.get_cmap("tab10").colors[: len(animals)])
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(10.4, 3.65),
        sharey=True,
        constrained_layout=True,
    )
    panels = (
        (
            full_value,
            full_eligible,
            "Full-session strips",
            "same cycle; fixed shape pair",
        ),
        (
            heldout_value,
            np.ones(len(animals), dtype=bool),
            "Target rates held out",
            "local vector; next cycle",
        ),
        (
            raw_value,
            raw_eligible,
            "Non-overlapping raw halves",
            "crossed temporal halves",
        ),
        (
            behavior_value,
            np.ones(len(animals), dtype=bool),
            "Kinematics adjusted",
            "additive; common reference",
        ),
    )
    for axis, panel in zip(axes, panels):
        _plot_panel(
            axis,
            panel[0],
            panel[1],
            animals,
            colors,
            title=panel[2],
            subtitle=panel[3],
        )
    axes[0].set_ylabel(
        r"Walled-profile advantage  $\Delta$ Spearman $r$",
        fontsize=9.5,
    )
    finite = np.concatenate(
        [value[np.isfinite(value)] for value, *_ in panels]
    )
    lower = min(-0.10, float(np.min(finite)) - 0.035)
    upper = max(0.38, float(np.max(finite)) + 0.05)
    axes[0].set_ylim(lower, upper)
    figure.suptitle(
        "Exact-location wall-conditioned cell profiles in longitudinal CA1",
        fontsize=13,
        fontweight="bold",
    )
    figure.supxlabel(
        (
            "Points are mice; bars use coverage-eligible mice where a gate "
            "applies (panels 1 and 3), otherwise all mice. Hollow points fail "
            "that gate and do not contribute to the corresponding bar."
        ),
        fontsize=7.5,
        color="#4e555b",
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
