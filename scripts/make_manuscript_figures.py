"""Generate manuscript figures from the tracked source-data files.

The figures deliberately distinguish the consistent aggregate directional
effect from the weaker query-pattern reliability result.  Run from any working
directory:

    python scripts/make_manuscript_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "source_data"
OUTPUT = ROOT / "manuscript" / "figures"

ANIMALS = [
    "QLAK-CA1-08",
    "QLAK-CA1-30",
    "QLAK-CA1-50",
    "QLAK-CA1-56",
]
SHORT = {animal: animal.rsplit("-", 1)[-1] for animal in ANIMALS}
ROLES = {
    "QLAK-CA1-08": "08 (exploratory)",
    "QLAK-CA1-30": "30 (prospectively locked)",
    "QLAK-CA1-50": "50 (prospectively locked)",
    "QLAK-CA1-56": "56 (correction caveat)",
}
COLORS = {
    "QLAK-CA1-08": "#D55E00",
    "QLAK-CA1-30": "#0072B2",
    "QLAK-CA1-50": "#009E73",
    "QLAK-CA1-56": "#CC79A7",
}
MARKERS = {
    "QLAK-CA1-08": "o",
    "QLAK-CA1-30": "s",
    "QLAK-CA1-50": "^",
    "QLAK-CA1-56": "D",
}


def load_json(name: str) -> dict:
    with (SOURCE / name).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def configure() -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / name)
    plt.close(fig)


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "#F4F4F4",
    edgecolor: str = "#4A4A4A",
    fontsize: float = 8.5,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def figure_1() -> None:
    fig = plt.figure(figsize=(12.6, 4.0), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.08, 0.95, 1.35])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    # A: arena and fixed query geometry.
    ax_a.set_aspect("equal")
    ax_a.add_patch(
        Rectangle((0, 0), 75, 75, facecolor="#FAFAFA", edgecolor="black", lw=2)
    )
    for x in (25, 50):
        ax_a.plot([x, x], [0, 75], color="#C9C9C9", lw=0.7, zorder=1)
        ax_a.plot([0, 75], [x, x], color="#C9C9C9", lw=0.7, zorder=1)
    for x, y in ((0, 50), (50, 50), (0, 0), (50, 0)):
        ax_a.add_patch(
            Rectangle(
                (x, y),
                25,
                25,
                facecolor="#9E9E9E",
                edgecolor="#666666",
                hatch="////",
                lw=0.7,
                zorder=2,
            )
        )
    segments = [
        ((0, 50), (25, 50)),
        ((25, 50), (25, 75)),
        ((50, 50), (75, 50)),
        ((50, 50), (50, 75)),
        ((0, 25), (25, 25)),
        ((25, 0), (25, 25)),
        ((50, 25), (75, 25)),
        ((50, 0), (50, 25)),
    ]
    for (x0, y0), (x1, y1) in segments:
        ax_a.plot([x0, x1], [y0, y1], color="#C62828", lw=3.0, zorder=4)

    # The actual 3 x 3 near-wall query grid for one example segment.
    query_x = np.array([10.0, 12.5, 15.0])
    query_y = np.array([47.5, 42.5, 37.5])
    xx, yy = np.meshgrid(query_x, query_y)
    ax_a.scatter(
        xx.ravel(),
        yy.ravel(),
        s=17,
        facecolor="#0072B2",
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
    )
    ax_a.annotate(
        r"$\mathbf{n}$",
        xy=(20, 39),
        xytext=(20, 48),
        arrowprops={"arrowstyle": "-|>", "color": "#0072B2", "lw": 1.4},
        ha="center",
        va="center",
        color="#0072B2",
        fontweight="bold",
    )
    ax_a.annotate(
        r"$\mathbf{t}$",
        xy=(18, 54),
        xytext=(9, 54),
        arrowprops={"arrowstyle": "-|>", "color": "#0072B2", "lw": 1.4},
        ha="center",
        va="center",
        color="#0072B2",
        fontweight="bold",
    )
    ax_a.text(37.5, 37.5, "accessible\nplus-shaped region", ha="center", va="center")
    ax_a.text(12.5, 62.5, "removed\n25 x 25 cm", ha="center", va="center", color="white")
    ax_a.set_xlim(-4, 79)
    ax_a.set_ylim(-4, 79)
    ax_a.set_xticks([0, 25, 50, 75])
    ax_a.set_yticks([0, 25, 50, 75])
    ax_a.set_xlabel("x position (cm)")
    ax_a.set_ylabel("y position (cm)")
    ax_a.set_title("Introduced-wall geometry")
    panel_label(ax_a, "A")

    # B: estimator intuition without treating the cross-product as PSD.
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.axis("off")
    panel_label(ax_b, "B")
    ax_b.set_title("Directional local response change", pad=12)
    center = np.array([0.48, 0.46])
    ax_b.scatter(*center, s=45, color="#333333", zorder=3)
    ax_b.text(center[0] - 0.02, center[1] - 0.08, r"$F(x)$", ha="center")
    arrows = [
        (np.array([0.0, 0.30]), "#D55E00", r"$\Delta F_n$", 2.7),
        (np.array([0.0, -0.18]), "#D55E00", "", 2.7),
        (np.array([0.18, 0.0]), "#0072B2", r"$\Delta F_t$", 1.5),
        (np.array([-0.12, 0.0]), "#0072B2", "", 1.5),
    ]
    for delta, color, label, width in arrows:
        end = center + delta
        ax_b.add_patch(
            FancyArrowPatch(
                center,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                lw=width,
                color=color,
            )
        )
        if label:
            offset = np.array([0.04, 0.0]) if delta[1] else np.array([0.0, 0.05])
            ax_b.text(*(end + offset), label, color=color, ha="center", va="center")
    ax_b.text(
        0.50,
        0.89,
        "equal physical steps",
        ha="center",
        fontsize=9,
        color="#4A4A4A",
    )
    ax_b.text(
        0.50,
        0.13,
        r"$\mathbf{n}^{\mathsf{T}}\Delta\hat g_{\times}\mathbf{n}"
        r" - \mathbf{t}^{\mathsf{T}}\Delta\hat g_{\times}\mathbf{t} > 0$",
        ha="center",
        fontsize=10,
    )
    ax_b.text(
        0.50,
        0.04,
        "cross-fold second-moment contrast",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )

    # C: bracketing and analysis pipeline.
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.axis("off")
    panel_label(ax_c, "C")
    ax_c.set_title("Frozen crossvalidated analysis", pad=12)
    draw_box(ax_c, (0.03, 0.70), 0.24, 0.16, "square\n(pre)")
    draw_box(
        ax_c,
        (0.38, 0.70),
        0.24,
        0.16,
        "plus\ncondition",
        facecolor="#FCE8E6",
        edgecolor="#C62828",
    )
    draw_box(ax_c, (0.73, 0.70), 0.24, 0.16, "square\n(post)")
    for x0, x1 in ((0.27, 0.38), (0.62, 0.73)):
        ax_c.annotate(
            "",
            xy=(x1, 0.78),
            xytext=(x0, 0.78),
            arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#555555"},
        )
    ax_c.text(
        0.50,
        0.61,
        "60 s blocks -> 4 position-balanced folds (1 s guards)",
        ha="center",
        fontsize=8.2,
    )
    draw_box(
        ax_c,
        (0.08, 0.38),
        0.35,
        0.14,
        "foldwise local-linear\nJacobians  $J^{(k)}$",
        facecolor="#EAF2F8",
        edgecolor="#0072B2",
    )
    draw_box(
        ax_c,
        (0.57, 0.38),
        0.35,
        0.14,
        "cross-fold tensor\n$\\hat g_{\\times}$",
        facecolor="#EAF2F8",
        edgecolor="#0072B2",
    )
    ax_c.annotate(
        "",
        xy=(0.57, 0.45),
        xytext=(0.43, 0.45),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#555555"},
    )
    draw_box(
        ax_c,
        (0.23, 0.10),
        0.54,
        0.15,
        "condition - mean(bracketing squares)\nnormal - tangent; square-drift comparator",
        facecolor="#EDF7ED",
        edgecolor="#2E7D32",
    )
    ax_c.annotate(
        "",
        xy=(0.50, 0.25),
        xytext=(0.74, 0.38),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#555555"},
    )
    save(fig, "figure1_design.png")


def primary_rows(summary: dict, animal: str) -> list[dict]:
    record = next(item for item in summary["animals"] if item["animal"] == animal)
    return [
        row
        for row in record["sequence_results"]
        if row["analysis_spec"] == "primary_occupancy_balanced_10cm"
    ]


def figure_2(summary: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.65), constrained_layout=True)
    exposures = np.array([1, 2, 3])

    for animal in ANIMALS:
        rows = primary_rows(summary, animal)
        values = [row["target_contrast"] for row in rows]
        axes[0].plot(
            exposures,
            values,
            color=COLORS[animal],
            marker=MARKERS[animal],
            ms=6,
            lw=1.8,
            label=ROLES[animal],
        )
        for exposure, row in zip(exposures, rows, strict=True):
            if row["gate_passes_all"]:
                axes[0].scatter(
                    exposure,
                    row["target_contrast"],
                    s=78,
                    facecolor=COLORS[animal],
                    edgecolor="black",
                    linewidth=0.8,
                    marker=MARKERS[animal],
                    zorder=5,
                )
    axes[0].axhline(0, color="#777777", lw=0.9, ls="--")
    axes[0].set_xticks(exposures)
    axes[0].set_xlabel("repeated exposure")
    axes[0].set_ylabel("normal - tangent contrast")
    axes[0].set_title("All 12 directional contrasts are positive")
    axes[0].legend(frameon=False, loc="upper left")
    panel_label(axes[0], "A")

    jitter = np.linspace(-0.16, 0.16, 3)
    for animal_index, animal in enumerate(ANIMALS):
        rows = primary_rows(summary, animal)
        color = COLORS[animal]
        for j, row in enumerate(rows):
            axes[1].scatter(
                abs(row["square_null_contrast"]),
                row["target_contrast"],
                color=color,
                marker=MARKERS[animal],
                s=45,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            if animal == "QLAK-CA1-50" and j == 2:
                axes[1].annotate(
                    "50, exp. 3",
                    (
                        abs(row["square_null_contrast"]),
                        row["target_contrast"],
                    ),
                    xytext=(13, -10),
                    textcoords="offset points",
                    fontsize=7.5,
                )
    upper = 3.45
    axes[1].plot([0, upper], [0, upper], color="#777777", ls="--", lw=1.0)
    axes[1].set_xlim(-0.03, 0.47)
    axes[1].set_ylim(-0.06, upper)
    axes[1].set_xlabel("|square virtual-wall drift|")
    axes[1].set_ylabel("true-wall contrast")
    axes[1].set_title("Every target exceeds frozen comparator")
    panel_label(axes[1], "B")

    for animal in ANIMALS:
        rows = primary_rows(summary, animal)
        values = [row["all_query_contrast_reliability"] for row in rows]
        axes[2].plot(
            exposures,
            values,
            color=COLORS[animal],
            marker=MARKERS[animal],
            ms=6,
            lw=1.8,
        )
    axes[2].axhline(0.3, color="#C62828", lw=1.2, ls="--", label="frozen gate = 0.3")
    axes[2].axhline(0, color="#777777", lw=0.8)
    axes[2].set_xticks(exposures)
    axes[2].set_ylim(-0.48, 0.70)
    axes[2].set_xlabel("repeated exposure")
    axes[2].set_ylabel("fold-pair contrast correlation")
    axes[2].set_title("Only 3 of 12 pass the reliability gate")
    axes[2].legend(frameon=False, loc="lower right")
    panel_label(axes[2], "C")
    save(fig, "figure2_primary_results.png")


def figure_3(common: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), constrained_layout=True)
    exposures = np.array([1, 2, 3])
    by_animal = {item["animal"]: item for item in common["animals"]}

    for animal in ANIMALS:
        record = by_animal[animal]
        seqs = [record["sequences"][f"sequence_{idx}"] for idx in exposures]
        contrast = [seq["target"]["anisotropy"][0] for seq in seqs]
        reliability = [
            seq["target_residual_reliability"]["contrast"] for seq in seqs
        ]
        axes[0].plot(
            exposures,
            contrast,
            color=COLORS[animal],
            marker=MARKERS[animal],
            ms=6,
            lw=1.8,
            label=ROLES[animal],
        )
        axes[1].plot(
            exposures,
            reliability,
            color=COLORS[animal],
            marker=MARKERS[animal],
            ms=6,
            lw=1.8,
        )
    axes[0].axhline(0, color="#777777", lw=0.9)
    axes[0].set_xticks(exposures)
    axes[0].set_xlabel("repeated exposure")
    axes[0].set_ylabel("common-support contrast")
    axes[0].set_title("Effect changes: two rise, two fall")
    axes[0].legend(frameon=False, loc="upper left")
    panel_label(axes[0], "A")

    axes[1].axhline(0.3, color="#C62828", lw=1.2, ls="--")
    axes[1].axhline(0, color="#777777", lw=0.8)
    axes[1].set_xticks(exposures)
    axes[1].set_ylim(-0.30, 0.72)
    axes[1].set_xlabel("repeated exposure")
    axes[1].set_ylabel("common-support reliability")
    axes[1].set_title("Reliability is nonmonotonic")
    panel_label(axes[1], "B")

    calibration = common["cohort_exact_boundary_segment_label_spin"]
    null = np.asarray(calibration["exact_animal_mean_statistics"], dtype=float)
    observed = calibration["observed_animal_mean"]
    axes[2].hist(
        null,
        bins=23,
        color="#BDBDBD",
        edgecolor="white",
        linewidth=0.5,
    )
    axes[2].axvline(observed, color="#C62828", lw=2.2)
    axes[2].axvline(-observed, color="#C62828", lw=1.2, ls=":")
    axes[2].annotate(
        f"observed = {observed:.3f}\n2-sided fraction = {calibration['two_sided_tail_fraction']:.4f}",
        xy=(observed, 18),
        xytext=(-92, 14),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#C62828"},
        fontsize=8,
    )
    axes[2].set_xlabel("animal-mean contrast under segment-label spin")
    axes[2].set_ylabel("exact labelings")
    axes[2].set_title("Orientation-label calibration")
    panel_label(axes[2], "C")
    save(fig, "figure3_experience_calibration.png")


def figure_s1(behavior: dict, shifts: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), constrained_layout=True)
    summary = behavior["cohort_descriptive_summary"]
    unadjusted = summary["animal_mean_unadjusted_near_anisotropy"]
    adjusted = summary["animal_mean_behavior_adjusted_near_anisotropy"]
    x = np.array([0, 1])
    for animal, before, after in zip(
        ANIMALS, unadjusted, adjusted, strict=True
    ):
        axes[0].plot(
            x,
            [before, after],
            color=COLORS[animal],
            marker=MARKERS[animal],
            ms=6,
            lw=1.8,
            label=ROLES[animal],
        )
    axes[0].axhline(0, color="#777777", lw=0.8)
    axes[0].set_xticks(x, ["frozen", "speed/heading\nadjusted"])
    axes[0].set_xlim(-0.25, 1.25)
    axes[0].set_ylabel("animal-mean contrast")
    axes[0].set_title("Post-outcome nuisance adjustment")
    axes[0].legend(frameon=False, loc="upper left")
    panel_label(axes[0], "A")

    clean = shifts["animals"]
    y = np.arange(len(clean))
    for yi, item in zip(y, clean, strict=True):
        axes[1].plot(
            [item["surrogate_minimum"], item["surrogate_maximum"]],
            [yi, yi],
            color="#777777",
            lw=5,
            solid_capstyle="butt",
        )
        axes[1].scatter(
            item["observed_mean_near_anisotropy"],
            yi,
            color="#C62828",
            s=55,
            zorder=3,
            label="observed" if yi == 0 else None,
        )
        axes[1].scatter(
            item["surrogate_mean"],
            yi,
            color="black",
            marker="|",
            s=100,
            zorder=3,
            label="surrogate mean" if yi == 0 else None,
        )
    axes[1].axvline(0, color="#B0B0B0", lw=0.8)
    axes[1].set_yticks(y, [SHORT[item["animal"]] for item in clean])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("animal-mean contrast")
    axes[1].set_ylabel("clean prospectively locked animal")
    axes[1].set_title("19 population trace-position shifts")
    axes[1].legend(frameon=False, loc="lower right")
    panel_label(axes[1], "B")
    save(fig, "figureS1_post_outcome_controls.png")


def main() -> None:
    configure()
    summary = load_json("plus_cohort_summary.json")
    common = load_json("plus_common_support_exploratory.json")
    behavior = load_json("plus_behavior_nuisance_control.json")
    shifts = load_json("plus_trace_shift_B19_summary.json")
    figure_1()
    figure_2(summary)
    figure_3(common)
    figure_s1(behavior, shifts)
    print(f"Wrote manuscript figures to {OUTPUT}")


if __name__ == "__main__":
    main()
