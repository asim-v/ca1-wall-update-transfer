"""Regression checks for the partial-generalization explainer figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "source_data"


def _read(name: str) -> dict:
    return json.loads((SOURCE / name).read_text(encoding="utf-8"))


def test_explainer_table_matches_tracked_artifacts() -> None:
    transfer = _read("boundary_fragment_cross_location_transfer.json")
    spatial = _read("boundary_fragment_cross_location_spatial_controls.json")
    mirror = _read("boundary_fragment_cross_location_mirror_open.json")
    with (
        SOURCE / "partial_generalization_explainer.csv"
    ).open(newline="", encoding="utf-8") as stream:
        rows = {row["animal"]: row for row in csv.DictReader(stream)}

    transfer_mode = transfer["cohort_descriptive"]["global_rate_demeaned"]
    cross = transfer_mode[
        "one_grid_step_same_direction_source_effect_r_to_target_residual"
    ]["animal_values"]
    cross_minus_exact = transfer_mode[
        "one_grid_step_same_direction_minus_exact_location_"
        "effect_r_to_target_residual"
    ]["animal_values"]
    spatial_mode = spatial["cohort_descriptive"]["global_rate_demeaned"]
    pooled = spatial_mode["same_all_minus_opposite_all_transfer"][
        "animal_values"
    ]
    strict = spatial_mode[
        "same_tangential_minus_opposite_tangential_transfer"
    ]["animal_values"]
    mirror_mode = mirror["cohort_descriptive"]["modes"][
        "global_rate_demeaned"
    ]
    wall = mirror_mode["source_effect_r_to_wall_target"]["values_by_animal"]
    open_target = mirror_mode["source_effect_r_to_mirror_open"][
        "values_by_animal"
    ]

    assert set(rows) == set(cross) == set(pooled)
    for animal, row in rows.items():
        assert np.isclose(float(row["cross_location_r"]), cross[animal])
        assert np.isclose(
            float(row["exact_location_r"]),
            cross[animal] - cross_minus_exact[animal],
        )
        assert np.isclose(
            float(row["pooled_correct_minus_wrong_r"]), pooled[animal]
        )
        if animal in strict:
            assert np.isclose(
                float(row["strict_tangential_correct_minus_wrong_r"]),
                strict[animal],
            )
        else:
            assert row["strict_tangential_correct_minus_wrong_r"] == ""
        if animal in wall:
            assert np.isclose(float(row["mirror_wall_r"]), wall[animal])
            assert np.isclose(
                float(row["mirror_open_r"]), open_target[animal]
            )
        else:
            assert row["mirror_wall_r"] == ""
            assert row["mirror_open_r"] == ""


def test_explainer_encodes_the_claim_and_its_boundary() -> None:
    with (
        SOURCE / "partial_generalization_explainer.csv"
    ).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    exact = np.array([float(row["exact_location_r"]) for row in rows])
    cross = np.array([float(row["cross_location_r"]) for row in rows])
    pooled = np.array(
        [float(row["pooled_correct_minus_wrong_r"]) for row in rows]
    )
    strict = np.array(
        [
            float(row["strict_tangential_correct_minus_wrong_r"])
            for row in rows
            if row["strict_tangential_correct_minus_wrong_r"]
        ]
    )

    assert np.all(cross > 0)
    assert np.isclose(np.mean(exact), 0.22975011666270705)
    assert np.isclose(np.mean(cross), 0.14766406504657636)
    assert np.sum(cross > exact) == 1
    assert np.all(pooled > 0)
    assert np.sum(strict > 0) == 3
