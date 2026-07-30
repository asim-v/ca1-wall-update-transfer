"""Regression checks for the stronger-claim evidence figure source table."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "source_data"


def _read(name: str) -> dict:
    return json.loads((SOURCE / name).read_text(encoding="utf-8"))


def test_local_wall_update_figure_table_matches_source_artifacts() -> None:
    context = _read("boundary_fragment_context_matched.json")
    single = _read("boundary_fragment_single_tile_counterfactual.json")
    behavior = _read(
        "boundary_fragment_single_tile_counterfactual_behavior_adjusted.json"
    )
    transfer = _read("boundary_fragment_cross_location_transfer.json")
    transfer_behavior = _read(
        "boundary_fragment_cross_location_behavior_adjusted.json"
    )
    mirror = _read("boundary_fragment_cross_location_mirror_open.json")
    spatial = _read(
        "boundary_fragment_cross_location_spatial_controls.json"
    )
    with (
        SOURCE / "local_wall_update_transfer_figure.csv"
    ).open(newline="", encoding="utf-8") as stream:
        rows = {row["animal"]: row for row in csv.DictReader(stream)}

    context_values = context["cohort"]["global_rate_demeaned"][
        "animal_values"
    ]
    single_values = single["cohort"]["global_rate_demeaned"][
        "animal_values"
    ]
    behavior_values = behavior["cohort"]["modes"][
        "speed_movement_direction_time"
    ]["animal_values"]
    translated = transfer["cohort_descriptive"]["global_rate_demeaned"]
    translated_values = translated[
        "one_grid_step_same_direction_source_effect_r_to_target_residual"
    ]["animal_values"]
    translated_minus_exact = translated[
        (
            "one_grid_step_same_direction_minus_exact_location_"
            "effect_r_to_target_residual"
        )
    ]["animal_values"]
    specificity = spatial["cohort_descriptive"]["global_rate_demeaned"][
        "same_all_effect_specificity_over_target_open"
    ]["animal_values"]
    translated_behavior = transfer_behavior["cohort_descriptive"][
        "speed_movement_direction_time"
    ]["source_effect_r_to_target_residual"]["animal_values"]
    mirror_mode = mirror["cohort_descriptive"]["modes"][
        "global_rate_demeaned"
    ]
    mirror_wall = mirror_mode["source_effect_r_to_wall_target"][
        "values_by_animal"
    ]
    mirror_open = mirror_mode["source_effect_r_to_mirror_open"][
        "values_by_animal"
    ]
    mirror_advantage = mirror_mode[
        "wall_minus_mirror_open_correlation_advantage"
    ]["values_by_animal"]

    assert set(rows) == set(single_values)
    for animal, row in rows.items():
        assert np.isclose(
            float(row["context_matched_wall_minus_open_delta_r"]),
            context_values[animal],
        )
        assert np.isclose(
            float(row["single_tile_wall_minus_open_delta_r"]),
            single_values[animal],
        )
        assert np.isclose(
            float(row["single_tile_behavior_adjusted_delta_r"]),
            behavior_values[animal],
        )
        expected_exact = (
            translated_values[animal]
            - translated_minus_exact[animal]
        )
        assert np.isclose(
            float(row["exact_location_effect_r_to_target_residual"]),
            expected_exact,
        )
        assert np.isclose(
            float(row["translated_effect_r_to_target_residual"]),
            translated_values[animal],
        )
        assert np.isclose(
            float(
                row[
                    "translated_behavior_adjusted_r_to_target_residual"
                ]
            ),
            translated_behavior[animal],
        )
        assert np.isclose(
            float(row["translated_effect_specificity_delta_r"]),
            specificity[animal],
        )
        assert np.isclose(
            float(row["translated_effect_r_to_target_residual"])
            - float(
                row["translated_effect_r_to_target_open_profile"]
            ),
            specificity[animal],
        )
        if animal in mirror_wall:
            assert np.isclose(
                float(row["mirror_control_wall_r"]),
                mirror_wall[animal],
            )
            assert np.isclose(
                float(row["mirror_control_open_r"]),
                mirror_open[animal],
            )
            assert np.isclose(
                float(row["mirror_control_advantage_delta_r"]),
                mirror_advantage[animal],
            )
        else:
            assert row["mirror_control_wall_r"] == ""
            assert row["mirror_control_open_r"] == ""
            assert row["mirror_control_advantage_delta_r"] == ""


def test_single_tile_near_exceeds_far_for_every_animal() -> None:
    near = _read("boundary_fragment_single_tile_counterfactual_near.json")
    far = _read("boundary_fragment_single_tile_counterfactual_far.json")
    near_values = near["cohort"]["global_rate_demeaned"]["animal_values"]
    far_values = far["cohort"]["global_rate_demeaned"]["animal_values"]
    with (
        SOURCE / "local_wall_update_transfer_figure.csv"
    ).open(newline="", encoding="utf-8") as stream:
        rows = {row["animal"]: row for row in csv.DictReader(stream)}

    assert len(rows) == 7
    assert set(rows) == set(near_values) == set(far_values)
    for animal, row in rows.items():
        assert np.isclose(
            float(row["single_tile_near_delta_r"]),
            near_values[animal],
        )
        assert np.isclose(
            float(row["single_tile_far_delta_r"]),
            far_values[animal],
        )
    assert all(
        float(row["single_tile_near_delta_r"])
        > float(row["single_tile_far_delta_r"])
        for row in rows.values()
    )
