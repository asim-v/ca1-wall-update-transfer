"""Focused tests for cross-location spatial and identity audits."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str) -> ModuleType:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SPATIAL = _load_script(
    "cross_location_spatial_controls",
    "scripts/audit_boundary_fragment_cross_location_spatial_controls.py",
)
IDENTITY = _load_script(
    "cross_location_cell_permutation",
    "scripts/audit_boundary_fragment_cross_location_cell_permutation.py",
)


def test_exact_grid_step_strip_geometry_is_classified_without_aliases() -> None:
    census = SPATIAL.geometry_census()

    assert census["same_tangential"] == {
        "directed_seam_pairs": 32,
        "orientation_relation": "same_signed_normal",
        "translation_axis": "tangential",
        "normal_displacement_cm": 0.0,
        "tangential_displacement_cm": 25.0,
        "strip_overlap_bins": 0,
        "strip_minimum_bin_center_distance_cm": 5.0,
    }
    assert census["same_normal"] == {
        "directed_seam_pairs": 24,
        "orientation_relation": "same_signed_normal",
        "translation_axis": "normal",
        "normal_displacement_cm": 25.0,
        "tangential_displacement_cm": 0.0,
        "strip_overlap_bins": 0,
        "strip_minimum_bin_center_distance_cm": 15.0,
    }
    assert census["opposite_normal_facing_overlap"][
        "strip_overlap_bins"
    ] == 5
    assert census["opposite_normal_away"][
        "strip_minimum_bin_center_distance_cm"
    ] == 30.0


def test_spatial_audit_reproduces_validation_and_exposes_matched_controls() -> None:
    source = ROOT / "results" / "source_data"
    validation = json.loads(
        (source / "boundary_fragment_cross_location_transfer.json").read_text(
            encoding="utf-8"
        )
    )
    audit = json.loads(
        (
            source
            / "boundary_fragment_cross_location_spatial_controls.json"
        ).read_text(encoding="utf-8")
    )
    for mode in ("raw_local_rate", "global_rate_demeaned"):
        expected = validation["cohort_descriptive"][mode]
        observed = audit["cohort_descriptive"][mode]
        assert observed["same_all_transfer"]["animal_values"] == expected[
            "one_grid_step_same_direction_source_effect_r_to_target_residual"
        ]["animal_values"]
        assert observed[
            "same_all_minus_opposite_all_transfer"
        ]["animal_values"] == expected[
            "one_grid_step_same_direction_minus_exact_distance_"
            "nonsame_source_effect_r"
        ]["animal_values"]

    primary = audit["cohort_descriptive"]["global_rate_demeaned"]
    assert primary[
        "same_tangential_minus_opposite_tangential_transfer"
    ]["positive_animals"] == 3
    assert primary[
        "same_normal_minus_opposite_normal_facing_overlap_transfer"
    ]["positive_animals"] == 4
    assert primary["same_tangential_transfer"]["positive_animals"] == 6
    assert primary[
        "opposite_tangential_transfer"
    ]["positive_animals"] == 6


def test_prepared_query_scores_source_groups_before_query_averaging() -> None:
    cells = np.arange(4, dtype=np.int64)
    target = np.array([-0.5, -0.5, 0.5, 0.5])
    same = np.stack(
        (
            target,
            np.array([-0.5, 0.5, -0.5, 0.5]),
        )
    )
    opposite = np.stack((-target,))
    query = IDENTITY.PreparedQuery(
        cell_ids=cells,
        same_source={mode: same for mode in IDENTITY.transfer.RATE_MODES},
        opposite_source={
            mode: opposite for mode in IDENTITY.transfer.RATE_MODES
        },
        target={
            mode: target for mode in IDENTITY.transfer.RATE_MODES
        },
    )

    primary, direction = IDENTITY._score_queries(
        [query],
        mode="global_rate_demeaned",
    )

    assert np.isclose(primary, 0.5)
    assert np.isclose(direction, 1.5)


def test_tracked_identity_diagnostic_reproduces_observed_scores() -> None:
    path = (
        ROOT
        / "results"
        / "source_data"
        / "boundary_fragment_cross_location_cell_permutation.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["settings"]["permutations"] == 999
    assert all(
        max(
            animal["validation_reproduction_max_absolute_error"].values()
        )
        <= 1e-12
        for animal in report["animals"]
    )
    assert all(
        animal["exchangeability"][
            "all_query_common_cell_sets_preserved_every_draw"
        ]
        for animal in report["animals"]
    )
    primary = report["cohort_descriptive"]["global_rate_demeaned"][
        "primary_same_normal_transfer"
    ]
    assert primary["positive_observed_animals"] == 7
    assert primary["maximum_animal_tail_fraction"] == 0.069
