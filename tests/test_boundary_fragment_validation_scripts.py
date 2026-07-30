"""Regression tests for the raw-event boundary validation scripts."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BEHAVIOR = _load_script(
    "boundary_fragment_behavior_adjusted",
    "scripts/run_boundary_fragment_behavior_adjusted.py",
)
RAW_SPLIT = _load_script(
    "boundary_fragment_raw_split",
    "scripts/run_boundary_fragment_raw_split.py",
)


def _synthetic_spatial_regression() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(1729)
    samples = 4_000
    bin_index = np.arange(samples) % 2
    position = np.column_stack(
        (
            2.5 + 5.0 * bin_index,
            np.full(samples, 2.5),
        )
    )
    design = rng.normal(size=(samples, 6))
    coefficient = np.array([0.3, -0.2, 0.1, 0.4, -0.1, 0.2])
    bin_effect = np.array([1.0, 2.0])
    response = (
        bin_effect[bin_index] + design @ coefficient
    )[:, None]
    reference = np.array([0.5, -0.25, 0.1, 0.0, 0.2, -0.4])
    return position, response, design, reference, bin_effect


def test_common_reference_fwl_recovers_known_bin_effects() -> None:
    position, response, design, reference, bin_effect = (
        _synthetic_spatial_regression()
    )
    maps, occupancy, _, _, _ = BEHAVIOR._adjusted_maps(
        position,
        response,
        np.ones(position.shape[0], dtype=bool),
        design,
        reference,
        frames_per_second=30.0,
        ridge_fraction=0.0,
    )

    coefficient = np.array([0.3, -0.2, 0.1, 0.4, -0.1, 0.2])
    expected = bin_effect + reference @ coefficient
    np.testing.assert_allclose(
        maps[BEHAVIOR.MODE_WITH_TIME][0, 0, :2],
        expected,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        occupancy[0, :2],
        np.full(2, position.shape[0] / 2 / 30.0),
    )


def test_inaccessible_frame_is_excluded_from_behavior_fit_and_support() -> None:
    position, response, design, reference, _ = (
        _synthetic_spatial_regression()
    )
    blocked_position = np.array([[27.5, 27.5]])
    position = np.vstack((position, blocked_position))
    analysis_valid = np.ones(position.shape[0], dtype=bool)
    analysis_valid[-1] = False

    first_response = np.vstack((response, [[1.0e6]]))
    second_response = np.vstack((response, [[-1.0e6]]))
    first = BEHAVIOR._adjusted_maps(
        position,
        first_response,
        analysis_valid,
        design,
        reference,
        frames_per_second=30.0,
        ridge_fraction=0.0,
    )
    second = BEHAVIOR._adjusted_maps(
        position,
        second_response,
        analysis_valid,
        design,
        reference,
        frames_per_second=30.0,
        ridge_fraction=0.0,
    )

    for mode in BEHAVIOR.MODES:
        np.testing.assert_allclose(
            first[0][mode],
            second[0][mode],
            equal_nan=True,
        )
        assert np.isnan(first[0][mode][0, 5, 5])
    assert first[1][5, 5] == 0.0
    assert first[3][5, 5] == 1.0 / 30.0
    assert first[2][0, 5, 5] == 1.0e6
    assert second[2][0, 5, 5] == -1.0e6


def test_behavior_maps_are_invariant_to_cell_chunking() -> None:
    position, response, design, reference, _ = (
        _synthetic_spatial_regression()
    )
    response = np.column_stack(
        (
            response[:, 0],
            0.5 * response[:, 0] + 0.2,
            -0.25 * response[:, 0] + 1.3,
            2.0 * response[:, 0] - 0.7,
            0.1 * response[:, 0] + 3.0,
        )
    )
    valid = np.ones(position.shape[0], dtype=bool)
    full = BEHAVIOR._adjusted_maps(
        position,
        response,
        valid,
        design,
        reference,
        frames_per_second=30.0,
        ridge_fraction=0.001,
    )
    pieces = [
        BEHAVIOR._adjusted_maps(
            position,
            response[:, start : start + 2],
            valid,
            design,
            reference,
            frames_per_second=30.0,
            ridge_fraction=0.001,
        )
        for start in range(0, response.shape[1], 2)
    ]

    for mode in BEHAVIOR.MODES:
        chunked = np.concatenate([piece[0][mode] for piece in pieces])
        np.testing.assert_allclose(
            chunked,
            full[0][mode],
            atol=1e-12,
            equal_nan=True,
        )
    np.testing.assert_allclose(
        np.concatenate([piece[2] for piece in pieces]),
        full[2],
        atol=1e-12,
        equal_nan=True,
    )
    for piece in pieces:
        np.testing.assert_array_equal(piece[1], full[1])
        np.testing.assert_array_equal(piece[3], full[3])


def test_behavior_animal_checkpoint_requires_exact_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "QLAK-CA1-test.complete.mat"
    source.write_bytes(b"input fingerprint")
    argument = argparse.Namespace(
        frames_per_second=30.0,
        velocity_half_window_frames=3,
        speed_cap_quantile=0.995,
        minimum_direction_speed_cm_s=2.0,
        behavior_ridge_fraction=0.001,
        trace_cell_chunk=128,
        minimum_seconds=0.5,
        minimum_bins=6,
        minimum_cells=20,
        match_nonfocal_context=True,
        match_global_counterfactual=True,
    )
    metadata = BEHAVIOR._cache_metadata(source, argument)
    cache = tmp_path / "checkpoint.json"
    result = {"animal": "QLAK-CA1-test", "records": 3}

    BEHAVIOR._write_cached_animal(cache, metadata, result)

    assert BEHAVIOR._read_cached_animal(cache, metadata) == result
    changed = json.loads(json.dumps(metadata))
    changed["settings"]["minimum_bins"] = 7
    assert BEHAVIOR._read_cached_animal(cache, changed) is None


def test_raw_half_keeps_occupancy_but_masks_inaccessible_rates() -> None:
    position = np.array(
        [
            [2.5, 2.5],
            [27.5, 27.5],
            [2.5, 2.5],
        ]
    )
    trace = np.array(
        [
            [1.0, 2.0],
            [100.0, 200.0],
            [3.0, 4.0],
        ]
    )
    accessible = RAW_SPLIT._accessible_bin_mask((4,))
    rate, occupancy = RAW_SPLIT._aggregate_half(
        position,
        trace,
        total_cells=4,
        registered_cells=np.array([1, 3]),
        accessible_bins=accessible,
    )

    assert not accessible[5, 5]
    assert accessible[0, 0]
    np.testing.assert_allclose(rate[[1, 3], 0, 0], [2.0, 3.0])
    assert np.isnan(rate[:, 5, 5]).all()
    assert np.isnan(rate[[0, 2]]).all()
    assert occupancy[0, 0] == 2.0 / 30.0
    assert occupancy[5, 5] == 1.0 / 30.0


def test_discovery_figure_table_matches_tracked_source_artifacts() -> None:
    source = ROOT / "results" / "source_data"
    fragment = json.loads(
        (source / "boundary_fragment_screen.json").read_text(
            encoding="utf-8"
        )
    )
    validation = json.loads(
        (source / "boundary_component_validation.json").read_text(
            encoding="utf-8"
        )
    )
    raw_split = json.loads(
        (source / "boundary_fragment_raw_split.json").read_text(
            encoding="utf-8"
        )
    )
    behavior = json.loads(
        (source / "boundary_fragment_behavior_adjusted.json").read_text(
            encoding="utf-8"
        )
    )
    with (source / "boundary_component_figure.csv").open(
        newline="",
        encoding="utf-8",
    ) as stream:
        rows = {row["animal"]: row for row in csv.DictReader(stream)}

    for item in fragment["animals"]:
        animal = item["animal"]
        raw_item = next(
            value for value in raw_split["animals"]
            if value["animal"] == animal
        )
        row = rows[animal]
        assert float(row["same_cycle_full_session_strip_delta_r"]) == (
            item["all_sequence_mean"]
        )
        assert float(
            row["target_rate_heldout_global_rate_demeaned_delta_r"]
        ) == validation["cohort"]["global_rate_demeaned"][
            "animal_values"
        ][animal]
        assert float(row["same_cycle_raw_split_delta_r"]) == (
            raw_item["primary"]["all_sequence_mean"]
        )
        assert float(
            row["target_rate_heldout_behavior_adjusted_delta_r"]
        ) == behavior["cohort"]["modes"][
            "speed_movement_direction_time"
        ]["animal_values"][animal]
