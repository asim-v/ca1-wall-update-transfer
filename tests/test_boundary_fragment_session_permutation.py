"""Focused tests for the coherent global-cell permutation diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    path = ROOT / "scripts" / "run_boundary_fragment_session_permutation.py"
    spec = importlib.util.spec_from_file_location(
        "boundary_fragment_session_permutation",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PERMUTATION = _load_script()


def test_membership_strata_preserve_every_record_subset() -> None:
    record_ids = [
        np.array([0, 1, 2, 3], dtype=np.int64),
        np.array([1, 2, 3, 4], dtype=np.int64),
        np.array([1, 2, 3, 5], dtype=np.int64),
    ]
    strata = PERMUTATION._membership_strata(
        record_ids,
        n_cells=7,
    )
    np.testing.assert_array_equal(
        next(value for value in strata if value.size == 3),
        [1, 2, 3],
    )

    mapping = PERMUTATION._draw_global_permutation(
        strata,
        n_cells=7,
        rng=np.random.default_rng(28),
    )
    np.testing.assert_array_equal(np.sort(mapping), np.arange(7))
    for ids in record_ids:
        np.testing.assert_array_equal(
            np.sort(mapping[ids]),
            ids,
        )


def test_overlapping_records_use_the_same_global_identity_mapping() -> None:
    first_ids = np.array([0, 1, 2, 3], dtype=np.int64)
    second_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    strata = PERMUTATION._membership_strata(
        [first_ids, second_ids],
        n_cells=5,
    )
    mapping = PERMUTATION._draw_global_permutation(
        strata,
        n_cells=5,
        rng=np.random.default_rng(4),
    )
    first_positions = PERMUTATION._permuted_positions(
        first_ids,
        mapping,
    )
    second_positions = PERMUTATION._permuted_positions(
        second_ids,
        mapping,
    )

    first_target_by_global_id = 10.0 * first_ids
    second_target_by_global_id = 100.0 * second_ids
    for global_id in (1, 2, 3):
        first_position = int(np.flatnonzero(first_ids == global_id)[0])
        second_position = int(np.flatnonzero(second_ids == global_id)[0])
        mapped_global_id = int(mapping[global_id])
        assert (
            first_target_by_global_id[first_positions[first_position]]
            == 10.0 * mapped_global_id
        )
        assert (
            second_target_by_global_id[second_positions[second_position]]
            == 100.0 * mapped_global_id
        )


def test_shared_draws_are_deterministic_and_score_both_modes() -> None:
    cells = np.arange(6, dtype=np.int64)
    wall = np.arange(6, dtype=np.float64)
    open_value = wall[::-1]
    target = wall.copy()
    vectors = {
        mode: (wall, open_value, target)
        for mode in PERMUTATION.validation.RATE_MODES
    }
    record = PERMUTATION._prepare_record(cells, vectors)
    first = PERMUTATION._permutation_diagnostic(
        [record],
        n_cells=6,
        draws=25,
        seed=1729,
    )
    second = PERMUTATION._permutation_diagnostic(
        [record],
        n_cells=6,
        draws=25,
        seed=1729,
    )

    assert first == second
    for mode in PERMUTATION.validation.RATE_MODES:
        assert np.isclose(
            first[0][mode]["observed_animal_mean_wall_minus_open"],
            2.0,
        )
        assert first[0][mode]["draws"] == 25
    assert first[1][
        "same_global_mapping_used_for_all_records_and_modes_per_draw"
    ]
    assert (
        first[1][
            "records_with_identity_pairing_changed_per_draw_min_mean_max"
        ][2]
        == 1
    )
    assert first[1]["derangements_enforced"] is False
