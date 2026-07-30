from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_blair_barrier_scar",
    ROOT / "scripts" / "run_blair_barrier_scar.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_matlab_intervals_are_inclusive_and_alternating():
    intervals = MODULE.normalize_intervals(
        np.asarray([[1, 3], [5, 6], [9, 9]]),
        n_frames=10,
    )
    assert intervals.tolist() == [[0, 2], [4, 5], [8, 8]]
    full = MODULE.interval_mask(10, intervals)
    odd = MODULE.interval_mask(10, intervals, range(0, 3, 2))
    even = MODULE.interval_mask(10, intervals, range(1, 3, 2))
    assert np.flatnonzero(full).tolist() == [0, 1, 2, 4, 5, 8]
    assert np.flatnonzero(odd).tolist() == [0, 1, 2, 8]
    assert np.flatnonzero(even).tolist() == [4, 5]
    assert not np.any(odd & even)
    assert np.array_equal(odd | even, full)


def test_direction_reconstruction_uses_official_constant_frame_occupancy():
    position_bin = np.tile(np.arange(1, 24), 2)
    intervals = np.asarray([[0, 22], [23, 45]])
    events = np.zeros((2, 46), dtype=np.float64)
    events[0, :23] = 1
    events[1, 23:] = 1
    maps = MODULE._direction_candidate(
        events=events,
        position_bin=position_bin,
        intervals=intervals,
        frame_interval_ms=100.0,
    )
    assert maps.frame_interval_ms == pytest.approx(100.0)
    assert np.allclose(maps.full, 0.005)
    assert np.allclose(maps.odd[0], 0.01)
    assert np.allclose(maps.odd[1], 0.0)
    assert np.allclose(maps.even[0], 0.0)
    assert np.allclose(maps.even[1], 0.01)


def test_correlations_and_fisher_transform():
    first = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert MODULE.correlation(
        first,
        first[::-1],
        method="pearson",
    ) == pytest.approx(-1.0)
    assert MODULE.correlation(
        first,
        np.asarray([10.0, 20.0, 30.0, 100.0]),
        method="spearman",
    ) == pytest.approx(1.0)
    assert np.isfinite(MODULE.fisher_z(1.0))


def test_primary_spatial_contrast_uses_equal_symmetric_controls():
    profile = np.zeros(23, dtype=np.float64)
    profile[list(MODULE.CENTER)] = 5.0
    profile[list(MODULE.CONTROL_LEFT)] = 1.0
    profile[list(MODULE.CONTROL_RIGHT)] = 3.0
    result = MODULE.spatial_summary(profile)
    assert result["former_barrier_mean"] == 5.0
    assert result["equal_weight_control_mean"] == 2.0
    assert result["primary_center_minus_control"] == 3.0


def test_exact_sign_flip_enumerates_animals_not_bins():
    result = MODULE.exact_sign_flip([1.0, 2.0])
    assert result["permutations"] == 4
    assert result["observed_mean"] == 1.5
    assert result["one_sided_positive_tail"] == 0.25
    assert result["two_sided_absolute_tail"] == 0.5


def test_json_safe_replaces_nonfinite_values():
    value = MODULE.json_safe(
        {"finite": 1.0, "missing": [float("nan"), float("inf")]}
    )
    assert value == {"finite": 1.0, "missing": [None, None]}


def test_cellreg_parser_selects_positive_pre_post_rows(tmp_path):
    directory = tmp_path / "cellmaps"
    directory.mkdir()
    savemat(
        directory / "Rat_barrier_cmap.mat",
        {
            "sessionNums": np.asarray([[2, 5, 9]]),
            "cmap": np.asarray(
                [
                    [1, 4, 7],
                    [2, 0, 8],
                    [3, 5, 0],
                    [4, 6, 9],
                ]
            ),
        },
    )
    pre, post, audit = MODULE.matched_cell_indices(
        tmp_path,
        rat="Rat",
        pre_session=5,
        post_session=9,
    )
    assert pre.tolist() == [3, 5]
    assert post.tolist() == [6, 8]
    assert audit["matched_pre_post_cells"] == 2


def _synthetic_direction(
    full: np.ndarray,
) -> object:
    return MODULE.DirectionMaps(
        full=full,
        odd=full.copy(),
        even=full.copy(),
        trips=10,
        frames_full=np.full(23, 20),
        frames_odd=np.full(23, 10),
        frames_even=np.full(23, 10),
        frame_interval_ms=100.0,
    )


def test_mode_detects_reliable_center_specific_change():
    cells = 30
    base = np.linspace(-1.0, 1.0, cells)[:, None]
    pre_full = np.repeat(base, 23, axis=1)
    post_full = pre_full.copy()
    post_full[:, list(MODULE.CENTER)] *= -1.0
    pre = MODULE.SessionMaps(
        local_activity=np.ones(cells, dtype=bool),
        directions={
            "LR": _synthetic_direction(pre_full),
            "RL": _synthetic_direction(pre_full),
        },
        event_field="S",
        candidate_reconstruction={},
        official_vmap_reconstruction={},
        time_normalization="synthetic",
        processed_cells=cells,
        raw_cells=cells,
    )
    post = MODULE.SessionMaps(
        local_activity=np.ones(cells, dtype=bool),
        directions={
            "LR": _synthetic_direction(post_full),
            "RL": _synthetic_direction(post_full),
        },
        event_field="S",
        candidate_reconstruction={},
        official_vmap_reconstruction={},
        time_normalization="synthetic",
        processed_cells=cells,
        raw_cells=cells,
    )
    result = MODULE.analyze_mode(
        pre,
        post,
        np.arange(cells),
        np.arange(cells),
        method="pearson",
    )
    assert (
        result["primary_spatial"]["primary_center_minus_control"] > 1.0
    )


@pytest.mark.skipif(
    not MODULE.DEFAULT_DATA.exists(),
    reason="external Blair dataset is not installed",
)
def test_released_hybrid_parser_reconstructs_historical_maps():
    maps = MODULE.load_session_maps(
        MODULE.DEFAULT_DATA,
        rat="Hipp12",
        session=5,
        role="pre",
    )
    assert maps.event_field == "S"
    assert maps.processed_cells == maps.raw_cells == 296
    assert (
        maps.candidate_reconstruction["S"]["LR"]
        <= MODULE.RECONSTRUCTION_TOLERANCE
    )
    pre, post, audit = MODULE.matched_cell_indices(
        MODULE.DEFAULT_DATA,
        rat="Hipp12",
        pre_session=5,
        post_session=9,
    )
    assert pre.size == post.size == audit["matched_pre_post_cells"]
    assert pre.size >= MODULE.MINIMUM_CELLS
