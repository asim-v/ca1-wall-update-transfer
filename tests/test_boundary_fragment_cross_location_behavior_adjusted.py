"""Tests for the behavior-adjusted cross-location transport."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    path = (
        ROOT
        / "scripts"
        / "run_boundary_fragment_cross_location_behavior_adjusted.py"
    )
    spec = importlib.util.spec_from_file_location(
        "boundary_fragment_cross_location_behavior_adjusted",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_script()


def _load_audit() -> ModuleType:
    path = (
        ROOT
        / "scripts"
        / (
            "audit_boundary_fragment_cross_location_behavior_adjusted_"
            "eligibility.py"
        )
    )
    spec = importlib.util.spec_from_file_location(
        "boundary_fragment_cross_location_behavior_adjusted_eligibility",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit()


def _record(
    relation: str,
    distance: float,
    first: float,
    second: float,
) -> dict[str, object]:
    metrics = {
        "source_effect_r_to_target_residual": first,
        "source_effect_specificity_over_target_open_r": second,
        "source_wall_minus_open_r_to_target_residual": first - second,
    }
    return {
        "source_seam": [0, 1],
        "orientation_relation": relation,
        "midpoint_distance_cm": distance,
        "cells": 30,
        "target_bins": 9,
        "source_bins": 8,
        "metrics": {
            mode: metrics.copy()
            for mode in MODULE.behavior.MODES
        },
    }


def test_bundle_rate_uses_requested_adjusted_map() -> None:
    first = np.zeros((3, 2, 2), dtype=float)
    second = np.zeros((3, 2, 2), dtype=float)
    first[:, 0, 0] = [1.0, 2.0, 3.0]
    second[:, 0, 0] = [4.0, 5.0, 6.0]
    bundle = {
        MODULE.behavior.MODE_NO_TIME: first,
        MODULE.behavior.MODE_WITH_TIME: second,
    }

    value = MODULE.ADJUSTED_RATE_MODES[
        MODULE.behavior.MODE_WITH_TIME
    ](
        bundle,
        np.ones((2, 2)),
        np.array([0, 2]),
        ((0, 0),),
    )

    np.testing.assert_allclose(value, [4.0, 6.0])


def test_primary_source_requires_same_normal_and_exact_grid_step() -> None:
    target = MODULE.transfer.OrientedSeam(0, 1)

    assert MODULE._is_primary_source(
        target,
        MODULE.transfer.OrientedSeam(3, 4),
    )
    assert not MODULE._is_primary_source(
        target,
        MODULE.transfer.OrientedSeam(4, 3),
    )
    assert not MODULE._is_primary_source(
        target,
        MODULE.transfer.OrientedSeam(0, 3),
    )
    assert not MODULE._is_primary_source(
        target,
        MODULE.transfer.OrientedSeam(6, 7),
    )


def test_primary_summary_equal_weights_source_seams() -> None:
    summary = MODULE._summarize_primary_records(
        [
            _record("same_signed_normal", 25.0, 0.8, 0.4),
            _record("same_signed_normal", 25.0, 0.2, -0.2),
        ]
    )

    assert summary is not None
    for mode in MODULE.behavior.MODES:
        assert summary["modes"][mode][
            "source_effect_r_to_target_residual"
        ] == 0.5
        assert summary["modes"][mode][
            "source_effect_specificity_over_target_open_r"
        ] == 0.1
        assert summary["modes"][mode][
            "source_wall_minus_open_r_to_target_residual"
        ] == 0.4


def test_animal_summary_equal_weights_target_queries() -> None:
    queries = []
    for value, pairs in ((0.9, 1), (-0.1, 9)):
        queries.append(
            {
                "eligible_primary_source_pairs": pairs,
                "modes": {
                    mode: {
                        metric: value
                        for metric in MODULE.PRIMARY_METRICS
                    }
                    for mode in MODULE.behavior.MODES
                },
            }
        )

    summary = MODULE._animal_mode_summary(
        queries,
        mode=MODULE.behavior.MODE_WITH_TIME,
    )

    assert summary["eligible_primary_source_pairs"] == 10
    for metric in MODULE.PRIMARY_METRICS:
        assert summary[metric]["mean"] == 0.4
        assert summary[metric]["target_queries"] == 2


def test_adjusted_frame_support_excludes_inaccessible_partition() -> None:
    position = np.array(
        [
            [2.5, 2.5],
            [27.5, 2.5],
            [80.0, 2.5],
        ]
    )
    accessibility = np.ones((15, 15), dtype=bool)
    accessibility[0:5, 0:5] = False

    selected = MODULE._analysis_valid_frames(position, accessibility)

    np.testing.assert_array_equal(selected, [False, True, False])


def test_checkpoint_requires_exact_provenance(tmp_path: Path) -> None:
    source = tmp_path / "QLAK-CA1-test.complete.mat"
    source.write_bytes(b"input")
    argument = argparse.Namespace(
        frames_per_second=30.0,
        velocity_half_window_frames=3,
        speed_cap_quantile=0.995,
        minimum_direction_speed_cm_s=2.0,
        behavior_ridge_fraction=0.001,
        trace_cell_chunk=64,
        minimum_seconds=0.5,
        minimum_bins=6,
        minimum_cells=20,
    )
    metadata = MODULE._cache_metadata(source, argument)
    cache = tmp_path / "checkpoint.json"
    result = {"animal": "test"}
    MODULE._write_cached_animal(cache, metadata, result)

    assert MODULE._read_cached_animal(cache, metadata) == result
    changed = {
        **metadata,
        "settings": {
            **metadata["settings"],
            "minimum_bins": 7,
        },
    }
    assert MODULE._read_cached_animal(cache, changed) is None


def test_checked_in_artifact_hashes_current_implementations() -> None:
    artifact_path = (
        ROOT
        / "results"
        / "source_data"
        / "boundary_fragment_cross_location_behavior_adjusted.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    recorded = artifact["provenance"]["implementation_sha256"]
    implementations = {
        "behavior_map_estimator": (
            ROOT / "scripts" / "run_boundary_fragment_behavior_adjusted.py"
        ),
        "cross_location_scorer": (
            ROOT / "scripts" / "run_boundary_fragment_cross_location_transfer.py"
        ),
        "adjusted_transfer_adapter": (
            ROOT
            / "scripts"
            / "run_boundary_fragment_cross_location_behavior_adjusted.py"
        ),
    }

    assert set(recorded) == set(implementations)
    for name, path in implementations.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert recorded[name] == digest


def test_eligibility_audit_compares_pair_identity_and_support() -> None:
    key = (1, 2, "o", 2, 12, 21, (4, 1))
    baseline = {
        key: (
            30,
            (
                ((3, 4), 30, 15, 14),
                ((6, 7), 30, 15, 13),
            ),
        )
    }

    exact = AUDIT._compare_signatures(baseline, baseline.copy())
    changed = {
        key: (
            30,
            (
                ((3, 4), 30, 15, 14),
                ((6, 7), 30, 15, 12),
            ),
        )
    }
    mismatch = AUDIT._compare_signatures(baseline, changed)

    assert exact["exact_match"]
    assert not mismatch["exact_match"]
    assert mismatch["support_mismatch_queries"]
