"""Tests for the exact mirror-lag open spatial control."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    path = (
        ROOT
        / "scripts"
        / "audit_boundary_fragment_cross_location_mirror_open.py"
    )
    spec = importlib.util.spec_from_file_location(
        "boundary_fragment_cross_location_mirror_open",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_script()


def test_mirror_control_preserves_orientation_distance_and_axis() -> None:
    seams = tuple(AUDIT.internal_seams())
    source = AUDIT.OrientedSeam(3, 4)
    target = AUDIT.OrientedSeam(0, 1)
    control = AUDIT._mirror_control_seam(
        source,
        target,
        seams=seams,
    )

    assert control == AUDIT.OrientedSeam(6, 7)
    assert AUDIT._translation_axis(source, target) == "tangential"
    assert AUDIT._translation_axis(source, control) == "tangential"
    assert np.isclose(
        AUDIT.transfer._midpoint_distance_cm(source, target),
        25.0,
    )
    assert np.isclose(
        AUDIT.transfer._midpoint_distance_cm(source, control),
        25.0,
    )


def test_edge_source_without_internal_mirror_is_ineligible() -> None:
    seams = tuple(AUDIT.internal_seams())
    source = AUDIT.OrientedSeam(0, 1)
    target = AUDIT.OrientedSeam(3, 4)
    assert (
        AUDIT._mirror_control_seam(
            source,
            target,
            seams=seams,
        )
        is None
    )


def test_translated_bin_pairs_are_bijective_and_relative() -> None:
    target = AUDIT.OrientedSeam(0, 1)
    control = AUDIT.OrientedSeam(6, 7)
    pairs = AUDIT._translated_bin_pairs(target, control)

    assert len(pairs) == 15
    assert len({first for first, _second in pairs}) == 15
    assert len({second for _first, second in pairs}) == 15
    shifts = {
        (second[0] - first[0], second[1] - first[1])
        for first, second in pairs
    }
    assert shifts == {(10, 0)}


def test_matched_support_drops_both_members_of_a_bin_pair() -> None:
    target = AUDIT.OrientedSeam(0, 1)
    control = AUDIT.OrientedSeam(6, 7)
    occupancy = {
        1: np.ones((15, 15), dtype=np.float64),
        2: np.ones((15, 15), dtype=np.float64),
    }
    first_pair = AUDIT._translated_bin_pairs(target, control)[0]
    occupancy[1][first_pair[1]] = 0.1

    target_bins, control_bins = AUDIT._matched_evaluation_bins(
        occupancy=occupancy,
        sessions=(1, 2),
        target=target,
        control=control,
        minimum_seconds=0.5,
    )

    assert len(target_bins) == len(control_bins) == 14
    assert first_pair[0] not in target_bins
    assert first_pair[1] not in control_bins


def test_primary_advantage_and_within_session_difference_are_distinct() -> None:
    source = np.arange(8, dtype=np.float64)
    wall = source.copy()
    mirror = source[::-1]
    metrics = AUDIT._mode_metrics(
        source_effect=source,
        wall_target=wall,
        mirror_open=mirror,
    )

    assert metrics is not None
    assert np.isclose(
        metrics["wall_minus_mirror_open_correlation_advantage"],
        2.0,
    )
    assert np.isclose(
        metrics[
            "source_effect_r_to_within_session_wall_minus_mirror_open"
        ],
        1.0,
    )


def test_checked_in_artifact_has_locked_schema_and_headlines() -> None:
    path = (
        ROOT
        / "results"
        / "source_data"
        / "boundary_fragment_cross_location_mirror_open.json"
    )
    assert path.exists()
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["design"]["source_distance_matched_cm"] == 25.0
    assert report["design"]["same_target_session_and_global_geometry"]
    assert report["design"]["paired_relative_target_control_bins"]
    assert report["design"]["primary_metric"].startswith(
        "source-effect r to wall target minus"
    )
    cohort = report["cohort_descriptive"]
    assert cohort["animals_with_eligible_queries"] == 5
    assert cohort["eligible_target_queries"] == 72
    assert (
        cohort["eligible_source_wall_open_mirror_triplets"]
        == 72
    )
    demeaned = cohort["modes"]["global_rate_demeaned"]
    primary = demeaned[
        "wall_minus_mirror_open_correlation_advantage"
    ]
    assert np.isclose(primary["animal_mean"], 0.11379738784064826)
    assert primary["positive_animals"] == 5
    wall = demeaned["source_effect_r_to_wall_target"]
    assert np.isclose(wall["animal_mean"], 0.06765917649091271)
    assert wall["positive_animals"] == 4
    direct = demeaned[
        "source_effect_r_to_within_session_wall_minus_mirror_open"
    ]
    assert np.isclose(direct["animal_mean"], 0.07706701867121425)
    assert direct["positive_animals"] == 4
    raw_primary = cohort["modes"]["raw_local_rate"][
        "wall_minus_mirror_open_correlation_advantage"
    ]
    assert raw_primary["positive_animals"] == 3
    for animal in report["animals"]:
        for query in animal["queries"]:
            withheld = query["withheld_training_session"]
            for record in query["records"]:
                assert record["translation_axis"] == "tangential"
                assert record["target_wall_state"] == "wall"
                assert record["mirror_control_state"] == "open"
                assert record["paired_relative_evaluation_bins"]
                assert np.isclose(
                    record[
                        "source_to_wall_midpoint_distance_cm"
                    ],
                    record[
                        "source_to_open_midpoint_distance_cm"
                    ],
                )
                assert withheld not in (
                    record["source_wall_training_sessions"]
                    + record["source_open_training_sessions"]
                )
