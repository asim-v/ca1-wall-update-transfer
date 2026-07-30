"""Tests for the context-matched focal-seam analysis."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATION = _load_script(
    "boundary_component_validation_context_test",
    "scripts/run_boundary_component_validation.py",
)


def test_nonfocal_context_excludes_only_the_focal_source() -> None:
    seam = VALIDATION.OrientedSeam(4, 1)

    context = VALIDATION.nonfocal_target_context((0, 4), seam)

    assert context == ((0, True), (2, False))


def test_nonfocal_context_is_invariant_to_unrelated_blocked_tiles() -> None:
    seam = VALIDATION.OrientedSeam(4, 1)

    first = VALIDATION.nonfocal_target_context((0, 4), seam)
    second = VALIDATION.nonfocal_target_context((0, 4, 6, 8), seam)

    assert first == second


def test_empty_exposure_summary_is_strict_json_safe() -> None:
    summary = VALIDATION._summarize_records(
        [],
        mode="global_rate_demeaned",
    )

    assert summary["local_predictions"] == 0
    assert summary["wall_minus_open_mean"] is None
    json.dumps(summary, allow_nan=False)


def test_tracked_context_matched_artifact_obeys_design() -> None:
    path = (
        ROOT
        / "results"
        / "source_data"
        / "boundary_fragment_context_matched.json"
    )
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["design"][
        "target_neural_rates_excluded_from_template_fitting"
    ]
    assert report["support"]["eligible_local_predictions"] == 252
    for animal in report["animals"]:
        for record in animal["records"]:
            assert record["nonfocal_context_matched"]
            assert record["target_environment"] not in (
                record["matching_wall_environments"]
            )
            assert record["target_environment"] not in (
                record["matching_open_environments"]
            )
            assert record["matching_wall_environments"]
            assert record["matching_open_environments"]


def test_tracked_single_tile_counterfactual_uses_only_u_vs_o() -> None:
    path = (
        ROOT
        / "results"
        / "source_data"
        / "boundary_fragment_single_tile_counterfactual.json"
    )
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["support"]["eligible_local_predictions"] == 46
    for animal in report["animals"]:
        for record in animal["records"]:
            assert record["global_counterfactual_matched"]
            assert record["seam"][0] == 5
            assert record["matching_wall_environments"] == ["u"]
            assert record["matching_open_environments"] == ["o"]
            assert record["target_environment"] in {"i", "l", "t"}
