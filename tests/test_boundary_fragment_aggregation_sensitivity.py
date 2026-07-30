"""Focused tests for held-out-record aggregation sensitivity."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    path = (
        ROOT
        / "scripts"
        / "run_boundary_fragment_aggregation_sensitivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "boundary_fragment_aggregation_sensitivity",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SENSITIVITY = _load_script()


def _record(
    value: float | None,
    *,
    pair: tuple[int, int],
    environment: str,
    seam: tuple[int, int],
    orientation: str = "horizontal_wall",
) -> dict[str, object]:
    return {
        "training_exposure": pair[0],
        "test_exposure": pair[1],
        "target_environment": environment,
        "seam": list(seam),
        "orientation": orientation,
        "correlations": {
            "raw_local_rate": {"wall_minus_open": value},
            "global_rate_demeaned": {"wall_minus_open": value},
        },
    }


def test_equal_group_estimators_handle_unequal_record_support() -> None:
    records = [
        _record(8.0, pair=(1, 2), environment="a", seam=(0, 1)),
        _record(4.0, pair=(1, 2), environment="a", seam=(0, 1)),
        _record(0.0, pair=(1, 2), environment="b", seam=(3, 4)),
        _record(-4.0, pair=(2, 3), environment="b", seam=(3, 4)),
    ]

    by_record = SENSITIVITY.aggregate_records(
        records,
        mode="raw_local_rate",
        scheme="prediction_record_equal",
    )
    by_pair = SENSITIVITY.aggregate_records(
        records,
        mode="raw_local_rate",
        scheme="exposure_pair_equal",
    )
    by_environment = SENSITIVITY.aggregate_records(
        records,
        mode="raw_local_rate",
        scheme="target_environment_equal",
    )
    by_seam = SENSITIVITY.aggregate_records(
        records,
        mode="raw_local_rate",
        scheme="exact_oriented_seam_equal",
    )

    assert by_record["wall_minus_open"] == pytest.approx(2.0)
    assert by_pair["wall_minus_open"] == pytest.approx(0.0)
    assert by_environment["wall_minus_open"] == pytest.approx(2.0)
    assert by_seam["wall_minus_open"] == pytest.approx(2.0)
    assert by_pair["finite_groups"] == 2
    assert by_pair["finite_records_per_finite_group"][
        "counts_sorted"
    ] == [1, 3]


def test_exact_oriented_seam_keeps_reverse_directions_separate() -> None:
    records = [
        _record(3.0, pair=(1, 2), environment="a", seam=(0, 1)),
        _record(-1.0, pair=(1, 2), environment="a", seam=(1, 0)),
    ]

    result = SENSITIVITY.aggregate_records(
        records,
        mode="global_rate_demeaned",
        scheme="exact_oriented_seam_equal",
    )

    assert result["finite_groups"] == 2
    assert result["wall_minus_open"] == pytest.approx(1.0)


def test_nonfinite_or_missing_values_are_excluded_with_support_reported() -> None:
    records = [
        _record(2.0, pair=(1, 2), environment="a", seam=(0, 1)),
        _record(None, pair=(1, 2), environment="a", seam=(0, 1)),
        _record(None, pair=(2, 3), environment="b", seam=(3, 4)),
    ]

    result = SENSITIVITY.aggregate_records(
        records,
        mode="raw_local_rate",
        scheme="exposure_pair_equal",
    )

    assert result["wall_minus_open"] == pytest.approx(2.0)
    assert result["finite_records"] == 1
    assert result["total_records"] == 3
    assert result["finite_groups"] == 1
    assert result["total_groups"] == 2


def test_current_record_weighting_reconstructs_committed_source_values() -> None:
    source_path = (
        ROOT
        / "results"
        / "source_data"
        / "boundary_component_validation.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))

    report = SENSITIVITY.build_report(source, source_path=source_path)
    check = report["current_weighting_reconstruction"]

    assert check["passed"] is True
    assert check["maximum_absolute_error_vs_animal_summary"] <= 1e-12
    assert check["maximum_absolute_error_vs_cohort_animal_value"] <= 1e-12
    assert check["maximum_absolute_error_vs_cohort_summary"] <= 1e-12
    assert report["scope"]["not_an_independent_validation"] is True
    assert report["scope"]["biological_unit"] == "animal"
