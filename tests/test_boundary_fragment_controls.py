"""Focused regression tests for stored-map boundary controls."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from ca1_geometry.seams import OrientedSeam


ROOT = Path(__file__).resolve().parents[1]


def _load_controls() -> ModuleType:
    path = ROOT / "scripts" / "run_boundary_fragment_controls.py"
    spec = importlib.util.spec_from_file_location(
        "boundary_fragment_controls",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTROLS = _load_controls()


def test_different_seam_control_excludes_orthogonal_and_opposite_walls() -> None:
    focal = OrientedSeam(0, 1)
    parallel = OrientedSeam(3, 4)
    orthogonal = OrientedSeam(0, 3)
    opposite_facing = OrientedSeam(2, 1)

    pairs = CONTROLS._same_direction_different_pairs(
        [focal],
        [focal, parallel, orthogonal, opposite_facing],
    )

    assert pairs == [(focal, parallel)]
