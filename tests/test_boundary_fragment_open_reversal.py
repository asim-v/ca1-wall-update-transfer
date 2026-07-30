"""Focused tests for symmetric open-target query construction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def _load_script() -> ModuleType:
    path = ROOT / "scripts" / "run_boundary_fragment_open_reversal.py"
    spec = importlib.util.spec_from_file_location(
        "boundary_fragment_open_reversal",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REVERSAL = _load_script()


def _blocked_sessions() -> list[tuple[int, ...]]:
    blocked = [tuple() for _ in range(31)]
    # Seam 0 -> 1 is open in the target test geometry (session 12).
    # Among other geometries in the preceding cycle, sessions 1 and 4 are
    # exact-location walls, while the remaining candidates are open.
    blocked[1] = (0,)
    blocked[4] = (0, 8)
    blocked[2] = (8,)  # withheld target-training geometry: intentionally open
    blocked[12] = (7,)  # target test geometry: open at seam 0 -> 1
    return blocked


def test_open_target_query_reverses_state_and_withholds_target_shape() -> None:
    query = REVERSAL._reversal_query(
        target_offset=2,
        training_exposure=0,
        blocked=_blocked_sessions(),
        seam=REVERSAL.OrientedSeam(0, 1),
    )

    assert query is not None
    assert query.train_baseline == 0
    assert query.test_baseline == 20
    assert query.withheld_target_train == 2
    assert query.target_test == 12
    assert query.matching_wall == (1, 4)
    assert 2 not in query.matching_open
    assert 2 not in query.matching_wall
    assert set(query.matching_open) == {3, 5, 6, 7, 8, 9}


def test_query_rejects_wall_target_and_missing_training_contrast() -> None:
    blocked = _blocked_sessions()
    blocked[12] = (0,)
    assert (
        REVERSAL._reversal_query(
            target_offset=2,
            training_exposure=0,
            blocked=blocked,
            seam=REVERSAL.OrientedSeam(0, 1),
        )
        is None
    )

    blocked = [tuple() for _ in range(31)]
    assert (
        REVERSAL._reversal_query(
            target_offset=2,
            training_exposure=0,
            blocked=blocked,
            seam=REVERSAL.OrientedSeam(0, 1),
        )
        is None
    )
