"""Compare temporal-fold schemes using positions and frozen queries only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ca1_geometry.arena import (
    introduced_boundaries,
    positions_on_accessible_support,
    segment_boundary_queries,
)
from ca1_geometry.io import Mat73Animal
from ca1_geometry.local_linear import LocalMapConfig, fit_local_linear
from ca1_geometry.pilot import (
    balanced_block_folds,
    occupancy_balance_weights,
    query_balanced_block_folds,
    temporal_folds,
)


BLOCKED = (4,)
N_FOLD = 4


def _session_support(
    position: np.ndarray,
    keep: np.ndarray,
    weight: np.ndarray,
    fold: np.ndarray,
    query: np.ndarray,
    bandwidth: float,
) -> tuple[np.ndarray, float | None]:
    config = LocalMapConfig(
        bandwidth=bandwidth,
        min_effective_samples=40.0,
        min_design_eigenratio=0.01,
    )
    local = []
    for fold_index in range(N_FOLD):
        selected = keep & (fold == fold_index) & (weight > 0)
        local.append(
            fit_local_linear(
                position[selected],
                np.ones((selected.sum(), 1)),
                query,
                config,
                sample_weight=weight[selected],
            )
        )
    valid = np.logical_and.reduce([item.valid for item in local])
    effective = np.stack([item.effective_n for item in local])
    minimum = float(np.min(effective[:, valid])) if valid.any() else None
    return valid, minimum


def _animal_result(path: Path, queries: Any) -> dict[str, Any]:
    with Mat73Animal(path) as animal:
        sequences = [
            (start, start + 1, start + 10)
            for start in range(0, animal.n_sessions - 10, 10)
        ]
        output: dict[str, Any] = {}
        for sequence_index, sessions in enumerate(sequences, start=1):
            position = [animal.position(session) for session in sessions]
            keep = [
                positions_on_accessible_support(value, BLOCKED)
                for value in position
            ]
            weights = occupancy_balance_weights(position, keep)
            scheme_fold = {
                "contiguous_quarters": [
                    temporal_folds(value.shape[0], N_FOLD)
                    for value in position
                ],
                "coarse_bin_balanced_b60_g1": [
                    balanced_block_folds(
                        position[index],
                        keep[index],
                        n_fold=N_FOLD,
                        block_frames=1_800,
                        guard_frames=30,
                    )
                    for index in range(3)
                ],
                "query_balanced_b60_g1": [
                    query_balanced_block_folds(
                        position[index],
                        keep[index],
                        queries.position,
                        bandwidth=10.0,
                        n_fold=N_FOLD,
                        block_frames=1_800,
                        guard_frames=30,
                    )
                    for index in range(3)
                ],
            }
            sequence_output: dict[str, Any] = {}
            for scheme, folds in scheme_fold.items():
                bandwidth_output: dict[str, Any] = {}
                for bandwidth in (7.5, 10.0):
                    support = [
                        _session_support(
                            position[index],
                            keep[index],
                            weights[index],
                            folds[index],
                            queries.position,
                            bandwidth,
                        )
                        for index in range(3)
                    ]
                    joint = np.logical_and.reduce(
                        [item[0] for item in support]
                    )
                    near = joint & (queries.distance < 5.0)
                    bandwidth_output[f"{bandwidth:g}_cm"] = {
                        "joint_valid_queries": int(joint.sum()),
                        "near_valid_queries": int(near.sum()),
                        "near_valid_segments": int(
                            np.unique(queries.segment_index[near]).size
                        ),
                        "session_valid_queries": [
                            int(item[0].sum()) for item in support
                        ],
                        "session_minimum_effective_n": [
                            item[1] for item in support
                        ],
                    }
                sequence_output[scheme] = bandwidth_output
            output[f"sequence_{sequence_index}"] = sequence_output
        return {
            "animal": path.stem.replace(".complete", ""),
            "position_only": True,
            "sequences": output,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("animal_files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    argument = parser.parse_args()
    queries = segment_boundary_queries(
        introduced_boundaries(BLOCKED),
        np.array([2.5, 7.5, 12.5]),
        tangential_fractions=np.array([0.4, 0.5, 0.6]),
    )
    result = {
        "query_count": int(queries.position.shape[0]),
        "animals": [
            _animal_result(path, queries) for path in argument.animal_files
        ],
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
