"""Rank deformation conditions using only foldwise behavioral support."""

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
    occupancy_balance_weights,
    query_balanced_block_folds,
)


N_FOLD = 4
BANDWIDTH_CM = 10.0


def _support(
    position: np.ndarray,
    keep: np.ndarray,
    weight: np.ndarray,
    fold: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    config = LocalMapConfig(
        bandwidth=BANDWIDTH_CM,
        min_effective_samples=40.0,
        min_design_eigenratio=0.01,
    )
    valid = []
    for fold_index in range(N_FOLD):
        selected = keep & (fold == fold_index) & (weight > 0)
        result = fit_local_linear(
            position[selected],
            np.ones((selected.sum(), 1)),
            query,
            config,
            sample_weight=weight[selected],
        )
        valid.append(result.valid)
    return np.logical_and.reduce(valid)


def _condition_result(
    animal: Mat73Animal,
    offset: int,
    n_sequence: int,
) -> dict[str, Any]:
    condition_sessions = [10 * index + offset for index in range(n_sequence)]
    environment = animal.environment(condition_sessions[0])
    blocked = animal.blocked(condition_sessions[0])
    if any(
        animal.environment(session) != environment
        or animal.blocked(session) != blocked
        for session in condition_sessions
    ):
        raise ValueError("condition identity changed between sequences")
    segments = introduced_boundaries(blocked)
    queries = segment_boundary_queries(
        segments,
        np.array([2.5, 7.5, 12.5]),
        tangential_fractions=np.array([0.4, 0.5, 0.6]),
    )

    sequence_result = []
    for sequence_index, condition_session in enumerate(condition_sessions):
        sessions = (
            10 * sequence_index,
            condition_session,
            10 * (sequence_index + 1),
        )
        position = [animal.position(session) for session in sessions]
        keep = [
            positions_on_accessible_support(value, blocked)
            for value in position
        ]
        weights = occupancy_balance_weights(position, keep)
        folds = [
            query_balanced_block_folds(
                position[index],
                keep[index],
                queries.position,
                bandwidth=BANDWIDTH_CM,
                n_fold=N_FOLD,
                block_frames=1_800,
                guard_frames=30,
            )
            for index in range(3)
        ]
        session_valid = [
            _support(
                position[index],
                keep[index],
                weights[index],
                folds[index],
                queries.position,
            )
            for index in range(3)
        ]
        joint = np.logical_and.reduce(session_valid)
        near = joint & (queries.distance < 5.0)
        valid_segments = np.unique(queries.segment_index[near])
        sequence_result.append(
            {
                "sequence": sequence_index + 1,
                "sessions_one_based": [
                    session + 1 for session in sessions
                ],
                "joint_valid_queries": int(joint.sum()),
                "query_total": int(joint.size),
                "near_valid_queries": int(near.sum()),
                "near_query_total": int(
                    np.count_nonzero(queries.distance < 5.0)
                ),
                "near_valid_segments": int(valid_segments.size),
                "segment_total": len(segments),
                "session_valid_queries": [
                    int(value.sum()) for value in session_valid
                ],
            }
        )

    near_fraction = [
        value["near_valid_queries"] / value["near_query_total"]
        for value in sequence_result
    ]
    segment_fraction = [
        value["near_valid_segments"] / value["segment_total"]
        for value in sequence_result
    ]
    joint_fraction = [
        value["joint_valid_queries"] / value["query_total"]
        for value in sequence_result
    ]
    return {
        "offset_within_sequence": offset,
        "environment": environment,
        "blocked": list(blocked),
        "boundary_segments": len(segments),
        "sequences": sequence_result,
        "support_score": {
            "minimum_near_query_fraction": min(near_fraction),
            "minimum_near_segment_fraction": min(segment_fraction),
            "minimum_all_query_fraction": min(joint_fraction),
            "mean_near_query_fraction": float(np.mean(near_fraction)),
        },
    }


def _rank_key(value: dict[str, Any]) -> tuple[float, ...]:
    score = value["support_score"]
    return (
        score["minimum_near_query_fraction"],
        score["minimum_near_segment_fraction"],
        score["minimum_all_query_fraction"],
        score["mean_near_query_fraction"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("animal_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    argument = parser.parse_args()
    with Mat73Animal(argument.animal_file) as animal:
        n_sequence = (animal.n_sessions - 1) // 10
        condition = [
            _condition_result(animal, offset, n_sequence)
            for offset in range(1, 10)
        ]
    ranked = sorted(condition, key=_rank_key, reverse=True)
    result = {
        "animal": argument.animal_file.stem.replace(".complete", ""),
        "position_only": True,
        "fold_scheme": "query-balanced 60-second blocks, 1-second guards",
        "bandwidth_cm": BANDWIDTH_CM,
        "ranking": [
            {
                "rank": index + 1,
                **value,
            }
            for index, value in enumerate(ranked)
        ],
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    compact = [
        {
            "rank": index + 1,
            "environment": value["environment"],
            **value["support_score"],
            "near_queries_by_sequence": [
                item["near_valid_queries"] for item in value["sequences"]
            ],
            "near_segments_by_sequence": [
                item["near_valid_segments"] for item in value["sequences"]
            ],
        }
        for index, value in enumerate(ranked)
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
