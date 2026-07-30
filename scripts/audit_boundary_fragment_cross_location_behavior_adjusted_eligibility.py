"""Audit exact eligibility identity for the behavior-adjusted transport.

The adjusted analysis deliberately uses the stored-map transfer's released
sampling maps, registration masks, templates, strip support, and scorer.  This
audit independently reruns the primary stored-map source selection and checks
the complete target-query and source-pair support signature against the
behavior-adjusted artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ca1_geometry.io import Mat73Animal  # noqa: E402
from ca1_geometry.seams import (  # noqa: E402
    SeamState,
    internal_seams,
    seam_state,
    seam_strip_bins,
)
import run_boundary_fragment_cross_location_behavior_adjusted as adjusted  # noqa: E402
import run_boundary_fragment_cross_location_transfer as transfer  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--adjusted-artifact",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "boundary_fragment_cross_location_behavior_adjusted.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / (
                "boundary_fragment_cross_location_behavior_adjusted_"
                "eligibility_audit.json"
            )
        ),
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-bins", type=int, default=6)
    parser.add_argument("--minimum-cells", type=int, default=20)
    return parser.parse_args()


def _query_key(query: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(query["training_exposure"]),
        int(query["test_exposure"]),
        str(query["target_environment"]),
        int(query["withheld_training_session"]),
        int(query["target_test_session"]),
        int(query["target_test_square_session"]),
        tuple(int(value) for value in query["target_seam"]),
    )


def _pair_signature(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(int(value) for value in record["source_seam"]),
        int(record["cells"]),
        int(record["target_bins"]),
        int(record["source_bins"]),
    )


def _adjusted_signatures(
    animal: dict[str, Any],
) -> dict[tuple[Any, ...], tuple[Any, ...]]:
    output = {}
    for query in animal["queries"]:
        key = _query_key(query)
        pairs = tuple(
            sorted(
                (
                    tuple(int(value) for value in pair["source_seam"]),
                    int(pair["cells"]),
                    int(pair["target_bins"]),
                    int(pair["source_bins"]),
                )
                for pair in query["source_pairs"]
            )
        )
        output[key] = (
            int(query["query_common_cells"]),
            pairs,
        )
    return output


def _stored_map_signatures(
    path: Path,
    *,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> dict[tuple[Any, ...], tuple[Any, ...]]:
    seams = internal_seams()
    strips = {seam: seam_strip_bins(seam) for seam in seams}
    with Mat73Animal(path) as animal:
        environment = [
            animal.environment(session)
            for session in range(animal.n_sessions)
        ]
        blocked = [
            animal.blocked(session)
            for session in range(animal.n_sessions)
        ]
        rate = {
            session: animal.stored_rate_maps(
                session,
                smoothed=False,
            )
            for session in range(animal.n_sessions)
        }
        occupancy = {
            session: animal.sampling_map(session)
            for session in range(animal.n_sessions)
        }
        registered = {
            session: animal.registered_cells(session)
            for session in range(animal.n_sessions)
        }
        repetitions = (animal.n_sessions - 1) // 10

    output: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for training_exposure in range(repetitions - 1):
        training_start = training_exposure * 10
        test_start = (training_exposure + 1) * 10
        for target_offset in range(1, 10):
            target_test = test_start + target_offset
            query_cells = transfer._query_common_cells(
                training_start=training_start,
                test_start=test_start,
                target_offset=target_offset,
                registered=registered,
            )
            if query_cells.size < minimum_cells:
                continue
            for target_seam in seams:
                if (
                    seam_state(blocked[target_test], target_seam)
                    is not SeamState.WALL
                ):
                    continue
                records = []
                for source_seam in seams:
                    if not adjusted._is_primary_source(
                        target_seam,
                        source_seam,
                    ):
                        continue
                    record = transfer._source_record(
                        training_start=training_start,
                        test_start=test_start,
                        target_offset=target_offset,
                        target_seam=target_seam,
                        source_seam=source_seam,
                        blocked=blocked,
                        rate=rate,
                        occupancy=occupancy,
                        registered=registered,
                        query_cells=query_cells,
                        strips=strips,
                        minimum_seconds=minimum_seconds,
                        minimum_bins=minimum_bins,
                        minimum_cells=minimum_cells,
                    )
                    if record is not None:
                        records.append(record)
                if not records:
                    continue
                key = (
                    training_exposure + 1,
                    training_exposure + 2,
                    environment[target_test],
                    training_start + target_offset + 1,
                    target_test + 1,
                    test_start + 11,
                    (target_seam.source, target_seam.target),
                )
                output[key] = (
                    int(query_cells.size),
                    tuple(
                        sorted(_pair_signature(record) for record in records)
                    ),
                )
    return output


def _serializable_key(key: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "training_exposure": key[0],
        "test_exposure": key[1],
        "target_environment": key[2],
        "withheld_training_session": key[3],
        "target_test_session": key[4],
        "target_test_square_session": key[5],
        "target_seam": list(key[6]),
    }


def _compare_signatures(
    expected: dict[tuple[Any, ...], tuple[Any, ...]],
    observed: dict[tuple[Any, ...], tuple[Any, ...]],
) -> dict[str, Any]:
    expected_keys = set(expected)
    observed_keys = set(observed)
    common = expected_keys & observed_keys
    support_mismatches = [
        key for key in sorted(common) if expected[key] != observed[key]
    ]
    expected_pairs = int(
        sum(len(value[1]) for value in expected.values())
    )
    observed_pairs = int(
        sum(len(value[1]) for value in observed.values())
    )
    return {
        "exact_match": (
            expected_keys == observed_keys and not support_mismatches
        ),
        "stored_map_target_queries": len(expected),
        "adjusted_target_queries": len(observed),
        "stored_map_source_pairs": expected_pairs,
        "adjusted_source_pairs": observed_pairs,
        "missing_adjusted_queries": [
            _serializable_key(key)
            for key in sorted(expected_keys - observed_keys)
        ],
        "extra_adjusted_queries": [
            _serializable_key(key)
            for key in sorted(observed_keys - expected_keys)
        ],
        "support_mismatch_queries": [
            _serializable_key(key) for key in support_mismatches
        ],
    }


def main() -> None:
    argument = parse_arguments()
    artifact = json.loads(
        argument.adjusted_artifact.read_text(encoding="utf-8")
    )
    adjusted_animals = {
        animal["animal"]: animal for animal in artifact["animals"]
    }
    paths = sorted(
        argument.data_dir.glob("QLAK-CA1-*.complete.mat")
    )
    results = []
    for path in paths:
        animal_name = path.name.removesuffix(".complete.mat")
        if animal_name not in adjusted_animals:
            continue
        expected = _stored_map_signatures(
            path,
            minimum_seconds=argument.minimum_seconds,
            minimum_bins=argument.minimum_bins,
            minimum_cells=argument.minimum_cells,
        )
        observed = _adjusted_signatures(
            adjusted_animals[animal_name]
        )
        comparison = _compare_signatures(expected, observed)
        results.append(
            {
                "animal": animal_name,
                **comparison,
            }
        )
        print(
            animal_name,
            {
                "exact_match": comparison["exact_match"],
                "queries": comparison["adjusted_target_queries"],
                "pairs": comparison["adjusted_source_pairs"],
            },
            flush=True,
        )

    all_match = (
        len(results) == len(adjusted_animals)
        and all(result["exact_match"] for result in results)
    )
    report = {
        "status": (
            "exact_eligibility_match"
            if all_match
            else "eligibility_mismatch"
        ),
        "question": (
            "Are target query/session/seam identities and the counts of "
            "original common cells and target/source strip bins the same "
            "between the stored-map and behavior-adjusted transports?"
        ),
        "settings": {
            "minimum_seconds": argument.minimum_seconds,
            "minimum_bins": argument.minimum_bins,
            "minimum_cells": argument.minimum_cells,
            "source_midpoint_distance_cm": adjusted.PRIMARY_DISTANCE_CM,
            "orientation_relation": "same_signed_normal",
        },
        "animals_expected": len(adjusted_animals),
        "animals_audited": len(results),
        "all_animals_exact_match": all_match,
        "animals": results,
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if not all_match:
        raise SystemExit("behavior-adjusted eligibility differs from baseline")


if __name__ == "__main__":
    main()
