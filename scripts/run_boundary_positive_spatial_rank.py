"""Fully enumerate matched spatial alternatives for wall-update transfer.

This implements Parts IIB and the spatial-ranking portion of Part I in the
frozen positive-model protocol.  Every admissible source seam at exactly 25 cm
is scored; neural values never select a source or matching tier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_boundary_fragment_cross_location_transfer as transfer  # noqa: E402
from ca1_geometry.io import Mat73Animal  # noqa: E402
from ca1_geometry.positive_models import (  # noqa: E402
    cohort_metric_summary,
    hierarchical_animal_summary,
    query_empirical_rank,
)
from ca1_geometry.seams import (  # noqa: E402
    OrientedSeam,
    SeamState,
    internal_seams,
    seam_frame,
    seam_state,
    seam_strip_bins,
)


LEDGER = (
    ROOT
    / "results"
    / "source_data"
    / "boundary_fragment_cross_location_transfer.json"
)
LEDGER_SHA256 = (
    "6B20DA7167018F8B31700E16996E2C2CCE60587833FBF5EC1100B4E951BA1CC0"
)
EXPECTED_ENUMERATED_QUERIES = 415
EXPECTED_PRIMARY_QUERIES = 405
EXPECTED_PRIMARY_SOURCE_PAIRS = 658
METRICS = (
    "centered_percentile_rank",
    "correct_minus_mean_alternative",
    "correct_minus_best_alternative",
    "centered_correct_win",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results"
            / "positive_model_spatial_rank_v1.json"
        ),
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-bins", type=int, default=6)
    parser.add_argument("--minimum-cells", type=int, default=20)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _midpoint(seam: OrientedSeam) -> np.ndarray:
    start, end, _normal = seam_frame(seam)
    return (start + end) / 2.0


def translation_axis(
    target: OrientedSeam,
    source: OrientedSeam,
) -> str | None:
    """Classify an exact-grid-step translation without neural values."""

    distance = transfer._midpoint_distance_cm(target, source)
    if not np.isclose(distance, 25.0, rtol=0.0, atol=1e-10):
        return None
    _start, _end, normal = seam_frame(target)
    tangent = np.asarray([-normal[1], normal[0]])
    displacement = _midpoint(source) - _midpoint(target)
    normal_cm = float(abs(displacement @ normal))
    tangential_cm = float(abs(displacement @ tangent))
    if np.isclose(normal_cm, 0.0) and np.isclose(tangential_cm, 25.0):
        return "tangential"
    if np.isclose(normal_cm, 25.0) and np.isclose(tangential_cm, 0.0):
        return "normal"
    raise AssertionError("an exact 25 cm lattice step was not axis aligned")


def _candidate_output(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for record in records:
        if not np.isclose(
            record["midpoint_distance_cm"],
            25.0,
            rtol=0.0,
            atol=1e-10,
        ):
            continue
        result.append(
            {
                "source_seam": record["source_seam"],
                "orientation_relation": record[
                    "orientation_relation"
                ],
                "translation_axis": record["translation_axis"],
                "midpoint_distance_cm": record[
                    "midpoint_distance_cm"
                ],
                "cells": record["cells"],
                "target_bins": record["target_bins"],
                "source_bins": record["source_bins"],
                "scores": {
                    mode: record["metrics"][mode][
                        "source_effect_r_to_target_residual"
                    ]
                    for mode in transfer.RATE_MODES
                },
            }
        )
    return result


def analyze_animal(
    path: Path,
    *,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> dict[str, Any]:
    seams = internal_seams()
    strips = {seam: seam_strip_bins(seam) for seam in seams}
    with Mat73Animal(path) as animal:
        environment = [
            animal.environment(session)
            for session in range(animal.n_sessions)
        ]
        blocked = [
            animal.blocked(session) for session in range(animal.n_sessions)
        ]
        rate = {
            session: animal.stored_rate_maps(session, smoothed=False)
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

    enumerated_queries = 0
    primary_queries = 0
    primary_source_pairs = 0
    output_queries: list[dict[str, Any]] = []
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
                    if source_seam.unordered == target_seam.unordered:
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
                    if record is None:
                        continue
                    record["translation_axis"] = translation_axis(
                        target_seam,
                        source_seam,
                    )
                    records.append(record)
                if not any(
                    record["orientation_relation"]
                    == "same_signed_normal"
                    for record in records
                ):
                    continue
                enumerated_queries += 1
                exact_correct = [
                    record
                    for record in records
                    if (
                        record["orientation_relation"]
                        == "same_signed_normal"
                        and np.isclose(
                            record["midpoint_distance_cm"],
                            25.0,
                            rtol=0.0,
                            atol=1e-10,
                        )
                    )
                ]
                if not exact_correct:
                    continue
                primary_queries += 1
                primary_source_pairs += len(exact_correct)
                tiers: dict[str, Any] = {}
                for tier in (
                    "tier1_exact_25cm",
                    "tier2_tangential",
                ):
                    by_mode = {
                        mode: query_empirical_rank(
                            records,
                            mode=mode,
                            tier=tier,
                        )
                        for mode in transfer.RATE_MODES
                    }
                    if any(value is not None for value in by_mode.values()):
                        tiers[tier] = by_mode
                output_queries.append(
                    {
                        "training_exposure": training_exposure + 1,
                        "test_exposure": training_exposure + 2,
                        "target_environment": environment[target_test],
                        "target_test_session": target_test + 1,
                        "target_test_square_session": test_start + 11,
                        "target_seam": [
                            target_seam.source,
                            target_seam.target,
                        ],
                        "query_common_cells": int(query_cells.size),
                        "candidate_sources": _candidate_output(records),
                        "tiers": tiers,
                    }
                )

    summaries: dict[str, Any] = {}
    for tier in ("tier1_exact_25cm", "tier2_tangential"):
        summaries[tier] = {}
        for mode in transfer.RATE_MODES:
            values = []
            for query in output_queries:
                score = query["tiers"].get(tier, {}).get(mode)
                if score is None:
                    continue
                values.append(
                    {
                        "training_exposure": query["training_exposure"],
                        "test_exposure": query["test_exposure"],
                        **{
                            metric: float(score[metric])
                            for metric in METRICS
                        },
                    }
                )
            summaries[tier][mode] = (
                hierarchical_animal_summary(
                    values,
                    metric_names=METRICS,
                )
                if values
                else None
            )
    return {
        "animal": path.name.removesuffix(".complete.mat"),
        "repetitions": repetitions,
        "ledger_reconstruction": {
            "enumerated_queries": enumerated_queries,
            "primary_queries": primary_queries,
            "primary_correct_source_pairs": primary_source_pairs,
        },
        "summaries": summaries,
        "queries": output_queries,
    }


def _cohort_summary(animals: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for tier in ("tier1_exact_25cm", "tier2_tangential"):
        output[tier] = {}
        for mode in transfer.RATE_MODES:
            eligible = [
                {
                    "animal": animal["animal"],
                    "summary": animal["summaries"][tier][mode],
                }
                for animal in animals
                if animal["summaries"][tier][mode] is not None
            ]
            output[tier][mode] = {
                metric: cohort_metric_summary(
                    eligible,
                    metric_name=metric,
                )
                for metric in METRICS
            }
    return output


def main() -> None:
    started = time.perf_counter()
    argument = parse_arguments()
    ledger_hash = _sha256(LEDGER)
    if ledger_hash != LEDGER_SHA256:
        raise RuntimeError(
            f"frozen ledger hash mismatch: {ledger_hash}"
        )
    paths = sorted(
        argument.data_dir.glob("QLAK-CA1-*.complete.mat")
    )
    if len(paths) != 7:
        raise FileNotFoundError(
            f"expected seven complete wall files, found {len(paths)}"
        )
    animals = []
    for path in paths:
        value = analyze_animal(
            path,
            minimum_seconds=argument.minimum_seconds,
            minimum_bins=argument.minimum_bins,
            minimum_cells=argument.minimum_cells,
        )
        animals.append(value)
        print(value["animal"], value["ledger_reconstruction"])

    census = {
        key: int(
            sum(
                animal["ledger_reconstruction"][key]
                for animal in animals
            )
        )
        for key in (
            "enumerated_queries",
            "primary_queries",
            "primary_correct_source_pairs",
        )
    }
    expected = {
        "enumerated_queries": EXPECTED_ENUMERATED_QUERIES,
        "primary_queries": EXPECTED_PRIMARY_QUERIES,
        "primary_correct_source_pairs": EXPECTED_PRIMARY_SOURCE_PAIRS,
    }
    if census != expected:
        raise RuntimeError(
            f"frozen query ledger reconstruction failed: {census}"
        )
    report = {
        "status": "frozen_local_empirical_spatial_rank_v1",
        "protocol_commit": "dac65b4ca608839ffbb12315546806ddea4ba512",
        "implementation_commits": [
            "bf7d438a9b18ee7775e483828bf07b78e1908fda",
            "d992fe6623612197c50e5feeedbdb1e78f2b9b69",
        ],
        "source_ledger": str(LEDGER.relative_to(ROOT)),
        "source_ledger_sha256": ledger_hash,
        "source_dataset_manifest": (
            "data/metadata/zenodo_13993254_manifest.json"
        ),
        "question": (
            "Does the same-signed-normal source relation outrank the full "
            "set of geometry- and support-admissible source seams at the "
            "same exact 25 cm distance?"
        ),
        "design": {
            "inferential_unit": "animal",
            "target_neural_outcome_used_for_source_selection": False,
            "candidate_matching": (
                "all admissible non-target sources at exact 25 cm, with "
                "wall/open training states and frozen common support"
            ),
            "primary_tier": "tier1_exact_25cm",
            "support_limited_tier": "tier2_tangential",
            "aggregation": (
                "sources within query, queries within exposure pair, "
                "exposure pairs within animal, animals within cohort"
            ),
        },
        "settings": {
            "minimum_seconds_per_common_bin": argument.minimum_seconds,
            "minimum_common_bins": argument.minimum_bins,
            "minimum_common_cells": argument.minimum_cells,
            "random_operations": False,
        },
        "ledger_reconstruction": census,
        "cohort": _cohort_summary(animals),
        "animals": animals,
        "execution": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
            "random_operations": False,
            "runtime_seconds": float(time.perf_counter() - started),
        },
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["cohort"], indent=2))


if __name__ == "__main__":
    main()
