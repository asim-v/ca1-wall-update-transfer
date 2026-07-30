"""Screen for reusable registered-cell boundary edits across global shapes.

This is an explicitly exploratory discovery analysis.  The compact output is
safe to track; raw experimental data remain outside Git.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ca1_geometry.boundary_fragments import (  # noqa: E402
    common_support_bins,
    independent_square_residual_correlation,
    local_cell_rate,
)
from ca1_geometry.io import Mat73Animal  # noqa: E402
from ca1_geometry.seams import (  # noqa: E402
    OrientedSeam,
    SeamState,
    internal_seams,
    seam_state,
    seam_strip_bins,
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
            / "source_data"
            / "boundary_fragment_screen.json"
        ),
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-bins", type=int, default=6)
    parser.add_argument("--minimum-cells", type=int, default=20)
    parser.add_argument(
        "--minimum-environment-pairs",
        type=int,
        default=12,
        help="Position-only sequence coverage gate; theoretical maximum is 18.",
    )
    return parser.parse_args()


def _session_cache(
    animal: Mat73Animal,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    list[str],
    list[tuple[int, ...]],
]:
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
    environment = [
        animal.environment(session) for session in range(animal.n_sessions)
    ]
    blocked = [
        animal.blocked(session) for session in range(animal.n_sessions)
    ]
    return rate, occupancy, registered, environment, blocked


def _comparison(
    *,
    sessions: tuple[int, int, int, int],
    seam: OrientedSeam,
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    strip: dict[OrientedSeam, tuple[tuple[int, int], ...]],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> dict[str, Any] | None:
    pre_square, first_target, second_target, post_square = sessions
    cells = np.flatnonzero(
        registered[pre_square]
        & registered[first_target]
        & registered[second_target]
        & registered[post_square]
    )
    if cells.size < minimum_cells:
        return None
    support = common_support_bins(
        [
            occupancy[pre_square],
            occupancy[first_target],
            occupancy[second_target],
            occupancy[post_square],
        ],
        strip[seam],
        minimum_seconds=minimum_seconds,
    )
    if len(support) < minimum_bins:
        return None
    value = {
        session: local_cell_rate(rate[session], cells, support)
        for session in sessions
    }
    correlation = independent_square_residual_correlation(
        value[first_target],
        value[second_target],
        value[pre_square],
        value[post_square],
    )
    if not np.isfinite(correlation.mean):
        return None
    return {
        "correlation": correlation.mean,
        "first_square_assignment": correlation.first_assignment,
        "second_square_assignment": correlation.second_assignment,
        "cells": int(cells.size),
        "bins": int(len(support)),
    }


def _summarize_environment_pair(
    records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    if not records["shared_wall"] or not records["changed_wall"]:
        return None
    shared = np.asarray(
        [item["correlation"] for item in records["shared_wall"]]
    )
    changed = np.asarray(
        [item["correlation"] for item in records["changed_wall"]]
    )
    all_records = records["shared_wall"] + records["changed_wall"]
    return {
        "shared_wall_mean": float(np.mean(shared)),
        "changed_wall_mean": float(np.mean(changed)),
        "contrast": float(np.mean(shared) - np.mean(changed)),
        "shared_wall_seams": int(shared.size),
        "changed_wall_seams": int(changed.size),
        "median_cells": float(
            np.median([item["cells"] for item in all_records])
        ),
        "median_bins": float(
            np.median([item["bins"] for item in all_records])
        ),
    }


def _sequence_screen(
    *,
    exposure: int,
    pre_square: int,
    post_square: int,
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    environment: list[str],
    blocked: list[tuple[int, ...]],
    seams: list[OrientedSeam],
    strip: dict[OrientedSeam, tuple[tuple[int, int], ...]],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
    minimum_environment_pairs: int,
) -> dict[str, Any]:
    targets = range(pre_square + 1, post_square)
    pair_summary: list[dict[str, Any]] = []
    for first_target, second_target in itertools.combinations(targets, 2):
        record: dict[str, list[dict[str, Any]]] = {
            "shared_wall": [],
            "changed_wall": [],
        }
        for seam in seams:
            first_state = seam_state(blocked[first_target], seam)
            second_state = seam_state(blocked[second_target], seam)
            if (
                first_state is SeamState.WALL
                and second_state is SeamState.WALL
            ):
                label = "shared_wall"
            elif {first_state, second_state} == {
                SeamState.WALL,
                SeamState.OPEN,
            }:
                label = "changed_wall"
            else:
                continue
            result = _comparison(
                sessions=(
                    pre_square,
                    first_target,
                    second_target,
                    post_square,
                ),
                seam=seam,
                rate=rate,
                occupancy=occupancy,
                registered=registered,
                strip=strip,
                minimum_seconds=minimum_seconds,
                minimum_bins=minimum_bins,
                minimum_cells=minimum_cells,
            )
            if result is not None:
                record[label].append(result)
        summary = _summarize_environment_pair(record)
        if summary is not None:
            pair_summary.append(
                {
                    "first_environment": environment[first_target],
                    "second_environment": environment[second_target],
                    **summary,
                }
            )

    eligible = len(pair_summary) >= minimum_environment_pairs
    if pair_summary:
        contrast = float(
            np.mean([item["contrast"] for item in pair_summary])
        )
        shared = float(
            np.mean([item["shared_wall_mean"] for item in pair_summary])
        )
        changed = float(
            np.mean([item["changed_wall_mean"] for item in pair_summary])
        )
    else:
        contrast = shared = changed = float("nan")
    return {
        "exposure": exposure,
        "pre_square_session": pre_square + 1,
        "post_square_session": post_square + 1,
        "comparable_environment_pairs": len(pair_summary),
        "coverage_eligible": eligible,
        "shared_wall_mean": shared,
        "changed_wall_mean": changed,
        "shared_minus_changed": contrast,
        "environment_pairs": pair_summary,
    }


def _heldout_environment_screen(
    *,
    exposure: int,
    target_environment: str,
    pre_square: int,
    post_square: int,
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    environment: list[str],
    blocked: list[tuple[int, ...]],
    seams: list[OrientedSeam],
    strip: dict[OrientedSeam, tuple[tuple[int, int], ...]],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> dict[str, Any]:
    target = next(
        session
        for session in range(pre_square + 1, post_square)
        if environment[session] == target_environment
    )
    target_walls = [
        seam
        for seam in seams
        if seam_state(blocked[target], seam) is SeamState.WALL
    ]
    training_summaries = []
    for training in range(pre_square + 1, post_square):
        if training == target:
            continue
        record: dict[str, list[dict[str, Any]]] = {
            "shared_wall": [],
            "changed_wall": [],
        }
        for seam in target_walls:
            training_state = seam_state(blocked[training], seam)
            if training_state is SeamState.WALL:
                label = "shared_wall"
            elif training_state is SeamState.OPEN:
                label = "changed_wall"
            else:
                continue
            result = _comparison(
                sessions=(pre_square, target, training, post_square),
                seam=seam,
                rate=rate,
                occupancy=occupancy,
                registered=registered,
                strip=strip,
                minimum_seconds=minimum_seconds,
                minimum_bins=minimum_bins,
                minimum_cells=minimum_cells,
            )
            if result is not None:
                record[label].append(result)
        summary = _summarize_environment_pair(record)
        if summary is not None:
            training_summaries.append(
                {
                    "training_environment": environment[training],
                    **summary,
                }
            )
    contrast = (
        float(
            np.mean(
                [item["contrast"] for item in training_summaries]
            )
        )
        if training_summaries
        else float("nan")
    )
    return {
        "exposure": exposure,
        "target_environment": target_environment,
        "target_session": target + 1,
        "comparable_training_environments": len(training_summaries),
        "shared_minus_changed": contrast,
        "training_environments": training_summaries,
    }


def analyze_animal(
    path: Path,
    *,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
    minimum_environment_pairs: int,
) -> dict[str, Any]:
    seams = internal_seams()
    strip = {seam: seam_strip_bins(seam) for seam in seams}
    with Mat73Animal(path) as animal:
        (
            rate,
            occupancy,
            registered,
            environment,
            blocked,
        ) = _session_cache(animal)
        repetitions = (animal.n_sessions - 1) // 10
        sequences = []
        heldout_environments: dict[str, list[dict[str, Any]]] = {
            name: []
            for name in sorted(set(environment) - {"square"})
        }
        for repetition in range(repetitions):
            pre_square = repetition * 10
            post_square = pre_square + 10
            sequence = _sequence_screen(
                exposure=repetition + 1,
                pre_square=pre_square,
                post_square=post_square,
                rate=rate,
                occupancy=occupancy,
                registered=registered,
                environment=environment,
                blocked=blocked,
                seams=seams,
                strip=strip,
                minimum_seconds=minimum_seconds,
                minimum_bins=minimum_bins,
                minimum_cells=minimum_cells,
                minimum_environment_pairs=minimum_environment_pairs,
            )
            sequences.append(sequence)
            for target_environment in heldout_environments:
                heldout_environments[target_environment].append(
                    _heldout_environment_screen(
                        exposure=repetition + 1,
                        target_environment=target_environment,
                        pre_square=pre_square,
                        post_square=post_square,
                        rate=rate,
                        occupancy=occupancy,
                        registered=registered,
                        environment=environment,
                        blocked=blocked,
                        seams=seams,
                        strip=strip,
                        minimum_seconds=minimum_seconds,
                        minimum_bins=minimum_bins,
                        minimum_cells=minimum_cells,
                    )
                )

    eligible = [
        item["shared_minus_changed"]
        for item in sequences
        if item["coverage_eligible"]
        and np.isfinite(item["shared_minus_changed"])
    ]
    all_sequence = [
        item["shared_minus_changed"]
        for item in sequences
        if np.isfinite(item["shared_minus_changed"])
    ]
    heldout_mean = {}
    for name, values in heldout_environments.items():
        finite = [
            item["shared_minus_changed"]
            for item in values
            if np.isfinite(item["shared_minus_changed"])
        ]
        heldout_mean[name] = (
            float(np.mean(finite)) if finite else None
        )
    return {
        "animal": path.name.removesuffix(".complete.mat"),
        "sessions": len(environment),
        "coverage_eligible_sequences": len(eligible),
        "eligible_animal_mean": (
            float(np.mean(eligible)) if eligible else None
        ),
        "all_sequence_mean": (
            float(np.mean(all_sequence)) if all_sequence else None
        ),
        "heldout_plus_mean": heldout_mean.get("+"),
        "heldout_environment_mean": heldout_mean,
        "sequences": sequences,
        "heldout_environments": heldout_environments,
    }


def cohort_summary(animals: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        item["eligible_animal_mean"]
        for item in animals
        if item["eligible_animal_mean"] is not None
    ]
    all_animal = [
        item["all_sequence_mean"]
        for item in animals
        if item["all_sequence_mean"] is not None
    ]
    plus = [
        item["heldout_plus_mean"]
        for item in animals
        if item["heldout_plus_mean"] is not None
    ]
    heldout_by_environment = {}
    environment_names = sorted(
        {
            name
            for animal in animals
            for name in animal["heldout_environment_mean"]
        }
    )
    for name in environment_names:
        value = [
            animal["heldout_environment_mean"][name]
            for animal in animals
            if animal["heldout_environment_mean"][name] is not None
        ]
        heldout_by_environment[name] = {
            "animals": len(value),
            "animal_mean": (
                float(np.mean(value)) if value else None
            ),
            "animal_median": (
                float(np.median(value)) if value else None
            ),
            "positive_animals": int(
                np.count_nonzero(np.asarray(value) > 0)
            ),
        }
    eligible_sequences = [
        sequence
        for animal in animals
        for sequence in animal["sequences"]
        if sequence["coverage_eligible"]
    ]
    return {
        "eligible_animals": len(eligible),
        "eligible_sequences": len(eligible_sequences),
        "positive_eligible_animals": int(
            np.count_nonzero(np.asarray(eligible) > 0)
        ),
        "positive_eligible_sequences": int(
            np.count_nonzero(
                [
                    item["shared_minus_changed"] > 0
                    for item in eligible_sequences
                ]
            )
        ),
        "eligible_animal_mean": (
            float(np.mean(eligible)) if eligible else None
        ),
        "eligible_animal_median": (
            float(np.median(eligible)) if eligible else None
        ),
        "all_animal_mean": (
            float(np.mean(all_animal)) if all_animal else None
        ),
        "positive_all_animals": int(
            np.count_nonzero(np.asarray(all_animal) > 0)
        ),
        "heldout_plus_animal_mean": (
            float(np.mean(plus)) if plus else None
        ),
        "positive_heldout_plus_animals": int(
            np.count_nonzero(np.asarray(plus) > 0)
        ),
        "heldout_by_environment": heldout_by_environment,
    }


def _json_safe(value: Any) -> Any:
    """Replace non-finite NumPy/Python scalars before strict JSON output."""

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    argument = parse_arguments()
    paths = sorted(
        argument.data_dir.glob("QLAK-CA1-*.complete.mat")
    )
    if not paths:
        raise FileNotFoundError(
            f"no complete animal files found beneath {argument.data_dir}"
        )
    animals = []
    for path in paths:
        result = analyze_animal(
            path,
            minimum_seconds=argument.minimum_seconds,
            minimum_bins=argument.minimum_bins,
            minimum_cells=argument.minimum_cells,
            minimum_environment_pairs=argument.minimum_environment_pairs,
        )
        animals.append(result)
        print(
            result["animal"],
            result["eligible_animal_mean"],
            result["heldout_plus_mean"],
        )

    report = {
        "status": (
            "exploratory_post_pilot_screen_not_confirmatory_inference"
        ),
        "question": (
            "Are registered-cell rate edits more reproducible across "
            "different global shapes when the same oriented seam is walled "
            "in both than when that seam changes between wall and open?"
        ),
        "settings": {
            "rate_maps": "released_unsmoothed_5_cm_event_probability",
            "strip_depth_cm": 15.0,
            "strip_length_cm": 25.0,
            "minimum_seconds_per_common_bin": argument.minimum_seconds,
            "minimum_common_bins": argument.minimum_bins,
            "minimum_common_cells": argument.minimum_cells,
            "minimum_environment_pairs_per_sequence": (
                argument.minimum_environment_pairs
            ),
            "correlation": "spearman",
            "baseline": (
                "average of correlations under opposite independent "
                "pre/post-square assignments"
            ),
            "inferential_unit": "animal",
        },
        "cohort": cohort_summary(animals),
        "animals": animals,
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    report = _json_safe(report)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["cohort"], indent=2))


if __name__ == "__main__":
    main()
