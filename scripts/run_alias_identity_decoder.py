"""Cross-validated population decoding of local-boundary twin identity.

The decoder retains every longitudinally registered cell, including silent
cells.  It predicts which member of a twin pair the animal occupies from a
one-second CA1 event window, with whole-minute held-out folds.  A second
decoder uses neural residuals after flexible tile-local position/trajectory
regression.  Exact row/column-displacement controls are evaluated with the
same cells.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ca1_geometry.alias_behavior import (  # noqa: E402
    blocked_conditional_neural_auc,
    blocked_ridge_auc,
    expanded_behavior_features,
    partition_behavior,
    route_js_divergence,
)
from ca1_geometry.aliases import (  # noqa: E402
    displacement_matched_control_pairs,
    exact_alias_pairs,
)
from ca1_geometry.io import Mat73Animal  # noqa: E402


METRICS = (
    "behavior_auc",
    "raw_neural_auc",
    "conditional_neural_auc",
    "route_js_bits",
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
            / "alias_identity_decoder.json"
        ),
    )
    parser.add_argument("--sample-hz", type=int, default=2)
    parser.add_argument("--neural-window-seconds", type=float, default=1.0)
    parser.add_argument("--minimum-class-samples", type=int, default=10)
    parser.add_argument(
        "--animal",
        action="append",
        default=[],
        help="Optional animal stem; repeat to select more than one.",
    )
    return parser.parse_args()


def _sample_session(
    animal: Mat73Animal,
    session: int,
    cells: np.ndarray,
    *,
    sample_hz: int,
    neural_window_seconds: float,
) -> dict[str, Any]:
    if 30 % sample_hz:
        raise ValueError("sample_hz must divide the 30 Hz recording rate")
    stride = 30 // sample_hz
    window_frames = int(round(30 * neural_window_seconds))
    if window_frames <= 0:
        raise ValueError("neural window must include at least one frame")
    half_before = window_frames // 2
    half_after = window_frames - half_before

    position = animal.position(session)
    trace = animal.trace(session, cells)
    candidate = np.arange(
        max(stride, half_before),
        position.shape[0] - half_after,
        stride,
    )
    # Prevent a neural window from crossing the one-minute CV group boundary.
    within_minute = candidate % (30 * 60)
    guard = (within_minute >= half_before) & (
        within_minute + half_after <= 30 * 60
    )
    frame = candidate[guard]
    finite = np.all(np.isfinite(position[frame]), axis=1)
    inside = np.all(
        (position[frame] >= 0.0) & (position[frame] <= 75.0),
        axis=1,
    )
    frame = frame[finite & inside]

    cumulative = np.vstack(
        (
            np.zeros((1, trace.shape[1]), dtype=np.float64),
            np.cumsum(trace, axis=0, dtype=np.float64),
        )
    )
    neural = (
        cumulative[frame + half_after]
        - cumulative[frame - half_before]
    )
    behavior = partition_behavior(position[frame])
    expanded = expanded_behavior_features(behavior)
    # Position one-hots make the common local spatial pattern flexible; the
    # position-by-heading interaction is reserved for a later sensitivity
    # because several pairwise samples are too small for 140 regressors.
    conditional_behavior = expanded[:, : 15 + 25]
    return {
        "behavior": behavior,
        "conditional_behavior": conditional_behavior,
        "neural": neural,
        "sample_index": frame // stride,
    }


def _decode_pair(
    sampled: dict[str, Any],
    pair: tuple[int, int],
    *,
    sample_hz: int,
    minimum_class_samples: int,
) -> dict[str, Any] | None:
    behavior = sampled["behavior"]
    selected = np.isin(behavior.partition, pair)
    label = (behavior.partition[selected] == pair[1]).astype(np.int64)
    counts = [
        int(np.count_nonzero(label == value)) for value in (0, 1)
    ]
    if min(counts) < 2 * minimum_class_samples:
        return None
    index = sampled["sample_index"][selected]
    samples_per_minute = sample_hz * 60
    behavior_auc = blocked_ridge_auc(
        behavior.features[selected],
        label,
        index,
        samples_per_group=samples_per_minute,
        minimum_class_samples=minimum_class_samples,
    )
    raw_auc, conditional_auc = blocked_conditional_neural_auc(
        sampled["conditional_behavior"][selected],
        sampled["neural"][selected],
        label,
        index,
        samples_per_group=samples_per_minute,
        minimum_class_samples=minimum_class_samples,
    )
    route_js, visits = route_js_divergence(
        behavior.partition,
        pair,
        minimum_samples=sample_hz,
    )
    if not np.all(np.isfinite([behavior_auc, raw_auc, conditional_auc])):
        return None
    return {
        "pair": list(pair),
        "samples": counts,
        "qualifying_route_visits": list(visits),
        "behavior_auc": behavior_auc,
        "raw_neural_auc": raw_auc,
        "conditional_neural_auc": conditional_auc,
        "route_js_bits": route_js,
    }


def _pair_across_exposures(
    pair: tuple[int, int],
    controls: tuple[tuple[int, int], ...],
    session_decoders: list[dict[tuple[int, int], dict[str, Any]]],
) -> dict[str, Any] | None:
    if any(pair not in decoder for decoder in session_decoders):
        return None
    eligible_controls = [
        control
        for control in controls
        if all(control in decoder for decoder in session_decoders)
    ]
    if not eligible_controls:
        return None
    exposures = []
    for index, decoder in enumerate(session_decoders):
        target = decoder[pair]
        record: dict[str, Any] = {
            "exposure": index + 1,
            "alias_samples": target["samples"],
            "alias_qualifying_route_visits": (
                target["qualifying_route_visits"]
            ),
        }
        for metric in METRICS:
            alias_value = target[metric]
            control_value = float(
                np.nanmean(
                    [decoder[control][metric] for control in eligible_controls]
                )
            )
            record[f"alias_{metric}"] = alias_value
            record[f"control_{metric}"] = control_value
            record[f"alias_minus_control_{metric}"] = (
                alias_value - control_value
            )
        exposures.append(record)
    keys = [
        key
        for key in exposures[0]
        if key not in {
            "exposure",
            "alias_samples",
            "alias_qualifying_route_visits",
        }
    ]
    return {
        "alias_pair": list(pair),
        "displacement_matched_controls": [
            list(control) for control in eligible_controls
        ],
        "exposures": exposures,
        "last_minus_first": {
            key: exposures[-1][key] - exposures[0][key] for key in keys
        },
    }


def _aggregate_records(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not records:
        return None
    n_exposures = len(records[0]["exposures"])
    numeric_keys = [
        key
        for key in records[0]["exposures"][0]
        if key
        not in {
            "exposure",
            "alias_samples",
            "alias_qualifying_route_visits",
        }
    ]
    exposures = []
    for index in range(n_exposures):
        record: dict[str, Any] = {"exposure": index + 1}
        for key in numeric_keys:
            record[key] = float(
                np.mean([item["exposures"][index][key] for item in records])
            )
        exposures.append(record)
    return {
        "eligible_records": len(records),
        "exposures": exposures,
        "last_minus_first": {
            key: exposures[-1][key] - exposures[0][key]
            for key in numeric_keys
        },
    }


def analyze_environment(
    animal: Mat73Animal,
    environment: str,
    sessions: list[int],
    *,
    sample_hz: int,
    neural_window_seconds: float,
    minimum_class_samples: int,
) -> dict[str, Any] | None:
    blocked = animal.blocked(sessions[0])
    aliases = exact_alias_pairs(blocked)
    if not aliases:
        return None
    cells = animal.common_registered_cells(*sessions)
    session_decoders = []
    all_pairs = set(aliases)
    control_lookup = {}
    for pair in aliases:
        controls = displacement_matched_control_pairs(blocked, pair)
        control_lookup[pair] = controls
        all_pairs.update(controls)

    for session in sessions:
        sampled = _sample_session(
            animal,
            session,
            cells,
            sample_hz=sample_hz,
            neural_window_seconds=neural_window_seconds,
        )
        decoder = {}
        for pair in sorted(all_pairs):
            result = _decode_pair(
                sampled,
                pair,
                sample_hz=sample_hz,
                minimum_class_samples=minimum_class_samples,
            )
            if result is not None:
                decoder[pair] = result
        session_decoders.append(decoder)

    pair_records = []
    for pair in aliases:
        result = _pair_across_exposures(
            pair,
            control_lookup[pair],
            session_decoders,
        )
        if result is not None:
            pair_records.append(result)
    aggregate = _aggregate_records(pair_records)
    if aggregate is None:
        return None
    return {
        "environment": environment,
        "sessions": [session + 1 for session in sessions],
        "common_registered_cells": int(cells.size),
        "pairs": pair_records,
        "aggregate": aggregate,
    }


def analyze_animal(
    path: Path,
    *,
    sample_hz: int,
    neural_window_seconds: float,
    minimum_class_samples: int,
) -> dict[str, Any]:
    with Mat73Animal(path) as animal:
        by_environment: dict[str, list[int]] = defaultdict(list)
        for session in range(animal.n_sessions):
            by_environment[animal.environment(session)].append(session)
        environments = []
        for environment, sessions in sorted(by_environment.items()):
            if not exact_alias_pairs(animal.blocked(sessions[0])):
                continue
            result = analyze_environment(
                animal,
                environment,
                sessions,
                sample_hz=sample_hz,
                neural_window_seconds=neural_window_seconds,
                minimum_class_samples=minimum_class_samples,
            )
            if result is not None:
                environments.append(result)
    aggregate = _aggregate_records(
        [environment["aggregate"] for environment in environments]
    )
    return {
        "animal": path.name.removesuffix(".complete.mat"),
        "environments": environments,
        "aggregate": aggregate,
    }


def cohort_summary(animals: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        animal["aggregate"]
        for animal in animals
        if animal["aggregate"] is not None
    ]
    keys = list(eligible[0]["last_minus_first"]) if eligible else []
    change = {}
    for key in keys:
        values = np.asarray(
            [animal["last_minus_first"][key] for animal in eligible]
        )
        change[key] = {
            "animal_mean": float(np.mean(values)),
            "animal_median": float(np.median(values)),
            "positive_animals": int(np.count_nonzero(values > 0)),
            "negative_animals": int(np.count_nonzero(values < 0)),
            "values": values.tolist(),
        }
    return {
        "eligible_animals": len(eligible),
        "last_minus_first": change,
    }


def _json_safe(value: Any) -> Any:
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
    if argument.animal:
        selected = set(argument.animal)
        paths = [
            path
            for path in paths
            if path.name.removesuffix(".complete.mat") in selected
        ]
    if not paths:
        raise FileNotFoundError("no selected complete animal files found")

    animals = []
    for path in paths:
        result = analyze_animal(
            path,
            sample_hz=argument.sample_hz,
            neural_window_seconds=argument.neural_window_seconds,
            minimum_class_samples=argument.minimum_class_samples,
        )
        animals.append(result)
        aggregate = result["aggregate"]
        print(
            result["animal"],
            None
            if aggregate is None
            else aggregate["last_minus_first"][
                "alias_minus_control_conditional_neural_auc"
            ],
        )

    report = {
        "status": (
            "post_outcome_exploratory_population_decoder_not_confirmatory"
        ),
        "question": (
            "Does held-out CA1 population information about which immediate-"
            "boundary-signature twin is occupied increase with exposure, "
            "beyond measured local behavior and exact displacement controls?"
        ),
        "settings": {
            "sample_hz": argument.sample_hz,
            "neural_window_seconds": argument.neural_window_seconds,
            "neural_features": (
                "event counts for every cell registered across all exposures "
                "of the environment; no activity-based cell selection"
            ),
            "cv": (
                "four whole-minute held-out folds with neural-window guards"
            ),
            "conditional_model": (
                "training-fold multivariate ridge prediction of neural "
                "activity from local 5 cm position bins plus local position, "
                "velocity, acceleration, speed, movement-direction harmonics, "
                "and +/-2 s displacement"
            ),
            "neural_decoder": (
                "training-standardized diagonal nearest-centroid classifier"
            ),
            "controls": (
                "non-alias accessible pairs with identical absolute row and "
                "column displacement, using the same registered-cell set"
            ),
            "minimum_class_samples_per_cv_split": (
                argument.minimum_class_samples
            ),
            "inferential_unit": "animal",
        },
        "cohort": cohort_summary(animals),
        "animals": animals,
    }
    report = _json_safe(report)
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["cohort"], indent=2))


if __name__ == "__main__":
    main()
