"""Independent-map reliability screen for local boundary aliases.

Rate maps are rebuilt from raw events in alternating one-minute blocks.  The
cross-tile statistic always compares maps estimated from independent blocks;
within-tile split-half reliability is tracked in parallel.
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

from ca1_geometry.aliases import (  # noqa: E402
    cellwise_cross_map_correlations,
    common_relative_support,
    distance_matched_control_pairs,
    exact_alias_pairs,
    mean_fisher_correlation,
)
from ca1_geometry.io import Mat73Animal  # noqa: E402
from ca1_geometry.rate_maps import fixed_rate_maps  # noqa: E402


MAP_MODES = {
    "raw": 0.0,
    "smoothed_sigma_1_bin": 1.0,
}
SUMMARY_KEYS = (
    "alias_cross_half_mean_r",
    "control_cross_half_mean_r",
    "alias_advantage_mean_r",
    "within_tile_reliability_mean_r",
    "alias_cross_half_fisher_r",
    "control_cross_half_fisher_r",
    "alias_advantage_fisher_r",
    "within_tile_reliability_fisher_r",
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
            / "alias_split_half_screen.json"
        ),
    )
    parser.add_argument("--block-seconds", type=int, default=60)
    parser.add_argument("--minimum-seconds", type=float, default=0.25)
    parser.add_argument("--minimum-bins", type=int, default=8)
    parser.add_argument("--minimum-stable-cells", type=int, default=10)
    parser.add_argument(
        "--animal",
        action="append",
        default=[],
        help="Optional animal stem; repeat to select more than one.",
    )
    return parser.parse_args()


def _independent_mean(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    result = np.full(first.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    result[valid] = 0.5 * (first[valid] + second[valid])
    return result


def _stable_summary(
    vectors: list[np.ndarray],
    *,
    minimum_stable_cells: int,
) -> dict[str, Any] | None:
    matrix = np.stack(vectors, axis=0)
    stable = np.all(np.isfinite(matrix), axis=0)
    if np.count_nonzero(stable) < minimum_stable_cells:
        return None
    values = matrix[:, stable]
    return {
        "stable_cells": int(np.count_nonzero(stable)),
        "exposures": [
            {
                "exposure": index + 1,
                "mean_r": float(np.mean(item)),
                "fisher_r": mean_fisher_correlation(item),
                "median_r": float(np.median(item)),
            }
            for index, item in enumerate(values)
        ],
    }


def _build_session_split_maps(
    animal: Mat73Animal,
    session: int,
    cells: np.ndarray,
    *,
    block_seconds: int,
) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray]]:
    position = animal.position(session)
    trace = animal.trace(session, cells)
    if block_seconds <= 0:
        raise ValueError("block_seconds must be positive")
    frame = np.arange(position.shape[0])
    finite = np.all(np.isfinite(position), axis=1) & np.all(
        np.isfinite(trace),
        axis=1,
    )
    inside = finite & np.all((position >= 0.0) & (position <= 75.0), axis=1)
    split = (frame // (30 * block_seconds)) % 2
    rates = {mode: [] for mode in MAP_MODES}
    occupancy = []
    for half in (0, 1):
        keep = inside & (split == half)
        if np.count_nonzero(keep) == 0:
            raise ValueError("empty temporal split")
        unsmoothed = fixed_rate_maps(
            position[keep],
            trace[keep],
            smoothing_sigma_bins=0.0,
        )
        # fixed_rate_maps follows physical y (south-to-north), whereas the
        # released maps and paper partition labels are north-to-south.
        occupancy.append(
            (
                unsmoothed.occupancy
                * np.count_nonzero(keep)
                / 30.0
            )[::-1, :]
        )
        rates["raw"].append(unsmoothed.rate[:, ::-1, :])
        rates["smoothed_sigma_1_bin"].append(
            fixed_rate_maps(
                position[keep],
                trace[keep],
                smoothing_sigma_bins=1.0,
            ).rate[:, ::-1, :]
        )
    return rates, occupancy


def _cross_vectors(
    rates: list[list[np.ndarray]],
    pair: tuple[int, int],
    support: np.ndarray,
) -> list[np.ndarray]:
    result = []
    for halves in rates:
        first = cellwise_cross_map_correlations(
            halves[0],
            halves[1],
            pair,
            support,
        )
        second = cellwise_cross_map_correlations(
            halves[1],
            halves[0],
            pair,
            support,
        )
        result.append(_independent_mean(first, second))
    return result


def _within_vectors(
    rates: list[list[np.ndarray]],
    pair: tuple[int, int],
    support: np.ndarray,
) -> list[np.ndarray]:
    result = []
    for halves in rates:
        first = cellwise_cross_map_correlations(
            halves[0],
            halves[1],
            (pair[0], pair[0]),
            support,
        )
        second = cellwise_cross_map_correlations(
            halves[0],
            halves[1],
            (pair[1], pair[1]),
            support,
        )
        result.append(_independent_mean(first, second))
    return result


def _pair_mode_record(
    *,
    rates: list[list[np.ndarray]],
    occupancy: list[np.ndarray],
    pair: tuple[int, int],
    controls: tuple[tuple[int, int], ...],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_stable_cells: int,
) -> dict[str, Any] | None:
    support = common_relative_support(
        occupancy,
        pair,
        minimum_seconds=minimum_seconds,
    )
    if np.count_nonzero(support) < minimum_bins:
        return None
    alias = _stable_summary(
        _cross_vectors(rates, pair, support),
        minimum_stable_cells=minimum_stable_cells,
    )
    within = _stable_summary(
        _within_vectors(rates, pair, support),
        minimum_stable_cells=minimum_stable_cells,
    )
    if alias is None or within is None:
        return None

    control_records = []
    for control in controls:
        control_support = common_relative_support(
            occupancy,
            control,
            minimum_seconds=minimum_seconds,
        )
        if np.count_nonzero(control_support) < minimum_bins:
            continue
        summary = _stable_summary(
            _cross_vectors(rates, control, control_support),
            minimum_stable_cells=minimum_stable_cells,
        )
        if summary is None:
            continue
        control_records.append(
            {
                "pair": list(control),
                "common_support_bins": int(
                    np.count_nonzero(control_support)
                ),
                **summary,
            }
        )
    if not control_records:
        return None

    exposures = []
    for index in range(len(alias["exposures"])):
        alias_mean = alias["exposures"][index]["mean_r"]
        alias_fisher = alias["exposures"][index]["fisher_r"]
        control_mean = float(
            np.mean(
                [
                    item["exposures"][index]["mean_r"]
                    for item in control_records
                ]
            )
        )
        control_fisher = float(
            np.mean(
                [
                    item["exposures"][index]["fisher_r"]
                    for item in control_records
                ]
            )
        )
        exposures.append(
            {
                "exposure": index + 1,
                "alias_cross_half_mean_r": alias_mean,
                "control_cross_half_mean_r": control_mean,
                "alias_advantage_mean_r": alias_mean - control_mean,
                "within_tile_reliability_mean_r": (
                    within["exposures"][index]["mean_r"]
                ),
                "alias_cross_half_fisher_r": alias_fisher,
                "control_cross_half_fisher_r": control_fisher,
                "alias_advantage_fisher_r": (
                    alias_fisher - control_fisher
                ),
                "within_tile_reliability_fisher_r": (
                    within["exposures"][index]["fisher_r"]
                ),
            }
        )
    return {
        "common_support_bins": int(np.count_nonzero(support)),
        "alias_stable_cells": alias["stable_cells"],
        "within_stable_cells": within["stable_cells"],
        "eligible_controls": len(control_records),
        "exposures": exposures,
        "last_minus_first": {
            key: exposures[-1][key] - exposures[0][key]
            for key in SUMMARY_KEYS
        },
    }


def _aggregate_records(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not records:
        return None
    n_exposures = len(records[0]["exposures"])
    exposures = []
    for index in range(n_exposures):
        record = {"exposure": index + 1}
        for key in SUMMARY_KEYS:
            record[key] = float(
                np.mean([item["exposures"][index][key] for item in records])
            )
        exposures.append(record)
    return {
        "eligible_records": len(records),
        "exposures": exposures,
        "last_minus_first": {
            key: exposures[-1][key] - exposures[0][key]
            for key in SUMMARY_KEYS
        },
    }


def analyze_environment(
    animal: Mat73Animal,
    environment: str,
    sessions: list[int],
    *,
    block_seconds: int,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_stable_cells: int,
) -> dict[str, Any] | None:
    blocked = animal.blocked(sessions[0])
    aliases = exact_alias_pairs(blocked)
    if not aliases:
        return None
    cells = animal.common_registered_cells(*sessions)

    rates_by_mode: dict[str, list[list[np.ndarray]]] = {
        mode: [] for mode in MAP_MODES
    }
    occupancy_by_session: list[list[np.ndarray]] = []
    for session in sessions:
        session_rates, session_occupancy = _build_session_split_maps(
            animal,
            session,
            cells,
            block_seconds=block_seconds,
        )
        for mode in MAP_MODES:
            rates_by_mode[mode].append(session_rates[mode])
        occupancy_by_session.append(session_occupancy)
    occupancy = [
        half
        for session_halves in occupancy_by_session
        for half in session_halves
    ]

    pairs = []
    for pair in aliases:
        modes = {}
        controls = distance_matched_control_pairs(blocked, pair)
        for mode in MAP_MODES:
            result = _pair_mode_record(
                rates=rates_by_mode[mode],
                occupancy=occupancy,
                pair=pair,
                controls=controls,
                minimum_seconds=minimum_seconds,
                minimum_bins=minimum_bins,
                minimum_stable_cells=minimum_stable_cells,
            )
            if result is not None:
                modes[mode] = result
        pairs.append({"alias_pair": list(pair), "modes": modes})

    aggregate = {}
    for mode in MAP_MODES:
        result = _aggregate_records(
            [item["modes"][mode] for item in pairs if mode in item["modes"]]
        )
        if result is not None:
            aggregate[mode] = result
    return {
        "environment": environment,
        "sessions": [session + 1 for session in sessions],
        "common_registered_cells": int(cells.size),
        "pairs": pairs,
        "aggregate": aggregate,
    }


def analyze_animal(
    path: Path,
    *,
    block_seconds: int,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_stable_cells: int,
) -> dict[str, Any]:
    with Mat73Animal(path) as animal:
        sessions_by_environment: dict[str, list[int]] = defaultdict(list)
        for session in range(animal.n_sessions):
            sessions_by_environment[animal.environment(session)].append(
                session
            )
        environments = []
        for environment, sessions in sorted(sessions_by_environment.items()):
            if not exact_alias_pairs(animal.blocked(sessions[0])):
                continue
            result = analyze_environment(
                animal,
                environment,
                sessions,
                block_seconds=block_seconds,
                minimum_seconds=minimum_seconds,
                minimum_bins=minimum_bins,
                minimum_stable_cells=minimum_stable_cells,
            )
            if result is not None:
                environments.append(result)

    aggregate = {}
    for mode in MAP_MODES:
        result = _aggregate_records(
            [
                environment["aggregate"][mode]
                for environment in environments
                if mode in environment["aggregate"]
            ]
        )
        if result is not None:
            aggregate[mode] = result
    return {
        "animal": path.name.removesuffix(".complete.mat"),
        "environments": environments,
        "aggregate": aggregate,
    }


def cohort_summary(animals: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for mode in MAP_MODES:
        eligible = [
            animal["aggregate"][mode]
            for animal in animals
            if mode in animal["aggregate"]
        ]
        mode_result = {
            "eligible_animals": len(eligible),
            "last_minus_first": {},
        }
        for key in SUMMARY_KEYS:
            values = np.asarray(
                [item["last_minus_first"][key] for item in eligible]
            )
            mode_result["last_minus_first"][key] = {
                "animal_mean": float(np.mean(values)),
                "animal_median": float(np.median(values)),
                "negative_animals": int(np.count_nonzero(values < 0)),
                "positive_animals": int(np.count_nonzero(values > 0)),
                "values": values.tolist(),
            }
        result[mode] = mode_result
    return result


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
            block_seconds=argument.block_seconds,
            minimum_seconds=argument.minimum_seconds,
            minimum_bins=argument.minimum_bins,
            minimum_stable_cells=argument.minimum_stable_cells,
        )
        animals.append(result)
        primary = result["aggregate"].get("smoothed_sigma_1_bin")
        print(
            result["animal"],
            None
            if primary is None
            else primary["last_minus_first"][
                "alias_advantage_mean_r"
            ],
            None
            if primary is None
            else primary["last_minus_first"][
                "within_tile_reliability_mean_r"
            ],
        )

    report = {
        "status": (
            "post_outcome_exploratory_split_half_screen_not_confirmatory"
        ),
        "question": (
            "Does the experience trend in local-boundary-twin similarity "
            "survive independent map estimates while within-tile reliability "
            "is tracked separately?"
        ),
        "settings": {
            "temporal_split": (
                f"alternating {argument.block_seconds}-second blocks"
            ),
            "rate_maps": (
                "fixed 5 cm bins rebuilt from raw 30 Hz calcium-rise events"
            ),
            "cross_tile_estimate": (
                "mean of tile A split 1 versus tile B split 2 and the "
                "opposite independent assignment"
            ),
            "minimum_seconds_per_split_pair_member_per_exposure": (
                argument.minimum_seconds
            ),
            "minimum_common_relative_bins": argument.minimum_bins,
            "minimum_stable_cells": argument.minimum_stable_cells,
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
