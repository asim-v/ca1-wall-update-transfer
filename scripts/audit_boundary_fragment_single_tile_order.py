"""Audit the fixed within-cycle order in the single-tile counterfactual.

The exact single-tile result contrasts environment ``u`` (focal wall) with
environment ``o`` (focal seam open).  In every released animal, ``u`` occurs
after ``o`` within every exposure cycle.  This script asks whether a generic
within-cycle order drift, estimated at the *same physical seam* while its
accessible-side state is unchanged, predicts the held-out later target as
well as the actual ``u - o`` edit.

Two placebos are fixed before inspecting their neural outcomes:

``exact_lag``
    Average later-minus-earlier edits from unchanged-state session pairs
    separated by exactly the animal's observed ``u - o`` sequence lag.

``broader_all_lags``
    Average pairwise slopes from every earlier/later unchanged-state pair,
    then multiply by the observed ``u - o`` sequence lag.  Equivalently,
    each pair edit is scaled by ``u_o_lag / pair_lag`` before averaging.

Only OPEN/OPEN and WALL/WALL pairs are used, so the strip lies on an
accessible target tile at both endpoints.  The held-out target environment's
training session is excluded from every placebo.  Within each query and
placebo family, the actual edit, every placebo endpoint, and the held-out
target residual use one identical registered-cell set and one identical set
of spatial bins.  The two placebo families are evaluated separately because
their required session unions differ.

This is a post-outcome falsification/sensitivity audit, not a randomized
order test and not population inference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_boundary_component_validation as validation  # noqa: E402
from ca1_geometry.boundary_fragments import (  # noqa: E402
    common_support_bins,
    spearman_correlation,
)
from ca1_geometry.io import Mat73Animal  # noqa: E402
from ca1_geometry.seams import (  # noqa: E402
    OrientedSeam,
    SeamState,
    internal_seams,
    seam_state,
    seam_strip_bins,
)


ACCESSIBLE_STATES = frozenset((SeamState.OPEN, SeamState.WALL))
VARIANTS = ("exact_lag", "broader_all_lags")
PRIMARY_METRICS = (
    "actual_edit_r",
    "placebo_order_r",
    "actual_minus_placebo_r",
    "drift_adjusted_actual_edit_r",
    "actual_wall_minus_open",
    "placebo_later_minus_earlier_r_mean",
    "actual_endpoint_minus_placebo_endpoint",
)


@dataclass(frozen=True, order=True)
class OrderPair:
    """One earlier/later pair with unchanged accessible focal-seam state."""

    earlier: int
    later: int
    state: SeamState

    @property
    def lag(self) -> int:
        return self.later - self.earlier


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
            / "boundary_fragment_single_tile_order_placebo.json"
        ),
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-bins", type=int, default=6)
    parser.add_argument("--minimum-cells", type=int, default=20)
    parser.add_argument(
        "--depths-cm",
        type=float,
        nargs="+",
        default=(2.5, 7.5, 12.5),
    )
    return parser.parse_args()


def _environment_offset(
    environment: list[str],
    *,
    cycle_start: int,
    name: str,
) -> int:
    matches = [
        offset
        for offset in range(1, 10)
        if environment[cycle_start + offset] == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {name!r} session in cycle at "
            f"{cycle_start}, found {matches}"
        )
    return matches[0]


def _candidate_order_pairs(
    *,
    cycle_start: int,
    target_offset: int,
    blocked: list[tuple[int, ...]],
    seam: OrientedSeam,
    exact_lag: int | None,
) -> tuple[OrderPair, ...]:
    """Return all predeclared same-state order pairs for one query."""

    if not 1 <= target_offset <= 9:
        raise ValueError("target_offset must lie in [1, 9]")
    if exact_lag is not None and exact_lag <= 0:
        raise ValueError("exact_lag must be positive")
    available = [
        cycle_start + offset
        for offset in range(1, 10)
        if offset != target_offset
    ]
    pairs: list[OrderPair] = []
    for earlier_index, earlier in enumerate(available):
        earlier_state = seam_state(blocked[earlier], seam)
        if earlier_state not in ACCESSIBLE_STATES:
            continue
        for later in available[earlier_index + 1 :]:
            if exact_lag is not None and later - earlier != exact_lag:
                continue
            later_state = seam_state(blocked[later], seam)
            if later_state is earlier_state:
                pairs.append(
                    OrderPair(
                        earlier=earlier,
                        later=later,
                        state=earlier_state,
                    )
                )
    return tuple(pairs)


def _required_sessions(
    *,
    train_baseline: int,
    test_baseline: int,
    target_test: int,
    open_session: int,
    wall_session: int,
    pairs: tuple[OrderPair, ...],
) -> tuple[int, ...]:
    """Return the exact session union used for cells and spatial support."""

    sessions = {
        train_baseline,
        test_baseline,
        target_test,
        open_session,
        wall_session,
    }
    for pair in pairs:
        sessions.add(pair.earlier)
        sessions.add(pair.later)
    return tuple(sorted(sessions))


def _lag_standardized_order_vector(
    pair_edits: Iterable[np.ndarray],
    pair_lags: Iterable[int],
    *,
    actual_lag: int,
) -> np.ndarray:
    """Average pairwise slopes and project them over ``actual_lag``."""

    edits = [np.asarray(value, dtype=np.float64) for value in pair_edits]
    lags = np.asarray(list(pair_lags), dtype=np.float64)
    if not edits:
        raise ValueError("at least one pair edit is required")
    if lags.shape != (len(edits),) or np.any(lags <= 0):
        raise ValueError("pair lags must be positive and align with edits")
    if actual_lag <= 0:
        raise ValueError("actual_lag must be positive")
    if any(value.shape != edits[0].shape for value in edits):
        raise ValueError("pair edits must have equal shapes")
    scaled = [
        value * (actual_lag / lag)
        for value, lag in zip(edits, lags, strict=True)
    ]
    return np.mean(np.stack(scaled), axis=0)


def _finite_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float | None:
    value = spearman_correlation(first, second)
    return float(value) if np.isfinite(value) else None


def _difference(
    first: float | None,
    second: float | None,
) -> float | None:
    if first is None or second is None:
        return None
    return float(first - second)


def _session_value(
    session: int,
    *,
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    cells: np.ndarray,
    bins: tuple[tuple[int, int], ...],
    rate_function: validation.RateFunction,
) -> np.ndarray:
    return rate_function(
        rate[session],
        occupancy[session],
        cells,
        bins,
    )


def _analyze_variant(
    *,
    variant: str,
    base_record: dict[str, Any],
    target_offset: int,
    training_exposure: int,
    actual_lag: int,
    environment: list[str],
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    strip: tuple[tuple[int, int], ...],
    pairs: tuple[OrderPair, ...],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Evaluate one query/placebo family on a common data subset."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if not pairs:
        return None, "no_geometry_pairs"
    train_start = training_exposure * 10
    test_start = (training_exposure + 1) * 10
    train_baseline = train_start
    test_baseline = (training_exposure + 2) * 10
    target_test = test_start + target_offset
    open_session = train_start + _environment_offset(
        environment,
        cycle_start=train_start,
        name="o",
    )
    wall_session = train_start + _environment_offset(
        environment,
        cycle_start=train_start,
        name="u",
    )
    required = _required_sessions(
        train_baseline=train_baseline,
        test_baseline=test_baseline,
        target_test=target_test,
        open_session=open_session,
        wall_session=wall_session,
        pairs=pairs,
    )
    cells = np.flatnonzero(
        np.logical_and.reduce([registered[item] for item in required])
    )
    if cells.size < minimum_cells:
        return None, "insufficient_common_cells"
    support = common_support_bins(
        [occupancy[item] for item in required],
        strip,
        minimum_seconds=minimum_seconds,
    )
    if len(support) < minimum_bins:
        return None, "insufficient_common_bins"

    modes: dict[str, dict[str, Any]] = {}
    pair_output: list[dict[str, Any]] = [
        {
            "earlier_session": pair.earlier + 1,
            "later_session": pair.later + 1,
            "earlier_environment": environment[pair.earlier],
            "later_environment": environment[pair.later],
            "lag_in_geometry_sequence": pair.lag,
            "seam_state_both_endpoints": pair.state.value,
            "scale_to_actual_u_o_lag": float(actual_lag / pair.lag),
            "modes": {},
        }
        for pair in pairs
    ]
    for mode, rate_function in validation.RATE_MODES.items():
        values = {
            session: _session_value(
                session,
                rate=rate,
                occupancy=occupancy,
                cells=cells,
                bins=support,
                rate_function=rate_function,
            )
            for session in required
        }
        train_square = values[train_baseline]
        target = values[target_test] - values[test_baseline]
        actual_open = values[open_session] - train_square
        actual_wall = values[wall_session] - train_square
        actual_edit = values[wall_session] - values[open_session]
        pair_edits = [
            values[pair.later] - values[pair.earlier]
            for pair in pairs
        ]
        placebo_order = _lag_standardized_order_vector(
            pair_edits,
            [pair.lag for pair in pairs],
            actual_lag=actual_lag,
        )
        drift_adjusted = actual_edit - placebo_order

        actual_wall_r = _finite_correlation(actual_wall, target)
        actual_open_r = _finite_correlation(actual_open, target)
        actual_edit_r = _finite_correlation(actual_edit, target)
        placebo_order_r = _finite_correlation(placebo_order, target)
        adjusted_r = _finite_correlation(drift_adjusted, target)
        placebo_endpoint_contrasts: list[float] = []
        for pair, edit, pair_item in zip(
            pairs,
            pair_edits,
            pair_output,
            strict=True,
        ):
            earlier = values[pair.earlier] - train_square
            later = values[pair.later] - train_square
            earlier_r = _finite_correlation(earlier, target)
            later_r = _finite_correlation(later, target)
            endpoint_contrast = _difference(later_r, earlier_r)
            if endpoint_contrast is not None:
                placebo_endpoint_contrasts.append(endpoint_contrast)
            pair_item["modes"][mode] = {
                "earlier_r": earlier_r,
                "later_r": later_r,
                "later_minus_earlier_r": endpoint_contrast,
                "raw_edit_r": _finite_correlation(edit, target),
            }
        placebo_endpoint_mean = (
            float(np.mean(placebo_endpoint_contrasts))
            if placebo_endpoint_contrasts
            else None
        )
        actual_endpoint = _difference(actual_wall_r, actual_open_r)
        modes[mode] = {
            "actual_wall_r": actual_wall_r,
            "actual_open_r": actual_open_r,
            "actual_wall_minus_open": actual_endpoint,
            "actual_edit_r": actual_edit_r,
            "placebo_order_r": placebo_order_r,
            "actual_minus_placebo_r": _difference(
                actual_edit_r,
                placebo_order_r,
            ),
            "drift_adjusted_actual_edit_r": adjusted_r,
            "placebo_later_minus_earlier_r_mean": (
                placebo_endpoint_mean
            ),
            "actual_endpoint_minus_placebo_endpoint": _difference(
                actual_endpoint,
                placebo_endpoint_mean,
            ),
        }

    return (
        {
            "variant": variant,
            "training_exposure": base_record["training_exposure"],
            "test_exposure": base_record["test_exposure"],
            "target_environment": base_record["target_environment"],
            "target_offset": target_offset,
            "withheld_training_session": base_record[
                "withheld_training_session"
            ],
            "test_session": base_record["test_session"],
            "training_square_session": train_baseline + 1,
            "test_square_session": test_baseline + 1,
            "seam": base_record["seam"],
            "orientation": base_record["orientation"],
            "actual_open_environment": "o",
            "actual_wall_environment": "u",
            "actual_open_session": open_session + 1,
            "actual_wall_session": wall_session + 1,
            "actual_u_minus_o_lag": actual_lag,
            "placebo_pairs": len(pairs),
            "placebo_pair_states": {
                state.value: int(
                    sum(pair.state is state for pair in pairs)
                )
                for state in (SeamState.OPEN, SeamState.WALL)
            },
            "required_sessions": [item + 1 for item in required],
            "required_session_count": len(required),
            "identical_cells_and_support_within_query": True,
            "cells": int(cells.size),
            "bins": int(len(support)),
            "modes": modes,
            "pairs": pair_output,
        },
        None,
    )


def _cycle_census(
    *,
    animal: str,
    environment: list[str],
    blocked: list[tuple[int, ...]],
    repetitions: int,
) -> dict[str, Any]:
    sequences = [
        environment[cycle * 10 + 1 : cycle * 10 + 10]
        for cycle in range(repetitions)
    ]
    blocked_by_cycle = [
        blocked[cycle * 10 + 1 : cycle * 10 + 10]
        for cycle in range(repetitions)
    ]
    first = sequences[0]
    o_offset = first.index("o") + 1
    u_offset = first.index("u") + 1
    o_blocked = blocked[o_offset]
    u_blocked = blocked[u_offset]
    return {
        "animal": animal,
        "exposure_cycles": repetitions,
        "geometry_order_by_exposure": sequences,
        "geometry_order_repeats_exactly": all(
            sequence == first for sequence in sequences[1:]
        ),
        "blocked_vectors_repeat_exactly": all(
            value == blocked_by_cycle[0]
            for value in blocked_by_cycle[1:]
        ),
        "o_offset": o_offset,
        "u_offset": u_offset,
        "u_after_o": u_offset > o_offset,
        "u_minus_o_lag": u_offset - o_offset,
        "o_blocked_partitions": list(o_blocked),
        "u_blocked_partitions": list(u_blocked),
        "u_o_symmetric_difference": sorted(
            set(o_blocked).symmetric_difference(u_blocked)
        ),
    }


def _metric_summary(
    values_by_animal: dict[str, float],
) -> dict[str, Any]:
    values = np.asarray(list(values_by_animal.values()), dtype=np.float64)
    if values.size == 0:
        return {
            "animals": 0,
            "animal_mean": None,
            "animal_median": None,
            "positive_animals": 0,
            "values_by_animal": {},
        }
    return {
        "animals": int(values.size),
        "animal_mean": float(np.mean(values)),
        "animal_median": float(np.median(values)),
        "positive_animals": int(np.count_nonzero(values > 0)),
        "values_by_animal": values_by_animal,
    }


def _animal_variant_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "records": len(records),
        "placebo_pairs_total": int(
            sum(item["placebo_pairs"] for item in records)
        ),
        "cells_min_median_max": None,
        "bins_min_median_max": None,
        "modes": {},
    }
    if records:
        cell = np.asarray([item["cells"] for item in records])
        bins = np.asarray([item["bins"] for item in records])
        output["cells_min_median_max"] = [
            int(np.min(cell)),
            float(np.median(cell)),
            int(np.max(cell)),
        ]
        output["bins_min_median_max"] = [
            int(np.min(bins)),
            float(np.median(bins)),
            int(np.max(bins)),
        ]
    for mode in validation.RATE_MODES:
        output["modes"][mode] = {}
        for metric in PRIMARY_METRICS:
            value = [
                item["modes"][mode][metric]
                for item in records
                if item["modes"][mode][metric] is not None
            ]
            output["modes"][mode][metric] = {
                "finite_records": len(value),
                "record_mean": (
                    float(np.mean(value)) if value else None
                ),
            }
    return output


def _cohort_variant_summary(
    animals: list[dict[str, Any]],
    *,
    variant: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "animals_with_analyzed_records": int(
            sum(
                animal["variants"][variant]["summary"]["records"] > 0
                for animal in animals
            )
        ),
        "analyzed_records": int(
            sum(
                animal["variants"][variant]["summary"]["records"]
                for animal in animals
            )
        ),
        "geometry_eligible_records": int(
            sum(
                animal["variants"][variant][
                    "geometry_eligible_records"
                ]
                for animal in animals
            )
        ),
        "unavailable_reasons": {},
        "modes": {},
    }
    reasons = sorted(
        {
            reason
            for animal in animals
            for reason in animal["variants"][variant][
                "unavailable_reasons"
            ]
        }
    )
    output["unavailable_reasons"] = {
        reason: int(
            sum(
                animal["variants"][variant][
                    "unavailable_reasons"
                ].get(reason, 0)
                for animal in animals
            )
        )
        for reason in reasons
    }
    for mode in validation.RATE_MODES:
        output["modes"][mode] = {}
        for metric in PRIMARY_METRICS:
            values_by_animal = {}
            for animal in animals:
                value = animal["variants"][variant]["summary"]["modes"][
                    mode
                ][metric]["record_mean"]
                if value is not None:
                    values_by_animal[animal["animal"]] = value
            output["modes"][mode][metric] = _metric_summary(
                values_by_animal
            )
    return output


def analyze_animal(
    path: Path,
    *,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
    strip_depths_cm: tuple[float, ...],
) -> dict[str, Any]:
    seams = internal_seams()
    strips = {
        seam: seam_strip_bins(seam, depths_cm=strip_depths_cm)
        for seam in seams
    }
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

    animal_name = path.name.removesuffix(".complete.mat")
    census = _cycle_census(
        animal=animal_name,
        environment=environment,
        blocked=blocked,
        repetitions=repetitions,
    )
    if not census["geometry_order_repeats_exactly"]:
        raise ValueError(f"{animal_name}: geometry order varies by exposure")
    if not census["blocked_vectors_repeat_exactly"]:
        raise ValueError(f"{animal_name}: blocked vectors vary by exposure")
    if census["u_o_symmetric_difference"] != [5]:
        raise ValueError(
            f"{animal_name}: u/o do not differ only at partition 5"
        )
    actual_lag = int(census["u_minus_o_lag"])
    if actual_lag <= 0:
        raise ValueError(f"{animal_name}: u is not after o")

    variant_records: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in VARIANTS
    }
    geometry_eligible = {variant: 0 for variant in VARIANTS}
    unavailable: dict[str, dict[str, int]] = {
        variant: {} for variant in VARIANTS
    }
    base_records = 0
    base_queries: list[dict[str, Any]] = []
    for training_exposure in range(repetitions - 1):
        train_start = training_exposure * 10
        for target_offset in range(1, 10):
            target_test = (
                (training_exposure + 1) * 10 + target_offset
            )
            for seam in seams:
                if (
                    seam_state(blocked[target_test], seam)
                    is not SeamState.WALL
                ):
                    continue
                base = validation._record(
                    target_offset=target_offset,
                    training_exposure=training_exposure,
                    environment=environment,
                    blocked=blocked,
                    rate=rate,
                    occupancy=occupancy,
                    registered=registered,
                    seam=seam,
                    strip=strips[seam],
                    minimum_seconds=minimum_seconds,
                    minimum_bins=minimum_bins,
                    minimum_cells=minimum_cells,
                    match_nonfocal_context=True,
                    match_global_counterfactual=True,
                )
                if base is None:
                    continue
                base_record, _base_vectors = base
                if (
                    base_record["matching_wall_environments"] != ["u"]
                    or base_record["matching_open_environments"] != ["o"]
                ):
                    raise AssertionError(
                        "strict single-tile record is not exactly u versus o"
                    )
                base_records += 1
                base_queries.append(
                    {
                        "training_exposure": (
                            base_record["training_exposure"]
                        ),
                        "target_environment": (
                            base_record["target_environment"]
                        ),
                        "seam": base_record["seam"],
                    }
                )
                family_pairs = {
                    "exact_lag": _candidate_order_pairs(
                        cycle_start=train_start,
                        target_offset=target_offset,
                        blocked=blocked,
                        seam=seam,
                        exact_lag=actual_lag,
                    ),
                    "broader_all_lags": _candidate_order_pairs(
                        cycle_start=train_start,
                        target_offset=target_offset,
                        blocked=blocked,
                        seam=seam,
                        exact_lag=None,
                    ),
                }
                for variant, pairs in family_pairs.items():
                    if pairs:
                        geometry_eligible[variant] += 1
                    record, reason = _analyze_variant(
                        variant=variant,
                        base_record=base_record,
                        target_offset=target_offset,
                        training_exposure=training_exposure,
                        actual_lag=actual_lag,
                        environment=environment,
                        rate=rate,
                        occupancy=occupancy,
                        registered=registered,
                        strip=strips[seam],
                        pairs=pairs,
                        minimum_seconds=minimum_seconds,
                        minimum_bins=minimum_bins,
                        minimum_cells=minimum_cells,
                    )
                    if record is not None:
                        variant_records[variant].append(record)
                    else:
                        assert reason is not None
                        unavailable[variant][reason] = (
                            unavailable[variant].get(reason, 0) + 1
                        )

    return {
        "animal": animal_name,
        "order_census": census,
        "base_single_tile_eligible_records": base_records,
        "base_queries": base_queries,
        "variants": {
            variant: {
                "geometry_eligible_records": geometry_eligible[variant],
                "unavailable_reasons": unavailable[variant],
                "summary": _animal_variant_summary(
                    variant_records[variant]
                ),
                "records": variant_records[variant],
            }
            for variant in VARIANTS
        },
    }


def main() -> None:
    argument = parse_arguments()
    paths = sorted(
        argument.data_dir.glob("QLAK-CA1-*.complete.mat")
    )
    if not paths:
        raise FileNotFoundError(
            f"no complete animal files found beneath {argument.data_dir}"
        )
    animals: list[dict[str, Any]] = []
    for path in paths:
        result = analyze_animal(
            path,
            minimum_seconds=argument.minimum_seconds,
            minimum_bins=argument.minimum_bins,
            minimum_cells=argument.minimum_cells,
            strip_depths_cm=tuple(argument.depths_cm),
        )
        animals.append(result)
        print(
            result["animal"],
            "lag",
            result["order_census"]["u_minus_o_lag"],
            {
                variant: result["variants"][variant]["summary"][
                    "records"
                ]
                for variant in VARIANTS
            },
        )

    cohort = {
        variant: _cohort_variant_summary(
            animals,
            variant=variant,
        )
        for variant in VARIANTS
    }
    census = {
        "animals": len(animals),
        "animals_u_after_o": int(
            sum(animal["order_census"]["u_after_o"] for animal in animals)
        ),
        "animals_order_repeated_every_exposure": int(
            sum(
                animal["order_census"][
                    "geometry_order_repeats_exactly"
                ]
                for animal in animals
            )
        ),
        "animals_blocked_vectors_repeated_every_exposure": int(
            sum(
                animal["order_census"][
                    "blocked_vectors_repeat_exactly"
                ]
                for animal in animals
            )
        ),
        "u_minus_o_lags_by_animal": {
            animal["animal"]: animal["order_census"]["u_minus_o_lag"]
            for animal in animals
        },
    }
    report = {
        "status": (
            "post_outcome_fixed_order_falsification_and_sensitivity"
        ),
        "question": (
            "Can generic same-seam within-cycle order drift explain the "
            "strict single-tile u-wall versus o-open held-out transfer?"
        ),
        "design": {
            "fixed_order_threat": (
                "u occurs after o in every animal and exposure cycle"
            ),
            "actual_predictor": (
                "u minus o local cell-rate vector at the focal seam"
            ),
            "target": (
                "held-out next-exposure target local rate minus the later "
                "outer-square baseline"
            ),
            "exact_lag_placebo": (
                "mean later-minus-earlier vector across non-target "
                "training pairs with identical accessible seam state and "
                "the exact u-o sequence lag"
            ),
            "broader_all_lags_placebo": (
                "mean pairwise later-minus-earlier slope across every "
                "non-target training pair with identical accessible seam "
                "state, multiplied by the u-o sequence lag"
            ),
            "placebo_states_allowed": ["open", "wall"],
            "target_training_environment_excluded": True,
            "same_physical_seam_for_actual_and_placebo": True,
            "identical_query_cells_and_bins_within_variant": True,
            "variant_supports_evaluated_separately": True,
            "no_outcome_based_pair_selection": True,
            "rate_modes": list(validation.RATE_MODES),
            "inferential_unit_for_descriptive_cohort_means": "animal",
            "randomized_order_test": False,
            "post_outcome_falsification": True,
        },
        "settings": {
            "minimum_seconds_per_common_bin": argument.minimum_seconds,
            "minimum_common_bins": argument.minimum_bins,
            "minimum_common_cells": argument.minimum_cells,
            "strip_depths_cm": list(argument.depths_cm),
        },
        "order_census": census,
        "cohort": cohort,
        "animals": animals,
        "limitations": [
            (
                "Environment order never varies within animal, so no "
                "analysis can identify an order effect independently of "
                "environment identity."
            ),
            (
                "Placebo endpoints are different global geometries and "
                "therefore mix generic sequence drift with geometry-specific "
                "changes."
            ),
            (
                "Exact-lag coverage is structurally sparse; the broader "
                "all-lag result is a labeled sensitivity analysis rather "
                "than an exact placebo."
            ),
            (
                "Queries within an animal are dependent; no query-level "
                "population p-values are reported."
            ),
        ],
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "order_census": census,
                "cohort": cohort,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
