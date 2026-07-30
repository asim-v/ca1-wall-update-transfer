"""Coherent registered-cell permutation for cross-location transfer.

Within each animal and draw, one constrained global-ID permutation is applied
to every held-out target residual and both rate modes. Training source
wall-minus-open vectors remain fixed. Permutations are restricted to global
cell-ID strata with identical inclusion across all target-query common-cell
sets, so every query retains exactly its observed cells.

This is an identity-dependence diagnostic, not population inference. It
preserves query/source reuse by aggregating exact-25-cm source correlations
within target query and target queries within animal on every draw.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, NamedTuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_boundary_fragment_cross_location_transfer as transfer  # noqa: E402
import run_boundary_fragment_session_permutation as identity  # noqa: E402
from ca1_geometry.boundary_fragments import common_support_bins  # noqa: E402
from ca1_geometry.io import Mat73Animal  # noqa: E402
from ca1_geometry.seams import (  # noqa: E402
    OrientedSeam,
    SeamState,
    internal_seams,
    seam_state,
    seam_strip_bins,
)


class PreparedQuery(NamedTuple):
    """Rank vectors for exact-distance sources sharing one target outcome."""

    cell_ids: np.ndarray
    same_source: dict[str, np.ndarray]
    opposite_source: dict[str, np.ndarray]
    target: dict[str, np.ndarray]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "boundary_fragment_cross_location_transfer.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results"
            / "source_data"
            / "boundary_fragment_cross_location_cell_permutation.json"
        ),
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-bins", type=int, default=6)
    parser.add_argument("--minimum-cells", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def _source_and_target_vectors(
    *,
    training_start: int,
    test_start: int,
    target_offset: int,
    target_seam: OrientedSeam,
    source_seam: OrientedSeam,
    blocked: list[tuple[int, ...]],
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    query_cells: np.ndarray,
    strips: dict[OrientedSeam, tuple[tuple[int, int], ...]],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> tuple[
    str,
    dict[str, tuple[np.ndarray, np.ndarray]],
] | None:
    """Reconstruct one eligible source record and its primary vectors."""

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
        return None
    templates = transfer._template_sessions(
        training_start=training_start,
        target_offset=target_offset,
        target_seam=target_seam,
        source_seam=source_seam,
        blocked=blocked,
    )
    if templates is None:
        raise AssertionError("eligible source record lost its templates")

    training_baseline = training_start
    target_test = test_start + target_offset
    test_baseline = test_start + 10
    target_support = common_support_bins(
        [
            occupancy[session]
            for session in (
                training_baseline,
                test_baseline,
                target_test,
                *templates.target_wall,
                *templates.target_open,
            )
        ],
        strips[target_seam],
        minimum_seconds=minimum_seconds,
    )
    source_support = common_support_bins(
        [
            occupancy[session]
            for session in (
                training_baseline,
                *templates.source_wall,
                *templates.source_open,
            )
        ],
        strips[source_seam],
        minimum_seconds=minimum_seconds,
    )
    if (
        len(target_support) < minimum_bins
        or len(source_support) < minimum_bins
    ):
        raise AssertionError("eligible source record lost spatial support")

    cells = np.asarray(query_cells, dtype=np.int64)
    output = {}
    for mode, rate_function in transfer.RATE_MODES.items():
        source_wall = transfer._mean_residual(
            templates.source_wall,
            baseline=training_baseline,
            seam_bins=source_support,
            cells=cells,
            rate=rate,
            occupancy=occupancy,
            rate_function=rate_function,
        )
        source_open = transfer._mean_residual(
            templates.source_open,
            baseline=training_baseline,
            seam_bins=source_support,
            cells=cells,
            rate=rate,
            occupancy=occupancy,
            rate_function=rate_function,
        )
        target = (
            rate_function(
                rate[target_test],
                occupancy[target_test],
                cells,
                target_support,
            )
            - rate_function(
                rate[test_baseline],
                occupancy[test_baseline],
                cells,
                target_support,
            )
        )
        source_rank = identity._rank_unit(source_wall - source_open)
        target_rank = identity._rank_unit(target)
        reproduced = float(source_rank @ target_rank)
        expected = record["metrics"][mode][
            "source_effect_r_to_target_residual"
        ]
        if not np.isclose(reproduced, expected, atol=1e-12, rtol=0):
            raise AssertionError(
                "rank vectors do not reproduce source-record correlation"
            )
        output[mode] = (source_rank, target_rank)
    return record["orientation_relation"], output


def _prepare_query(
    *,
    training_start: int,
    test_start: int,
    target_offset: int,
    target_seam: OrientedSeam,
    blocked: list[tuple[int, ...]],
    rate: dict[int, np.ndarray],
    occupancy: dict[int, np.ndarray],
    registered: dict[int, np.ndarray],
    strips: dict[OrientedSeam, tuple[tuple[int, int], ...]],
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
) -> PreparedQuery | None:
    cells = transfer._query_common_cells(
        training_start=training_start,
        test_start=test_start,
        target_offset=target_offset,
        registered=registered,
    )
    if cells.size < minimum_cells:
        return None

    by_relation: dict[str, dict[str, list[np.ndarray]]] = {
        relation: {
            mode: [] for mode in transfer.RATE_MODES
        }
        for relation in ("same_signed_normal", "opposite_normal")
    }
    target_by_mode: dict[str, np.ndarray] = {}
    for source_seam in internal_seams():
        if source_seam.unordered == target_seam.unordered:
            continue
        if not np.isclose(
            transfer._midpoint_distance_cm(target_seam, source_seam),
            25.0,
        ):
            continue
        relation = transfer._orientation_relation(
            target_seam,
            source_seam,
        )
        if relation not in by_relation:
            continue
        result = _source_and_target_vectors(
            training_start=training_start,
            test_start=test_start,
            target_offset=target_offset,
            target_seam=target_seam,
            source_seam=source_seam,
            blocked=blocked,
            rate=rate,
            occupancy=occupancy,
            registered=registered,
            query_cells=cells,
            strips=strips,
            minimum_seconds=minimum_seconds,
            minimum_bins=minimum_bins,
            minimum_cells=minimum_cells,
        )
        if result is None:
            continue
        result_relation, vectors = result
        if result_relation != relation:
            raise AssertionError("orientation relation changed")
        for mode, (source_rank, target_rank) in vectors.items():
            by_relation[relation][mode].append(source_rank)
            if mode in target_by_mode and not np.array_equal(
                target_by_mode[mode],
                target_rank,
            ):
                raise AssertionError(
                    "one target query produced multiple target vectors"
                )
            target_by_mode[mode] = target_rank

    if not by_relation["same_signed_normal"][
        next(iter(transfer.RATE_MODES))
    ]:
        return None
    same = {
        mode: np.stack(by_relation["same_signed_normal"][mode])
        for mode in transfer.RATE_MODES
    }
    opposite = {
        mode: (
            np.stack(by_relation["opposite_normal"][mode])
            if by_relation["opposite_normal"][mode]
            else np.empty((0, cells.size), dtype=np.float64)
        )
        for mode in transfer.RATE_MODES
    }
    return PreparedQuery(
        cell_ids=cells.astype(np.int64, copy=False),
        same_source=same,
        opposite_source=opposite,
        target=target_by_mode,
    )


def _score_queries(
    queries: list[PreparedQuery],
    *,
    mode: str,
    positions: list[np.ndarray] | None = None,
) -> tuple[float, float | None]:
    primary = []
    direction = []
    if positions is None:
        positions = [
            np.arange(query.cell_ids.size, dtype=np.int64)
            for query in queries
        ]
    for query, order in zip(queries, positions, strict=True):
        target = query.target[mode][order]
        same = query.same_source[mode] @ target
        primary.append(float(np.mean(same)))
        if query.opposite_source[mode].shape[0]:
            opposite = query.opposite_source[mode] @ target
            direction.append(
                float(np.mean(same) - np.mean(opposite))
            )
    return (
        float(np.mean(primary)),
        float(np.mean(direction)) if direction else None,
    )


def _permutation_diagnostic(
    queries: list[PreparedQuery],
    *,
    n_cells: int,
    draws: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if draws < 1:
        raise ValueError("draws must be positive")
    strata = identity._membership_strata(
        [query.cell_ids for query in queries],
        n_cells=n_cells,
    )
    observed = {
        mode: _score_queries(queries, mode=mode)
        for mode in transfer.RATE_MODES
    }
    null = {
        mode: {
            "primary": np.empty(draws),
            "direction": np.empty(draws),
        }
        for mode in transfer.RATE_MODES
    }
    used_ids = np.sort(np.concatenate(strata))
    fixed_fraction = np.empty(draws)
    changed_queries = np.empty(draws, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for draw in range(draws):
        permutation = identity._draw_global_permutation(
            strata,
            n_cells=n_cells,
            rng=rng,
        )
        fixed_fraction[draw] = float(
            np.mean(permutation[used_ids] == used_ids)
        )
        positions = [
            identity._permuted_positions(
                query.cell_ids,
                permutation,
            )
            for query in queries
        ]
        changed_queries[draw] = int(
            sum(
                np.any(
                    order
                    != np.arange(order.size, dtype=np.int64)
                )
                for order in positions
            )
        )
        for mode in transfer.RATE_MODES:
            primary, direction = _score_queries(
                queries,
                mode=mode,
                positions=positions,
            )
            null[mode]["primary"][draw] = primary
            if direction is None:
                raise AssertionError("direction contrast disappeared")
            null[mode]["direction"][draw] = direction

    result = {}
    for mode in transfer.RATE_MODES:
        primary_observed, direction_observed = observed[mode]
        if direction_observed is None:
            raise AssertionError("direction contrast is not estimable")
        mode_result = {}
        for label, value in (
            ("primary_same_normal_transfer", primary_observed),
            ("same_minus_opposite_direction", direction_observed),
        ):
            null_key = (
                "primary"
                if label == "primary_same_normal_transfer"
                else "direction"
            )
            values = null[mode][null_key]
            mode_result[label] = {
                "observed": value,
                "draws": draws,
                "null_mean": float(np.mean(values)),
                "null_standard_deviation": float(
                    np.std(values, ddof=1)
                ),
                "null_quantiles_0_025_0_5_0_975": [
                    float(item)
                    for item in np.quantile(
                        values,
                        [0.025, 0.5, 0.975],
                    )
                ],
                "one_sided_plus_one_tail_fraction": float(
                    (1 + np.count_nonzero(values >= value))
                    / (draws + 1)
                ),
                "observed_exceeds_all_draws": bool(
                    value > np.max(values)
                ),
            }
        result[mode] = mode_result

    sizes = np.asarray([value.size for value in strata])
    used_cells = int(np.sum(sizes))
    exchangeable_cells = int(np.sum(sizes[sizes > 1]))
    diagnostics = {
        "animal_global_cells": n_cells,
        "global_cells_used_by_queries": used_cells,
        "membership_strata": len(strata),
        "nontrivial_membership_strata": int(
            np.count_nonzero(sizes > 1)
        ),
        "cells_in_nontrivial_strata": exchangeable_cells,
        "fraction_used_cells_in_nontrivial_strata": (
            exchangeable_cells / used_cells
        ),
        "largest_membership_stratum": int(np.max(sizes)),
        "fixed_point_fraction_among_used_ids_min_mean_max": [
            float(np.min(fixed_fraction)),
            float(np.mean(fixed_fraction)),
            float(np.max(fixed_fraction)),
        ],
        "queries_with_identity_pairing_changed_min_mean_max": [
            int(np.min(changed_queries)),
            float(np.mean(changed_queries)),
            int(np.max(changed_queries)),
        ],
        "all_query_common_cell_sets_preserved_every_draw": True,
        "same_global_mapping_used_for_all_queries_and_modes_per_draw": True,
    }
    return result, diagnostics


def analyze_animal(
    path: Path,
    *,
    minimum_seconds: float,
    minimum_bins: int,
    minimum_cells: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    seams = internal_seams()
    strips = {seam: seam_strip_bins(seam) for seam in seams}
    with Mat73Animal(path) as animal:
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
        n_cells = animal.n_cells

    queries = []
    for training_exposure in range(repetitions - 1):
        training_start = training_exposure * 10
        test_start = (training_exposure + 1) * 10
        for target_offset in range(1, 10):
            target_test = test_start + target_offset
            for target_seam in seams:
                if (
                    seam_state(blocked[target_test], target_seam)
                    is not SeamState.WALL
                ):
                    continue
                query = _prepare_query(
                    training_start=training_start,
                    test_start=test_start,
                    target_offset=target_offset,
                    target_seam=target_seam,
                    blocked=blocked,
                    rate=rate,
                    occupancy=occupancy,
                    registered=registered,
                    strips=strips,
                    minimum_seconds=minimum_seconds,
                    minimum_bins=minimum_bins,
                    minimum_cells=minimum_cells,
                )
                if query is not None:
                    queries.append(query)
    if not queries:
        raise ValueError(f"{path.name} has no eligible exact-distance queries")
    mode_result, exchangeability = _permutation_diagnostic(
        queries,
        n_cells=n_cells,
        draws=permutations,
        seed=seed,
    )
    return {
        "animal": path.name.removesuffix(".complete.mat"),
        "prepared_target_queries": len(queries),
        "direction_contrast_queries": int(
            sum(
                query.opposite_source[
                    next(iter(transfer.RATE_MODES))
                ].shape[0]
                > 0
                for query in queries
            )
        ),
        "common_cells_min_median_max": [
            int(np.min([query.cell_ids.size for query in queries])),
            float(np.median([query.cell_ids.size for query in queries])),
            int(np.max([query.cell_ids.size for query in queries])),
        ],
        "exchangeability": exchangeability,
        "modes": mode_result,
    }


def _cohort_summary(
    animals: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for mode in transfer.RATE_MODES:
        mode_result = {}
        for label in (
            "primary_same_normal_transfer",
            "same_minus_opposite_direction",
        ):
            values = np.asarray(
                [
                    animal["modes"][mode][label]["observed"]
                    for animal in animals
                ]
            )
            tails = np.asarray(
                [
                    animal["modes"][mode][label][
                        "one_sided_plus_one_tail_fraction"
                    ]
                    for animal in animals
                ]
            )
            mode_result[label] = {
                "animals": len(animals),
                "positive_observed_animals": int(
                    np.count_nonzero(values > 0)
                ),
                "animal_mean_observed": float(np.mean(values)),
                "animals_observed_exceeds_all_draws": int(
                    sum(
                        animal["modes"][mode][label][
                            "observed_exceeds_all_draws"
                        ]
                        for animal in animals
                    )
                ),
                "animal_tail_fractions": {
                    animal["animal"]: animal["modes"][mode][label][
                        "one_sided_plus_one_tail_fraction"
                    ]
                    for animal in animals
                },
                "maximum_animal_tail_fraction": float(np.max(tails)),
            }
        result[mode] = mode_result
    return result


def _validate_observed_reproduction(
    animal: dict[str, Any],
    validation_animal: dict[str, Any],
) -> dict[str, float]:
    output = {}
    for mode in transfer.RATE_MODES:
        expected_primary = validation_animal["modes"][mode][
            "one_grid_step_same_direction_source_effect_r_to_"
            "target_residual"
        ]["mean"]
        expected_direction = validation_animal["modes"][mode][
            "one_grid_step_same_direction_minus_exact_distance_"
            "nonsame_source_effect_r"
        ]["mean"]
        observed = animal["modes"][mode]
        output[mode] = float(
            max(
                abs(
                    observed["primary_same_normal_transfer"]["observed"]
                    - expected_primary
                ),
                abs(
                    observed["same_minus_opposite_direction"]["observed"]
                    - expected_direction
                ),
            )
        )
    if max(output.values()) > 1e-12:
        raise AssertionError(
            "permutation records do not reproduce validation summary"
        )
    return output


def main() -> None:
    argument = parse_arguments()
    paths = sorted(
        argument.data_dir.glob("QLAK-CA1-*.complete.mat")
    )
    if not paths:
        raise FileNotFoundError(
            f"no complete animal files found beneath {argument.data_dir}"
        )
    validation = json.loads(
        argument.validation.read_text(encoding="utf-8")
    )
    validation_by_animal = {
        animal["animal"]: animal
        for animal in validation["animals"]
    }

    animals = []
    for index, path in enumerate(paths):
        animal = analyze_animal(
            path,
            minimum_seconds=argument.minimum_seconds,
            minimum_bins=argument.minimum_bins,
            minimum_cells=argument.minimum_cells,
            permutations=argument.permutations,
            seed=argument.seed + index,
        )
        animal["validation_reproduction_max_absolute_error"] = (
            _validate_observed_reproduction(
                animal,
                validation_by_animal[animal["animal"]],
            )
        )
        animals.append(animal)
        print(
            animal["animal"],
            {
                mode: {
                    label: value[
                        "one_sided_plus_one_tail_fraction"
                    ]
                    for label, value in result.items()
                }
                for mode, result in animal["modes"].items()
            },
        )

    report = {
        "status": "coherent_cross_location_cell_identity_diagnostic",
        "question": (
            "Does exact-25-cm cross-location transfer depend on coherent "
            "registered-cell identity alignment?"
        ),
        "design": {
            "one_global_id_mapping_per_animal_draw": True,
            "same_mapping_all_target_queries_and_rate_modes": True,
            "training_source_profiles_fixed": True,
            "target_residual_identities_permuted": True,
            "query_common_cell_sets_preserved": True,
            "source_and_target_neural_sessions_disjoint": True,
            "aggregation_reproduced_each_draw": (
                "source correlations within query, queries within animal"
            ),
            "inferential_scope": (
                "restricted cell-identity diagnostic, not population "
                "inference"
            ),
        },
        "settings": {
            "minimum_seconds_per_common_bin": argument.minimum_seconds,
            "minimum_common_bins": argument.minimum_bins,
            "minimum_common_cells": argument.minimum_cells,
            "permutations": argument.permutations,
            "seed": argument.seed,
        },
        "cohort_descriptive": _cohort_summary(animals),
        "animals": animals,
    }
    argument.output.parent.mkdir(parents=True, exist_ok=True)
    argument.output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["cohort_descriptive"], indent=2))


if __name__ == "__main__":
    main()
