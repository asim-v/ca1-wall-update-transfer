"""Session-consistent cell-identity permutation for local wall prediction.

This is a descriptive diagnostic for the target-rate-held-out analysis in
``run_boundary_component_validation.py``.  Within one animal and one draw, a
single permutation of global cell IDs is used for every eligible prediction
record and both rate modes.

The permutation is constrained to preserve every record's common registered
cell set.  Specifically, cells are exchangeable only when they have the same
inclusion pattern across all eligible record-specific common-cell sets.  This
produces the largest strata within which a global permutation maps every
record's cell set onto itself.

For each template cell ID ``i``, the target residual belonging to global cell
ID ``permutation[i]`` is paired with the fixed wall/open template at ``i``.
The same mapping is therefore applied to both target-side sessions before
their subtraction, to every target record, and to both rate modes.  Training
templates are not permuted.  This breaks cross-exposure cell-identity
alignment while retaining vector values, record supports, registration
subsets, and dependencies among records.  It is not population inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_boundary_component_validation as validation  # noqa: E402
from ca1_geometry.io import Mat73Animal  # noqa: E402
from ca1_geometry.seams import (  # noqa: E402
    OrientedSeam,
    SeamState,
    internal_seams,
    seam_state,
    seam_strip_bins,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class PreparedRecord(NamedTuple):
    """Rank-normalized score terms tied to their global cell IDs."""

    cell_ids: IntArray
    by_mode: dict[str, tuple[FloatArray, FloatArray]]


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
            / "boundary_fragment_session_permutation.json"
        ),
    )
    parser.add_argument("--minimum-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-bins", type=int, default=6)
    parser.add_argument("--minimum-cells", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def _record_cell_ids(
    *,
    target_offset: int,
    training_exposure: int,
    blocked: list[tuple[int, ...]],
    registered: dict[int, NDArray[np.bool_]],
    seam: OrientedSeam,
) -> IntArray:
    """Reconstruct the exact common-cell subset used by validation._record."""

    train_start = training_exposure * 10
    test_start = (training_exposure + 1) * 10
    test_baseline = (training_exposure + 2) * 10
    target_test = test_start + target_offset
    matching_wall = [
        train_start + offset
        for offset in range(1, 10)
        if offset != target_offset
        and seam_state(blocked[train_start + offset], seam)
        is SeamState.WALL
    ]
    matching_open = [
        train_start + offset
        for offset in range(1, 10)
        if offset != target_offset
        and seam_state(blocked[train_start + offset], seam)
        is SeamState.OPEN
    ]
    required = [
        train_start,
        test_baseline,
        target_test,
        *matching_wall,
        *matching_open,
    ]
    return np.flatnonzero(
        np.logical_and.reduce([registered[item] for item in required])
    ).astype(np.int64, copy=False)


def _rank_unit(value: FloatArray) -> FloatArray:
    """Return centered unit-length ranks for a complete finite vector."""

    array = np.asarray(value, dtype=np.float64).ravel()
    if array.size < 3 or not np.isfinite(array).all():
        raise ValueError(
            "permutation records require at least three finite cell values"
        )
    ranks = rankdata(array)
    centered = ranks - np.mean(ranks)
    norm = float(np.linalg.norm(centered))
    if norm <= np.finfo(float).eps:
        raise ValueError("permutation record has constant ranks")
    return centered / norm


def _prepare_record(
    cell_ids: IntArray,
    vectors: dict[str, tuple[FloatArray, FloatArray, FloatArray]],
) -> PreparedRecord:
    ids = np.asarray(cell_ids, dtype=np.int64).ravel()
    if ids.size < 3 or np.any(np.diff(ids) <= 0):
        raise ValueError("cell_ids must be unique and strictly increasing")
    by_mode: dict[str, tuple[FloatArray, FloatArray]] = {}
    for mode in validation.RATE_MODES:
        wall, open_value, target = vectors[mode]
        if not (wall.shape == open_value.shape == target.shape == ids.shape):
            raise ValueError("cell IDs and prediction vectors must align")
        template_delta = _rank_unit(wall) - _rank_unit(open_value)
        by_mode[mode] = (template_delta, _rank_unit(target))
    return PreparedRecord(cell_ids=ids, by_mode=by_mode)


def _membership_strata(
    record_cell_ids: list[IntArray],
    *,
    n_cells: int,
) -> tuple[IntArray, ...]:
    """Partition used global IDs by record-inclusion signature.

    Any permutation within one returned stratum leaves every supplied record
    subset invariant.  Conversely, cells with different signatures cannot be
    exchanged without changing at least one record's registered-cell subset.
    """

    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    if not record_cell_ids:
        raise ValueError("at least one record cell set is required")
    membership = np.zeros(
        (n_cells, len(record_cell_ids)),
        dtype=np.bool_,
    )
    for record_index, raw_ids in enumerate(record_cell_ids):
        ids = np.asarray(raw_ids, dtype=np.int64).ravel()
        if (
            ids.size < 1
            or np.any(ids < 0)
            or np.any(ids >= n_cells)
            or np.unique(ids).size != ids.size
        ):
            raise ValueError("record cell IDs must be unique valid indices")
        membership[ids, record_index] = True
    used = np.flatnonzero(np.any(membership, axis=1))
    packed = np.packbits(membership[used], axis=1)
    groups: dict[bytes, list[int]] = {}
    for cell_id, signature in zip(used, packed, strict=True):
        groups.setdefault(signature.tobytes(), []).append(int(cell_id))
    strata = tuple(
        np.asarray(group, dtype=np.int64)
        for group in sorted(groups.values(), key=lambda value: value[0])
    )
    if sum(stratum.size for stratum in strata) != used.size:
        raise AssertionError("membership partition lost global cell IDs")
    return strata


def _draw_global_permutation(
    strata: tuple[IntArray, ...],
    *,
    n_cells: int,
    rng: np.random.Generator,
) -> IntArray:
    """Draw one global bijection constrained by membership strata."""

    permutation = np.arange(n_cells, dtype=np.int64)
    for stratum in strata:
        if stratum.size > 1:
            permutation[stratum] = rng.permutation(stratum)
    if np.unique(permutation).size != n_cells:
        raise AssertionError("cell-identity mapping is not a bijection")
    return permutation


def _permuted_positions(
    cell_ids: IntArray,
    permutation: IntArray,
) -> IntArray:
    """Locate globally permuted target identities inside one record vector."""

    permuted_ids = permutation[cell_ids]
    positions = np.searchsorted(cell_ids, permuted_ids)
    if (
        np.any(positions >= cell_ids.size)
        or not np.array_equal(cell_ids[positions], permuted_ids)
    ):
        raise AssertionError(
            "global permutation did not preserve a record common-cell subset"
        )
    return positions.astype(np.int64, copy=False)


def _observed_scores(
    records: list[PreparedRecord],
) -> dict[str, float]:
    return {
        mode: float(
            np.mean(
                [
                    template_delta @ target
                    for record in records
                    for template_delta, target in [record.by_mode[mode]]
                ]
            )
        )
        for mode in validation.RATE_MODES
    }


def _permutation_diagnostic(
    records: list[PreparedRecord],
    *,
    n_cells: int,
    draws: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score shared global-ID permutations for both rate modes."""

    if draws < 1:
        raise ValueError("draws must be positive")
    strata = _membership_strata(
        [record.cell_ids for record in records],
        n_cells=n_cells,
    )
    observed = _observed_scores(records)
    null = {
        mode: np.empty(draws, dtype=np.float64)
        for mode in validation.RATE_MODES
    }
    used_ids = np.sort(np.concatenate(strata))
    fixed_fraction = np.empty(draws, dtype=np.float64)
    changed_identity_records = np.empty(draws, dtype=np.int64)
    changed_target_records = {
        mode: np.empty(draws, dtype=np.int64)
        for mode in validation.RATE_MODES
    }
    rng = np.random.default_rng(seed)
    for draw in range(draws):
        permutation = _draw_global_permutation(
            strata,
            n_cells=n_cells,
            rng=rng,
        )
        fixed_fraction[draw] = float(
            np.mean(permutation[used_ids] == used_ids)
        )
        changed_identity_count = 0
        changed_target_count = {
            mode: 0 for mode in validation.RATE_MODES
        }
        score_sum = {mode: 0.0 for mode in validation.RATE_MODES}
        for record in records:
            positions = _permuted_positions(
                record.cell_ids,
                permutation,
            )
            if np.any(
                positions
                != np.arange(record.cell_ids.size, dtype=np.int64)
            ):
                changed_identity_count += 1
            for mode in validation.RATE_MODES:
                template_delta, target = record.by_mode[mode]
                if not np.array_equal(target[positions], target):
                    changed_target_count[mode] += 1
                score_sum[mode] += float(
                    template_delta @ target[positions]
                )
        changed_identity_records[draw] = changed_identity_count
        for mode in validation.RATE_MODES:
            null[mode][draw] = score_sum[mode] / len(records)
            changed_target_records[mode][draw] = (
                changed_target_count[mode]
            )

    result = {}
    for mode in validation.RATE_MODES:
        values = null[mode]
        result[mode] = {
            "observed_animal_mean_wall_minus_open": observed[mode],
            "draws": draws,
            "null_mean": float(np.mean(values)),
            "null_standard_deviation": float(np.std(values, ddof=1)),
            "null_quantiles_0_025_0_5_0_975": [
                float(value)
                for value in np.quantile(values, [0.025, 0.5, 0.975])
            ],
            "one_sided_plus_one_tail_fraction": float(
                (1 + np.count_nonzero(values >= observed[mode]))
                / (draws + 1)
            ),
            "observed_exceeds_all_draws": bool(
                observed[mode] > np.max(values)
            ),
            "records_with_target_rank_vector_changed_per_draw_min_mean_max": [
                int(np.min(changed_target_records[mode])),
                float(np.mean(changed_target_records[mode])),
                int(np.max(changed_target_records[mode])),
            ],
        }

    used_cells = int(
        sum(stratum.size for stratum in strata)
    )
    exchangeable = int(
        sum(stratum.size for stratum in strata if stratum.size > 1)
    )
    stratum_sizes = np.asarray(
        [stratum.size for stratum in strata],
        dtype=np.int64,
    )
    diagnostics = {
        "animal_global_cells": n_cells,
        "global_cells_used_by_eligible_records": used_cells,
        "membership_strata": len(strata),
        "nontrivial_membership_strata": int(
            np.count_nonzero(stratum_sizes > 1)
        ),
        "singleton_membership_strata": int(
            np.count_nonzero(stratum_sizes == 1)
        ),
        "cells_in_nontrivial_strata": exchangeable,
        "fraction_used_cells_in_nontrivial_strata": (
            exchangeable / used_cells
        ),
        "largest_membership_stratum": int(np.max(stratum_sizes)),
        "fixed_point_fraction_among_used_ids_min_mean_max": [
            float(np.min(fixed_fraction)),
            float(np.mean(fixed_fraction)),
            float(np.max(fixed_fraction)),
        ],
        "records_with_identity_pairing_changed_per_draw_min_mean_max": [
            int(np.min(changed_identity_records)),
            float(np.mean(changed_identity_records)),
            int(np.max(changed_identity_records)),
        ],
        "derangements_enforced": False,
        "exchangeability_limitation": (
            f"{used_cells - exchangeable} of {used_cells} used global IDs "
            "are singleton-stratum IDs and cannot move; ordinary random "
            "permutations also permit fixed points within non-singleton "
            "strata. This is therefore a restricted identity diagnostic, "
            "with its effective strength quantified by the fixed-point and "
            "changed-record metrics above."
        ),
        "all_record_common_cell_sets_preserved_every_draw": True,
        "same_global_mapping_used_for_all_records_and_modes_per_draw": True,
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
        n_cells = animal.n_cells

    prepared: list[PreparedRecord] = []
    base_scores: dict[str, list[float]] = {
        mode: [] for mode in validation.RATE_MODES
    }
    record_cell_counts: list[int] = []
    for training_exposure in range(repetitions - 1):
        for target_offset in range(1, 10):
            target_test = (training_exposure + 1) * 10 + target_offset
            for seam in seams:
                if (
                    seam_state(blocked[target_test], seam)
                    is not SeamState.WALL
                ):
                    continue
                result = validation._record(
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
                )
                if result is None:
                    continue
                record, vectors = result
                cell_ids = _record_cell_ids(
                    target_offset=target_offset,
                    training_exposure=training_exposure,
                    blocked=blocked,
                    registered=registered,
                    seam=seam,
                )
                if cell_ids.size != record["cells"]:
                    raise AssertionError(
                        "global cell IDs disagree with validation record"
                    )
                prepared_record = _prepare_record(cell_ids, vectors)
                prepared.append(prepared_record)
                record_cell_counts.append(int(cell_ids.size))
                for mode in validation.RATE_MODES:
                    base_scores[mode].append(
                        record["correlations"][mode]["wall_minus_open"]
                    )

    if not prepared:
        raise ValueError(f"{path.name} has no eligible prediction records")
    observed = _observed_scores(prepared)
    reproduction_error = {
        mode: float(
            abs(observed[mode] - np.mean(base_scores[mode]))
        )
        for mode in validation.RATE_MODES
    }
    if max(reproduction_error.values()) > 1e-12:
        raise AssertionError(
            "rank-vector scores do not reproduce validation correlations"
        )

    diagnostic, exchangeability = _permutation_diagnostic(
        prepared,
        n_cells=n_cells,
        draws=permutations,
        seed=seed,
    )
    return {
        "animal": path.name.removesuffix(".complete.mat"),
        "eligible_local_predictions": len(prepared),
        "record_common_cells_min_median_max": [
            int(np.min(record_cell_counts)),
            float(np.median(record_cell_counts)),
            int(np.max(record_cell_counts)),
        ],
        "validation_score_reproduction_max_absolute_error": float(
            max(reproduction_error.values())
        ),
        "exchangeability": exchangeability,
        "modes": diagnostic,
    }


def _cohort_summary(animals: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for mode in validation.RATE_MODES:
        observed = np.asarray(
            [
                animal["modes"][mode][
                    "observed_animal_mean_wall_minus_open"
                ]
                for animal in animals
            ],
            dtype=np.float64,
        )
        tails = np.asarray(
            [
                animal["modes"][mode][
                    "one_sided_plus_one_tail_fraction"
                ]
                for animal in animals
            ],
            dtype=np.float64,
        )
        result[mode] = {
            "animals": int(observed.size),
            "positive_observed_animals": int(
                np.count_nonzero(observed > 0)
            ),
            "animal_mean_observed_wall_minus_open": float(
                np.mean(observed)
            ),
            "animals_observed_exceeds_all_draws": int(
                sum(
                    animal["modes"][mode][
                        "observed_exceeds_all_draws"
                    ]
                    for animal in animals
                )
            ),
            "descriptive_animal_tail_fractions": {
                animal["animal"]: animal["modes"][mode][
                    "one_sided_plus_one_tail_fraction"
                ]
                for animal in animals
            },
            "maximum_descriptive_animal_tail_fraction": float(
                np.max(tails)
            ),
            "population_inference_performed": False,
        }
    return result


def main() -> None:
    argument = parse_arguments()
    if argument.permutations < 999:
        raise ValueError(
            "session-consistent diagnostic requires at least 999 permutations"
        )
    paths = sorted(
        argument.data_dir.glob("QLAK-CA1-*.complete.mat")
    )
    if not paths:
        raise FileNotFoundError(
            f"no complete animal files found beneath {argument.data_dir}"
        )

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
        animals.append(animal)
        print(
            animal["animal"],
            {
                mode: {
                    "observed": round(
                        result[
                            "observed_animal_mean_wall_minus_open"
                        ],
                        4,
                    ),
                    "tail": result[
                        "one_sided_plus_one_tail_fraction"
                    ],
                }
                for mode, result in animal["modes"].items()
            },
        )

    report = {
        "status": (
            "exploratory_session_consistent_cell_identity_permutation"
        ),
        "question": (
            "Does target-rate-held-out local wall prediction depend on "
            "registered-cell identity alignment across exposure cycles?"
        ),
        "design": {
            "unit_of_permutation": (
                "animal-level global cell ID mapping, shared by every "
                "eligible record and both rate modes within a draw"
            ),
            "permuted_vector": (
                "the target-side square-residual local cell-rate vector; "
                "target test and target square-baseline identities receive "
                "the same mapping before subtraction"
            ),
            "fixed_vectors": (
                "matching-wall and matching-open training templates"
            ),
            "cell_pairing": (
                "template at global ID i is paired with target residual at "
                "global ID permutation[i]"
            ),
            "exchangeability_constraint": (
                "global IDs are permuted only within strata having identical "
                "membership across all eligible record-specific common-cell "
                "sets, so one coherent mapping preserves every subset"
            ),
            "destroyed_structure": (
                "cross-exposure correspondence of registered-cell identities"
            ),
            "preserved_structure": [
                "one global mapping across all dependent records/sessions",
                "same mapping for raw and global-rate-demeaned modes",
                "record-specific common registered-cell subsets",
                "target vector values and ranks",
                "wall/open template vectors",
                "spatial support and record count",
                "dependencies caused by reused sessions and cells",
            ],
            "interpretation": (
                "animal-level descriptive permutation diagnostic; the tail "
                "fraction is not a population-level p-value"
            ),
            "exchangeability_caveat": (
                "Preserving every overlapping common-cell subset requires "
                "singleton membership strata for some IDs, and ordinary "
                "within-stratum permutations can have additional fixed "
                "points. The resulting null can be weak when few IDs are "
                "exchangeable; per-animal exchangeable fractions, fixed-"
                "point fractions, and changed-record counts are reported."
            ),
            "population_inference_performed": False,
        },
        "settings": {
            "minimum_seconds_per_common_bin": argument.minimum_seconds,
            "minimum_common_bins": argument.minimum_bins,
            "minimum_common_cells": argument.minimum_cells,
            "permutations_per_animal": argument.permutations,
            "seed": argument.seed,
            "animal_seed_rule": "seed plus zero-based sorted animal index",
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
