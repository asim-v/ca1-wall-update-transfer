"""Frozen positive-model and empirical-rank utilities.

The functions in this module contain no dataset-specific I/O.  They implement
the query-level ranking, hierarchical animal aggregation, and exact animal
sign-flip inference declared in POSITIVE_MODEL_ADJUDICATION_PROTOCOL.md.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PRIMARY_SCORE = "source_effect_r_to_target_residual"


def midrank_percentile(value: float, candidates: Sequence[float]) -> float:
    """Return the empirical mid-CDF of ``value`` among ``candidates``.

    The result is zero when every candidate is greater, one when every
    candidate is smaller, and 0.5 for a complete tie.  Candidate values must be
    finite; adding ``value`` as an extra candidate would distort the requested
    rank and is deliberately avoided.
    """

    array = np.asarray(candidates, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError("at least one candidate is required")
    if not np.isfinite(value) or not np.isfinite(array).all():
        raise ValueError("rank values must be finite")
    below = np.count_nonzero(array < value)
    tied = np.count_nonzero(array == value)
    return float((below + 0.5 * tied) / array.size)


def query_empirical_rank(
    records: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    tier: str = "tier1_exact_25cm",
    score_key: str = PRIMARY_SCORE,
) -> dict[str, Any] | None:
    """Score the correct relation against every admissible source.

    Records are assumed to have passed the common-cell, common-bin, and neural
    session-separation gates upstream.  This function applies only the frozen
    geometric tier and never selects a source from its neural score.
    """

    if tier not in {"tier1_exact_25cm", "tier2_tangential"}:
        raise ValueError(f"unknown matching tier: {tier}")
    eligible = []
    for record in records:
        distance = float(record["midpoint_distance_cm"])
        if not np.isclose(distance, 25.0, rtol=0.0, atol=1e-10):
            continue
        if tier == "tier2_tangential" and (
            record.get("translation_axis") != "tangential"
        ):
            continue
        score = float(record["metrics"][mode][score_key])
        if not np.isfinite(score):
            continue
        eligible.append((record, score))

    correct = [
        score
        for record, score in eligible
        if record["orientation_relation"] == "same_signed_normal"
    ]
    alternatives = [
        score
        for record, score in eligible
        if record["orientation_relation"] != "same_signed_normal"
    ]
    if not correct or not alternatives:
        return None

    correct_mean = float(np.mean(correct))
    alternative_mean = float(np.mean(alternatives))
    best_alternative = float(np.max(alternatives))
    candidate_scores = [score for _record, score in eligible]
    relations: dict[str, int] = {}
    for record, _score in eligible:
        relation = str(record["orientation_relation"])
        relations[relation] = relations.get(relation, 0) + 1
    correct_wins = bool(correct_mean > best_alternative)
    return {
        "tier": tier,
        "candidate_sources": len(eligible),
        "correct_sources": len(correct),
        "alternative_sources": len(alternatives),
        "orientation_counts": relations,
        "correct_relation_mean": correct_mean,
        "alternative_mean": alternative_mean,
        "best_alternative": best_alternative,
        "correct_percentile_rank": midrank_percentile(
            correct_mean,
            candidate_scores,
        ),
        "centered_percentile_rank": midrank_percentile(
            correct_mean,
            candidate_scores,
        )
        - 0.5,
        "correct_minus_mean_alternative": (
            correct_mean - alternative_mean
        ),
        "correct_minus_best_alternative": (
            correct_mean - best_alternative
        ),
        "correct_beats_every_alternative": correct_wins,
        "centered_correct_win": float(correct_wins) - 0.5,
    }


def hierarchical_animal_summary(
    queries: Sequence[Mapping[str, Any]],
    *,
    metric_names: Iterable[str],
) -> dict[str, Any]:
    """Aggregate queries within exposure pair and pairs within animal."""

    metrics = tuple(metric_names)
    if not metrics:
        raise ValueError("at least one metric is required")
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for query in queries:
        key = (
            int(query["training_exposure"]),
            int(query["test_exposure"]),
        )
        grouped.setdefault(key, []).append(query)
    if not grouped:
        raise ValueError("at least one query is required")

    pair_values: dict[str, dict[str, float]] = {name: {} for name in metrics}
    for pair, values in sorted(grouped.items()):
        label = f"{pair[0]}->{pair[1]}"
        for name in metrics:
            finite = np.asarray(
                [float(value[name]) for value in values],
                dtype=np.float64,
            )
            finite = finite[np.isfinite(finite)]
            if finite.size:
                pair_values[name][label] = float(np.mean(finite))

    output: dict[str, Any] = {
        "eligible_queries": len(queries),
        "eligible_exposure_pairs": len(grouped),
        "metrics": {},
    }
    for name in metrics:
        values = np.asarray(
            list(pair_values[name].values()),
            dtype=np.float64,
        )
        if values.size == 0:
            raise ValueError(f"metric has no finite values: {name}")
        output["metrics"][name] = {
            "animal_value": float(np.mean(values)),
            "exposure_pair_values": pair_values[name],
        }
    return output


def exact_sign_flip(values: Mapping[str, float]) -> dict[str, Any]:
    """Enumerate exact one- and two-sided animal sign-flip distributions."""

    if not values:
        raise ValueError("at least one animal value is required")
    animals = tuple(sorted(values))
    observed_values = np.asarray(
        [float(values[animal]) for animal in animals],
        dtype=np.float64,
    )
    if not np.isfinite(observed_values).all():
        raise ValueError("animal values must be finite")
    observed = float(np.mean(observed_values))
    null = np.asarray(
        [
            np.mean(observed_values * np.asarray(signs, dtype=np.float64))
            for signs in product((-1.0, 1.0), repeat=len(animals))
        ],
        dtype=np.float64,
    )
    tolerance = 1e-15
    leave_one_out = {
        animal: float(np.mean(np.delete(observed_values, index)))
        for index, animal in enumerate(animals)
    }
    return {
        "animals": len(animals),
        "animal_values": {
            animal: float(values[animal]) for animal in animals
        },
        "positive_animals": int(np.count_nonzero(observed_values > 0)),
        "observed_animal_mean": observed,
        "observed_animal_median": float(np.median(observed_values)),
        "sign_assignments": int(null.size),
        "one_sided_tail_fraction": float(
            np.count_nonzero(null >= observed - tolerance) / null.size
        ),
        "two_sided_tail_fraction": float(
            np.count_nonzero(
                np.abs(null) >= abs(observed) - tolerance
            )
            / null.size
        ),
        "null_means": null.tolist(),
        "leave_one_animal_out_means": leave_one_out,
    }


def cohort_metric_summary(
    animals: Sequence[Mapping[str, Any]],
    *,
    metric_name: str,
) -> dict[str, Any]:
    """Extract one hierarchical animal metric and run exact inference."""

    values = {
        str(animal["animal"]): float(
            animal["summary"]["metrics"][metric_name]["animal_value"]
        )
        for animal in animals
    }
    return exact_sign_flip(values)
