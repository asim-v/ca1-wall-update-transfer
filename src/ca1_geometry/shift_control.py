"""Utilities for diagnostic population trace-position shift controls."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


class PlusOnePValues(TypedDict):
    """Monte Carlo tail areas with the observed statistic added once."""

    n_surrogate_requested: int
    n_surrogate_finite: int
    greater_or_equal: float | None
    two_sided_absolute: float | None


def unique_sequence_sessions(
    sequences: Iterable[Sequence[int]],
) -> tuple[int, ...]:
    """Return sorted session IDs, representing shared sessions only once."""

    session = {
        int(value)
        for sequence in sequences
        for value in sequence
    }
    if not session:
        raise ValueError("at least one sequence session is required")
    if min(session) < 0:
        raise ValueError("session IDs must be non-negative")
    return tuple(sorted(session))


def generate_population_shift_lags(
    session_lengths: Mapping[int, int],
    *,
    n_shuffle: int,
    seed: int,
    minimum_lag_frames: int,
) -> dict[int, IntArray]:
    """Draw one circular lag per session and surrogate.

    Keys identify recording sessions, so a bracketing square reused by two
    exposure sequences receives one lag, not two. Each lag is shared by every
    neuron in that session. Lags are uniform over integers whose circular
    distance from zero is at least ``minimum_lag_frames``.
    """

    if n_shuffle < 1:
        raise ValueError("n_shuffle must be positive")
    if minimum_lag_frames < 1:
        raise ValueError("minimum_lag_frames must be positive")
    if not session_lengths:
        raise ValueError("at least one session length is required")

    ordered = sorted(
        (int(key), int(value))
        for key, value in session_lengths.items()
    )
    if ordered[0][0] < 0:
        raise ValueError("session IDs must be non-negative")
    if len({key for key, _ in ordered}) != len(ordered):
        raise ValueError("session IDs must be unique")

    generator = np.random.default_rng(seed)
    output: dict[int, IntArray] = {}
    for session, n_frame in ordered:
        if n_frame <= 2 * minimum_lag_frames:
            raise ValueError(
                f"session {session} has {n_frame} frames; it must exceed "
                "twice minimum_lag_frames"
            )
        output[session] = generator.integers(
            minimum_lag_frames,
            n_frame - minimum_lag_frames + 1,
            size=n_shuffle,
            dtype=np.int64,
        )
    return output


def circular_shift_population(
    response: ArrayLike,
    lag_frames: int,
) -> FloatArray:
    """Circularly shift all neurons by one common nonzero lag."""

    event = np.asarray(response, dtype=np.float64)
    if event.ndim != 2 or event.shape[0] < 2:
        raise ValueError("response must have shape (frame, neuron)")
    lag = int(lag_frames) % event.shape[0]
    if lag == 0:
        raise ValueError("lag must be nonzero modulo the session length")
    return np.roll(event, shift=lag, axis=0)


def plus_one_pvalues(
    observed: float,
    surrogate: ArrayLike,
) -> PlusOnePValues:
    """Return directional and absolute two-sided Monte Carlo p-values.

    Non-finite surrogates are reported and excluded. Thus the denominator is
    one plus the number of finite surrogates, rather than silently treating an
    invalid fit as a non-exceedance.
    """

    values = np.asarray(surrogate, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    result: PlusOnePValues = {
        "n_surrogate_requested": int(values.size),
        "n_surrogate_finite": int(finite.size),
        "greater_or_equal": None,
        "two_sided_absolute": None,
    }
    if not np.isfinite(observed) or finite.size == 0:
        return result

    denominator = finite.size + 1
    result["greater_or_equal"] = float(
        (1 + np.count_nonzero(finite >= observed)) / denominator
    )
    result["two_sided_absolute"] = float(
        (
            1
            + np.count_nonzero(
                np.abs(finite) >= abs(float(observed))
            )
        )
        / denominator
    )
    return result
