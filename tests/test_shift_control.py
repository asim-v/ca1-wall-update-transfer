import numpy as np

from ca1_geometry.shift_control import (
    circular_shift_population,
    generate_population_shift_lags,
    plus_one_pvalues,
    unique_sequence_sessions,
)


def test_shared_sequence_session_gets_one_deterministic_lag_stream():
    sequences = [(0, 5, 10), (10, 15, 20)]
    sessions = unique_sequence_sessions(sequences)
    assert sessions == (0, 5, 10, 15, 20)

    lengths = {session: 1_000 + session for session in sessions}
    first = generate_population_shift_lags(
        lengths,
        n_shuffle=7,
        seed=31,
        minimum_lag_frames=100,
    )
    second = generate_population_shift_lags(
        lengths,
        n_shuffle=7,
        seed=31,
        minimum_lag_frames=100,
    )
    assert tuple(first) == sessions
    assert len(first) == 5
    for session in sessions:
        np.testing.assert_array_equal(first[session], second[session])
        assert np.all(first[session] >= 100)
        assert np.all(first[session] <= lengths[session] - 100)


def test_circular_shift_uses_one_lag_for_the_whole_population():
    response = np.column_stack((np.arange(6), np.arange(6) + 100))
    shifted = circular_shift_population(response, 2)
    np.testing.assert_array_equal(shifted, np.roll(response, 2, axis=0))
    np.testing.assert_array_equal(
        shifted[:, 1] - shifted[:, 0],
        np.full(6, 100.0),
    )


def test_plus_one_pvalues_include_observed_and_ties():
    result = plus_one_pvalues(2.0, [-4.0, 1.0, 2.0, np.nan])
    assert result["n_surrogate_requested"] == 4
    assert result["n_surrogate_finite"] == 3
    assert result["greater_or_equal"] == 0.5
    assert result["two_sided_absolute"] == 0.75
