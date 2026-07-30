import numpy as np
import pytest

from ca1_geometry.aliases import (
    boundary_context,
    cellwise_cross_map_correlations,
    cellwise_partition_correlations,
    common_relative_support,
    distance_matched_control_pairs,
    displacement_matched_control_pairs,
    exact_alias_pairs,
    extract_partition,
    grid_displacement_signature,
    mean_fisher_correlation,
    squared_grid_distance,
)


@pytest.mark.parametrize(
    ("blocked", "expected"),
    [
        ((4,), ((1, 7), (3, 5))),
        ((4, 5), ((1, 7), (2, 8))),
        ((0, 8), ((1, 3), (5, 7))),
        ((3, 5), ((0, 6), (2, 8))),
    ],
)
def test_exact_alias_pairs_match_the_four_alias_environments(
    blocked, expected
):
    assert exact_alias_pairs(blocked) == expected


def test_context_includes_outer_and_blocked_neighbor_walls():
    assert boundary_context((4,), 1) == (True, False, True, False)
    assert boundary_context((4,), 7) == (True, False, True, False)
    with pytest.raises(ValueError):
        boundary_context((4,), 4)


def test_partition_extraction_respects_north_to_south_numbering():
    grid = np.arange(225, dtype=float).reshape(15, 15)
    assert np.array_equal(extract_partition(grid, 0), grid[0:5, 0:5])
    assert np.array_equal(extract_partition(grid, 8), grid[10:15, 10:15])


def test_distance_controls_match_distance_and_exclude_aliases():
    controls = distance_matched_control_pairs((4,), (1, 7))
    assert controls
    assert all(squared_grid_distance(pair) == 4 for pair in controls)
    assert not set(controls).intersection(exact_alias_pairs((4,)))


def test_displacement_controls_match_axis_as_well_as_distance():
    controls = displacement_matched_control_pairs((4,), (1, 7))
    assert controls
    assert all(grid_displacement_signature(pair) == (2, 0) for pair in controls)
    assert not set(controls).intersection(exact_alias_pairs((4,)))


def test_common_support_requires_both_tiles_on_every_exposure():
    occupancy = [np.ones((15, 15)), np.ones((15, 15))]
    occupancy[0][0, 5] = 0.2  # relative bin (0, 0) in partition 1
    occupancy[1][10, 5] = 0.3  # same relative bin in partition 7
    support = common_relative_support(
        occupancy,
        (1, 7),
        minimum_seconds=0.5,
    )
    assert support.shape == (5, 5)
    assert not support[0, 0]
    assert support.sum() == 24


def test_cellwise_correlations_preserve_sign_and_constant_is_nan():
    rate = np.zeros((3, 15, 15), dtype=float)
    pattern = np.arange(25, dtype=float).reshape(5, 5)
    rate[0, 0:5, 5:10] = pattern
    rate[0, 10:15, 5:10] = pattern
    rate[1, 0:5, 5:10] = pattern
    rate[1, 10:15, 5:10] = pattern[::-1, ::-1]
    result = cellwise_partition_correlations(
        rate,
        (1, 7),
        np.ones((5, 5), dtype=bool),
    )
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(-1.0)
    assert np.isnan(result[2])


def test_cross_map_correlation_uses_opposite_rate_maps():
    first = np.zeros((1, 15, 15), dtype=float)
    second = np.zeros_like(first)
    pattern = np.arange(25, dtype=float).reshape(5, 5)
    first[0, 0:5, 5:10] = pattern
    second[0, 10:15, 5:10] = pattern
    result = cellwise_cross_map_correlations(
        first,
        second,
        (1, 7),
        np.ones((5, 5), dtype=bool),
    )
    assert result[0] == pytest.approx(1.0)


def test_mean_fisher_correlation():
    assert mean_fisher_correlation([0.2, 0.2, np.nan]) == pytest.approx(0.2)
