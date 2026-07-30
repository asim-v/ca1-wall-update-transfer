import numpy as np
import pytest

from ca1_geometry.field_evacuation import (
    map_bin_partition,
    nearest_accessible_clamps,
    nearest_accessible_reflections,
    reflection_contrast,
)


def test_map_bin_partition_uses_direct_paper_row_order():
    assert map_bin_partition(0, 0) == 0
    assert map_bin_partition(7, 7) == 4
    assert map_bin_partition(14, 14) == 8


def test_center_peak_reflects_to_all_four_equidistant_neighbors():
    queries = nearest_accessible_reflections((4,), (7, 7))
    assert {query.direction for query in queries} == {
        "north",
        "east",
        "south",
        "west",
    }
    assert {query.projected_bin for query in queries} == {
        (2, 7),
        (7, 12),
        (12, 7),
        (7, 2),
    }


def test_off_center_peak_uses_nearest_accessible_face():
    query = nearest_accessible_reflections((4,), (5, 8))
    assert len(query) == 1
    assert query[0].direction == "north"
    assert query[0].projected_bin == (4, 8)
    assert len(query[0].tangential_control_bins) == 4


def test_reflection_contrast_compares_tangential_bins():
    query = nearest_accessible_reflections((4,), (5, 8))
    rate = np.zeros((15, 15))
    rate[query[0].projected_bin] = 2.0
    assert reflection_contrast(rate, query) == pytest.approx(2.0)


def test_clamp_uses_first_accessible_bin_at_same_tangent():
    query = nearest_accessible_clamps((4,), (7, 8))
    east = next(item for item in query if item.direction == "east")
    assert east.projected_bin == (7, 10)
