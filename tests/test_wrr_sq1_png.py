"""Figure 6 preserves legitimate intervals while tolerating float roundoff."""

import numpy as np
import pytest

from scripts.utilities import render_wrr_sq1_png as plot


def test_whisker_length_treats_adjacent_floats_as_a_zero_error_bar():
    center = np.array([150.0, 30.0, 10.0])
    lo = np.array([150.0, np.nextafter(30.0, np.inf), 8.0])
    hi = np.array([np.nextafter(150.0, 0.0), 30.0, 14.0])
    np.testing.assert_array_equal(plot.whisker_lengths(center, lo, hi), [[0, 0, 2], [0, 0, 4]])


def test_whisker_length_refuses_a_materially_inconsistent_interval():
    with pytest.raises(ValueError, match="outside the IQR"):
        plot.whisker_lengths(np.array([8.0]), np.array([9.0]), np.array([10.0]))


@pytest.mark.parametrize("center,lo,hi", [(1.0, 2.0, 1e16), (2.0, -1e16, 1.0)])
def test_far_iqr_bound_does_not_expand_tolerance_of_other_bound(center, lo, hi):
    with pytest.raises(ValueError, match="outside the IQR"):
        plot.whisker_lengths(np.array([center]), np.array([lo]), np.array([hi]))
