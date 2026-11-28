#!/usr/bin/env python

"""Tests for empirical_pvals: the p-value of a spot's score from its row of
background scores, under each null. perform_spot_testing is tested in
test_permutation_gridded_sparse.py, st.tl.cci.run's null in
test_permutation_run_null.py."""

import unittest

import numpy as np

from stlearn.tl.cci.permutation import empirical_pvals


class TestPermutation(unittest.TestCase):
    """The p-value of a spot's score from its row of background scores."""

    def setUp(self):
        self.scores = np.array([3.0, 3.0, 3.0, 3.0])
        self.background = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # no random pair detected
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],  # two detected, both below
                [0, 0, 0, 0, 0, 0, 1, 2, 3, 4],  # four detected, two at or above
                [1, 1, 2, 2, 2, 2, 2, 3, 8, 9],  # all detected, three above
            ],
            dtype=float,
        )

    def test_random_counts_every_pair(self):
        """The share of all the random pairs at or above the score, or at least
        one of them divided by the number of pairs."""
        np.testing.assert_array_equal(
            empirical_pvals(self.scores, self.background, "random"),
            [
                # None detected or 1, 1 / 10 (shape of background column)
                0.1,
                # None detected or 1, 1 / 10 (shape of background column)
                0.1,
                # n > 3 = 2, 2 / 10 (shape of background column)
                0.2,
                # n > 3 = 3, 3 / 10 (shape of background column)
                0.3,
            ],
        )

    def test_codetected_counts_detected_pairs(self):
        """The share of all the codetected pairs at or above the score."""
        np.testing.assert_array_equal(
            empirical_pvals(self.scores, self.background, "codetected"),
            [
                # None detected so 1/1
                1,
                # n > 3 = 0 + 1, 1 / 3 (non-zero = 2 + 1)
                1 / 3,
                # n > 3 = 2 + 1, 3 / 5 (non-zero = 4 + 1)
                0.6,
                # n > 3 = 3 + 1, 4 / 11 (non-zero = 10 + 1)
                4 / 11,
            ],
        )

    def test_codetected_is_the_default(self):
        np.testing.assert_array_equal(
            empirical_pvals(self.scores, self.background),
            empirical_pvals(self.scores, self.background, "codetected"),
        )


if __name__ == "__main__":
    unittest.main()
