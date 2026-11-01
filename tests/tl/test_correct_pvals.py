#!/usr/bin/env python

import unittest

import numpy as np
import scipy.sparse
from statsmodels.stats.multitest import multipletests

from stlearn.tl.cci._results import correct_pvals, scored_entries, to_dense
from tests.utils import RESULTS_LRS, scores_and_pvals


def _sparse(scores, pvals):
    """The scores above 0 and their p-values as the pipeline's sparse matrices."""
    sparse_scores = scored_entries(scores)
    entries = sparse_scores.tocoo()
    sparse_pvals = sparse_scores.copy()
    sparse_pvals.data = pvals[entries.row, entries.col]
    return sparse_scores, sparse_pvals


class TestCorrectPvals(unittest.TestCase):
    def setUp(self):
        self.scores, self.pvals = scores_and_pvals()

    def test_per_spot(self):
        """Each spot is corrected over the LRs scored in it; the rest stay 1."""
        padjs = correct_pvals(self.scores, self.pvals, "spot", "fdr_bh")
        for spot in range(len(self.scores)):
            tested = self.scores[spot] > 0
            expected = np.ones(len(RESULTS_LRS))
            if tested.any():
                expected[tested] = multipletests(
                    self.pvals[spot, tested], method="fdr_bh"
                )[1]
            np.testing.assert_allclose(padjs[spot], expected, err_msg=spot)

    def test_per_lr(self):
        """Each LR is corrected over the spots it is scored in; the rest stay 1."""
        padjs = correct_pvals(self.scores, self.pvals, "LR", "fdr_bh")
        for j in range(len(RESULTS_LRS)):
            tested = self.scores[:, j] > 0
            expected = np.ones(len(self.scores))
            expected[tested] = multipletests(self.pvals[tested, j], method="fdr_bh")[1]
            np.testing.assert_allclose(padjs[:, j], expected, err_msg=RESULTS_LRS[j])

    def test_no_correction(self):
        padjs = correct_pvals(self.scores, self.pvals, None)
        np.testing.assert_array_equal(padjs, self.pvals)
        self.assertIsNot(padjs, self.pvals)

    def test_sparse_inputs_give_sparse_padjs(self):
        """The pipeline's sparse scores and p-values (entries for the scored
        spots) give a CSC matrix of adjusted p-values on the same entries, the
        values the dense arrays give, along either axis or uncorrected."""
        scores, pvals = _sparse(self.scores, self.pvals)
        for axis in ("spot", "LR", None):
            with self.subTest(axis=axis):
                padjs = correct_pvals(scores, pvals, axis, "fdr_bh")
                self.assertIsInstance(padjs, scipy.sparse.csc_matrix)
                np.testing.assert_array_equal(padjs.indptr, scores.indptr)
                np.testing.assert_array_equal(padjs.indices, scores.indices)
                expected = correct_pvals(self.scores, self.pvals, axis, "fdr_bh")
                np.testing.assert_array_equal(to_dense(padjs, 1.0), expected)

    def test_fdr_bh_is_statsmodels_to_the_bit(self):
        """The in-house Benjamini-Hochberg gives exactly multipletests' values,
        with ties, p-values of 1, a group of one, and many small groups."""
        rng = np.random.default_rng(1)
        n_spots, n_lrs = 500, 12
        scores = rng.gamma(2.0, 1.0, size=(n_spots, n_lrs))
        scores[rng.random(scores.shape) < 0.6] = 0
        scores[0, 1:] = 0  # a spot with one LR
        pvals = np.round(rng.uniform(0, 1, size=scores.shape), 2)  # ties
        pvals[rng.random(scores.shape) < 0.1] = 1.0
        pvals[scores == 0] = 1
        for axis, groups in [("spot", scores), ("LR", scores.T)]:
            padjs = correct_pvals(scores, pvals, axis, "fdr_bh")
            ps = pvals if axis == "spot" else pvals.T
            got = padjs if axis == "spot" else padjs.T
            for i in range(len(groups)):
                tested = groups[i] > 0
                if tested.any():
                    expected = multipletests(ps[i, tested], method="fdr_bh")[1]
                    np.testing.assert_array_equal(got[i, tested], expected)
                np.testing.assert_array_equal(got[i, ~tested], 1)

    def test_other_methods_use_statsmodels(self):
        padjs = correct_pvals(self.scores, self.pvals, "spot", "bonferroni")
        for spot in range(len(self.scores)):
            tested = self.scores[spot] > 0
            if tested.any():
                np.testing.assert_array_equal(
                    padjs[spot, tested],
                    multipletests(self.pvals[spot, tested], method="bonferroni")[1],
                )

    def test_unknown_axis(self):
        with self.assertRaisesRegex(ValueError, "correct_axis"):
            correct_pvals(self.scores, self.pvals, "spots")
