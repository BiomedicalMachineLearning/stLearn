#!/usr/bin/env python

import unittest

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

from stlearn.tl.cci._results import (
    LR_RESULTS,
    correct_pvals,
    lr_result,
    scored_entries,
    store_lr_results,
)
from tests.utils import RESULTS_LRS, scores_and_pvals

CUTOFF = 0.05


class TestStoreLrResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scores, cls.pvals = scores_and_pvals()
        cls.padjs = correct_pvals(cls.scores, cls.pvals, "spot", "fdr_bh")
        # An LR table, LR pairs in the inputs' order, with a stale column as
        # run_cci leaves behind.
        cls.lr = AnnData(
            obs=pd.DataFrame(index=[f"spot{i}" for i in range(len(cls.scores))]),
            var=pd.DataFrame({"n_cci_sig_x": 1.0}, index=RESULTS_LRS),
        )
        store_lr_results(cls.lr, cls.scores, cls.pvals, cls.padjs, CUTOFF)
        cls.summary = cls.lr.var
        # Where each stored column came from in the inputs.
        cls.order = np.array([list(RESULTS_LRS).index(lr) for lr in cls.summary.index])

    def test_summary_counts(self):
        """Per LR: scored spots, significant spots by adjusted and by raw
        p-value. Most significant LR first; stale columns dropped."""
        self.assertEqual(
            list(self.summary.columns), ["n_spots", "n_spots_sig", "n_spots_sig_pval"]
        )
        order = self.order
        np.testing.assert_array_equal(
            self.summary["n_spots"], (self.scores > 0).sum(0)[order]
        )
        np.testing.assert_array_equal(
            self.summary["n_spots_sig"], (self.padjs < CUTOFF).sum(0)[order]
        )
        np.testing.assert_array_equal(
            self.summary["n_spots_sig_pval"], (self.pvals < CUTOFF).sum(0)[order]
        )
        self.assertGreater(self.summary["n_spots_sig"].iloc[0], 0)
        self.assertTrue((np.diff(self.summary["n_spots_sig"]) <= 0).all())

    def test_layers_follow_summary(self):
        order = self.order
        expected = {
            "lr_scores": self.scores[:, order],
            "p_vals": self.pvals[:, order],
            "p_adjs": self.padjs[:, order],
            "neg_log10_p_adjs": -np.log10(self.padjs[:, order]),
            "lr_sig_scores": np.where(
                self.padjs[:, order] < CUTOFF, self.scores[:, order], 0
            ),
        }
        self.assertEqual(set(expected), set(LR_RESULTS))
        self.assertEqual(set(self.lr.layers), set(LR_RESULTS))
        for key, value in expected.items():
            np.testing.assert_allclose(lr_result(self.lr, key), value, err_msg=key)

    def test_same_table_from_sparse_inputs(self):
        """The pipeline's sparse scores and p-values (entries for the scored
        spots) give the table the dense arrays give; the inputs are left as they
        were."""
        scores = scored_entries(self.scores)
        entries = scores.tocoo()  # in the stored order, as nonzero() is not
        pvals = scores.copy()
        pvals.data = self.pvals[entries.row, entries.col]
        padjs = correct_pvals(scores, pvals, "spot", "fdr_bh")
        given = [m.copy() for m in (scores, pvals, padjs)]

        lr = self._table()
        store_lr_results(lr, scores, pvals, padjs, CUTOFF)

        pd.testing.assert_frame_equal(lr.var, self.summary)
        for key in LR_RESULTS:
            with self.subTest(key=key):
                self.assertIsInstance(lr.layers[key], scipy.sparse.csc_matrix)
                np.testing.assert_array_equal(
                    lr_result(lr, key), lr_result(self.lr, key)
                )
        for before, after in zip(given, (scores, pvals, padjs), strict=True):
            np.testing.assert_array_equal(after.toarray(), before.toarray())
        np.testing.assert_array_equal(self.scores, scores_and_pvals()[0])

    def _table(self):
        return AnnData(
            obs=pd.DataFrame(index=[f"spot{i}" for i in range(len(self.scores))]),
            var=pd.DataFrame(index=RESULTS_LRS),
        )

    def test_replaces_previous_results_and_keeps_other_layers_aligned(self):
        """Storing again, as adj_pvals does with the table's own layers and new
        p-values, re-ranks the pairs; a layer of the table that is not a result
        moves with them, and the summary and layers are the fresh ones."""
        lr = self._table()
        store_lr_results(lr, self.scores, self.pvals, self.padjs, CUTOFF)
        lr.layers["mine"] = lr.layers["lr_scores"].copy()
        # Move the significance, so the ranking changes: the top pair's
        # p-values to 1 and the last pair's to 0.001, in the sparse entries.
        pvals = lr.layers["p_vals"].copy()
        pvals.data[pvals.indptr[0] : pvals.indptr[1]] = 1.0
        pvals.data[pvals.indptr[-2] : pvals.indptr[-1]] = 0.001
        padjs = correct_pvals(lr.layers["lr_scores"], pvals, None)
        store_lr_results(lr, lr.layers["lr_scores"], pvals, padjs, CUTOFF)
        self.assertEqual(lr.var_names[0], self.lr.var_names[-1])
        self.assertTrue((np.diff(lr.var["n_spots_sig"]) <= 0).all())
        np.testing.assert_array_equal(
            lr.layers["mine"].toarray(), lr_result(lr, "lr_scores")
        )
        np.testing.assert_array_equal(lr_result(lr, "p_adjs"), lr_result(lr, "p_vals"))
        for name in RESULTS_LRS:
            with self.subTest(lr=name):
                np.testing.assert_array_equal(
                    lr_result(lr, "lr_scores", name),
                    lr_result(self.lr, "lr_scores", name),
                )
