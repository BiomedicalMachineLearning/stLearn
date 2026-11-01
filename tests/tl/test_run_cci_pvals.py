#!/usr/bin/env python

import unittest

import numpy as np
from anndata import AnnData

import stlearn as st
from tests.utils import CCI_LRS, cci_table


class TestRunCCIPvals(unittest.TestCase):
    """run_cci's permutation p-values."""

    @classmethod
    def setUpClass(cls):
        cls.n_perms = 19
        # 1 / (19 + 1) is the cutoff exactly
        cls.p_cutoff = 0.05
        adata = cci_table()
        settings = dict(min_spots=2, distance=1.5, n_pairs=100, verbose=False)
        lr: AnnData = st.tl.cci.run(adata, CCI_LRS, use_label="cell_type", **settings)
        st.tl.cci.run_cci(
            adata,
            "cell_type",
            lr,
            min_spots=0,
            sig_spots=False,
            n_perms=cls.n_perms,
            p_cutoff=cls.p_cutoff,
            verbose=False,
        )
        cls.lr = lr

    def test_never_zero(self):
        """The observed counts are one of the permutations, so p is at least
        1 / (n_perms + 1), and some reach it."""
        pvals = np.stack(
            [p.to_numpy() for p in self.lr.uns["per_lr_cci_pvals_cell_type"].values()]
        )
        smallest = 1 / (self.n_perms + 1)
        self.assertTrue((pvals >= smallest).all())
        self.assertTrue(np.isclose(pvals, smallest).any())

    def test_significant_below_cutoff(self):
        """An interaction is significant when p < p_cutoff, in the counts and the
        significant matrices alike, so p == p_cutoff is not."""
        pvals = self.lr.uns["per_lr_cci_pvals_cell_type"]
        sig = self.lr.uns["per_lr_cci_cell_type"]
        raw = self.lr.uns["per_lr_cci_raw_cell_type"]
        at_cutoff = 0
        for lr_name, p in pvals.items():
            p = p.to_numpy()
            at_cutoff += np.isclose(p, self.p_cutoff).sum()
            significant = p < self.p_cutoff
            np.testing.assert_array_equal(
                sig[lr_name].to_numpy(), np.where(significant, raw[lr_name], 0)
            )
            self.assertEqual(
                self.lr.var.loc[lr_name, "n_cci_sig_cell_type"], significant.sum()
            )
        self.assertGreater(at_cutoff, 0)


if __name__ == "__main__":
    unittest.main()
