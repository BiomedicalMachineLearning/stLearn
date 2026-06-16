#!/usr/bin/env python

import unittest

import numpy as np
import pandas as pd
from anndata import AnnData

import stlearn as st
from stlearn.tl.cci.base import calc_neighbours, get_lrs_scores


def _make_cci_adata(dtype=np.float64, n_side=8, n_extra_genes=40, seed=0):
    rng = np.random.default_rng(seed)
    n_spots = n_side * n_side

    lr_genes = ["Lig0", "Rec0", "Lig1", "Rec1"]  # two L-R pairs
    extra_genes = [f"gene{i}" for i in range(n_extra_genes)]
    var_names = lr_genes + extra_genes

    X = rng.poisson(3.0, size=(n_spots, len(var_names))).astype(dtype)

    rows, cols = np.divmod(np.arange(n_spots), n_side)
    obs = pd.DataFrame(
        {"imagerow": rows.astype(float), "imagecol": cols.astype(float)},
        index=[f"spot{i}" for i in range(n_spots)],
    )
    var = pd.DataFrame(index=var_names)
    return AnnData(X=X, obs=obs, var=var)


LRS = np.array(["Lig0_Rec0", "Lig1_Rec1"])


class TestBase(unittest.TestCase):

    def test_lr_core_accepts_integer_expression(self):
        adata = _make_cci_adata(dtype=np.int64)
        neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
        het_vals = np.ones(len(adata))  # default het in run() is ones

        # This line raises the unification TypingError on the un-fixed code.
        lr_scores, new_lrs = get_lrs_scores(adata, LRS, neighbours, het_vals, 0)

        self.assertEqual(lr_scores.dtype, np.float64)
        self.assertEqual(lr_scores.shape[0], len(adata))
        self.assertEqual(lr_scores.shape[1], len(new_lrs))

    def test_lr_summary_wide_enough_for_int64_counts(self):
        adata = _make_cci_adata(dtype=np.float64)

        st.tl.cci.run(
            adata,
            LRS,
            min_spots=0,
            distance=1.5,
            n_pairs=100,
            verbose=False,
        )

        summary = adata.uns["lr_summary"]
        for col in ("n_spots", "n_spots_sig", "n_spots_sig_pval"):
            # int32 (itemsize 4) is the bug; int64 (itemsize 8) is the fix.
            self.assertGreaterEqual(
                summary[col].dtype.itemsize,
                8,
                msg=f"lr_summary['{col}'] is {summary[col].dtype}; "
                f"too narrow to hold int64 counts (bug 2).",
            )
        st.tl.cci.adj_pvals(adata, correct_axis="spot", pval_adj_cutoff=0.05)
        self.assertIn("p_adjs", adata.obsm)
