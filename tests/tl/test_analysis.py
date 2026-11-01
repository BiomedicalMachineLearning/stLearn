#!/usr/bin/env python

import tempfile
import unittest

import anndata
import numpy as np
import pandas as pd
from anndata import AnnData

import stlearn as st
from stlearn.tl.cci import LRPairs


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


LRS = LRPairs.from_names(["Lig0_Rec0", "Lig1_Rec1"], species="human")


class TestBase(unittest.TestCase):
    def test_lr_summary_wide_enough_for_int64_counts(self):
        adata = _make_cci_adata(dtype=np.float64)

        lr_results = st.tl.cci.run(
            adata,
            LRS,
            min_spots=0,
            distance=1.5,
            n_pairs=100,
            verbose=False,
        )

        summary = lr_results.var
        for col in ("n_spots", "n_spots_sig", "n_spots_sig_pval"):
            # int32 (itemsize 4) is the bug; int64 (itemsize 8) is the fix.
            self.assertGreaterEqual(
                summary[col].dtype.itemsize,
                8,
                msg=f"lr_summary['{col}'] is {summary[col].dtype}; "
                f"too narrow to hold int64 counts (bug 2).",
            )
        st.tl.cci.adj_pvals(lr_results, pval_adj_cutoff=0.05, correct_axis="spot")
        self.assertIn("p_adjs", lr_results.layers)

    def test_run_takes_only_lr_pairs(self):
        """'L_R' names or a database name are refused, saying how to make LRPairs."""
        settings = {"min_spots": 0, "distance": 1.5, "n_pairs": 100, "verbose": False}
        for lrs in (LRS.names(), list(LRS.names()), "connectomeDB2025"):
            with self.subTest(lrs=lrs), self.assertRaises(TypeError) as raised:
                st.tl.cci.run(_make_cci_adata(), lrs, **settings)
            self.assertIn("LRPairs.from_names", str(raised.exception))

    def test_run_merges_a_repeated_pair(self):
        """A pair listed twice is tested once."""
        settings = {"min_spots": 0, "distance": 1.5, "n_pairs": 100, "verbose": False}
        names = [*LRS.names(), LRS.names()[0]]
        lrs = st.tl.cci.LRPairs.from_names(names, species="human")
        lr_results = st.tl.cci.run(_make_cci_adata(), lrs, **settings)
        self.assertEqual(sorted(lr_results.var_names), sorted(LRS.names()))

    def test_results_record_sources_and_species(self):
        """Each tested pair's source is in var['source'], kept by adj_pvals and in a
        saved LR table; the species is in uns['lr_species']."""
        lrs = st.tl.cci.LRPairs.from_names(
            ["Lig0_Rec0"], species="human", source="db"
        ) | st.tl.cci.LRPairs.from_names(
            ["Lig0_Rec0", "Lig1_Rec1"], species="human", source="mine"
        )
        lr_results = st.tl.cci.run(
            _make_cci_adata(),
            lrs,
            min_spots=0,
            distance=1.5,
            n_pairs=100,
            verbose=False,
        )
        expected = {"Lig0_Rec0": "db | mine", "Lig1_Rec1": "mine"}
        self.assertEqual(lr_results.var["source"].to_dict(), expected)
        self.assertEqual(lr_results.uns["lr_species"], "human")

        st.tl.cci.adj_pvals(lr_results, correct_axis="LR")
        self.assertEqual(lr_results.var["source"].to_dict(), expected)

        with tempfile.TemporaryDirectory() as folder:
            path = f"{folder}/lr_results.h5ad"
            lr_results.write_h5ad(path)
            saved = anndata.read_h5ad(path)
        self.assertEqual(saved.var["source"].to_dict(), expected)
        self.assertEqual(saved.uns["lr_species"], "human")

        with self.assertRaisesRegex(ValueError, "pairs are human"):
            st.tl.cci.run_lr_go(lr_results, r_path="R", species="mouse")
