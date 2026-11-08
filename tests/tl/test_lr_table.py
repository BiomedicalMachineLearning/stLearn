#!/usr/bin/env python

"""st.tl.cci.run's results in a table of their own."""

import unittest
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stlearn as st
from stlearn.tl.cci import LR_RESULTS, lr_result
from tests.utils import CCI_LRS, cci_table

FEATURES = ["nonzero-median", "zero-prop", "median_rank", "prop_rank", "mean_rank"]
SUMMARY = ["source", "n_spots", "n_spots_sig", "n_spots_sig_pval", *FEATURES]
# varm key of the ligands' and receptors' expression quantiles, run's defaults.
QUANTILES = "lr_quantiles"
RUN_QUANTILES = [0.5, 0.75, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.9975, 0.999, 1]
RUN_CCI_UNS = [
    "lr_cci_cell_type",
    "lr_cci_raw_cell_type",
    "per_lr_cci_cell_type",
    "per_lr_cci_pvals_cell_type",
    "per_lr_cci_raw_cell_type",
]
# Where results were kept before they had a table of their own.
OLD_UNS = ["lr_summary", "lrfeatures", *RUN_CCI_UNS]
GO_TERMS = pd.DataFrame(
    {"Description": ["a", "b"], "p.adjust": [0.01, 0.02], "Count": [3, 2]}
)


def run(adata: AnnData):
    return st.tl.cci.run(
        adata,
        CCI_LRS,
        min_spots=2,
        distance=1.5,
        n_pairs=100,
        n_cpus=1,
        use_label="cell_type",
        verbose=False,
    )


def run_cci(adata: AnnData, **kwargs) -> None:
    st.tl.cci.run_cci(
        adata, "cell_type", min_spots=1, n_perms=20, n_cpus=1, verbose=False, **kwargs
    )


class TestLrTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adata = cci_table()
        cls.lr = run(cls.adata)
        cls.with_cci = None
        if isinstance(cls.lr, AnnData):
            cls.with_cci = cls.lr.copy()
            run_cci(cls.adata.copy(), lr_results=cls.with_cci)

    def setUp(self):
        self.assertIsInstance(self.lr, AnnData, "st.tl.cci.run returns the LR table")

    def test_run_returns_the_lr_table(self):
        lr = self.lr
        self.assertTrue(lr.obs_names.equals(self.adata.obs_names))
        self.assertLessEqual(set(lr.var_names), set(CCI_LRS.names()))
        self.assertEqual(lr.var_names[0], "LigA_RecA")  # the most significant
        self.assertEqual(list(lr.var.columns), SUMMARY)
        self.assertEqual(sorted(lr.layers), sorted(LR_RESULTS))
        np.testing.assert_array_equal(
            lr.var["n_spots_sig"], (lr_result(lr, "p_adjs") < 0.05).sum(axis=0)
        )
        self.assertEqual(list(lr.uns), ["lr_species"])
        self.assertEqual(list(lr.varm), [QUANTILES])

    def _genes(self, name):
        """The expression of pair `name`'s ligand and receptor, spots by 2."""
        ligand, receptor = name.split("_")
        return self.adata[:, [ligand, receptor]].X.toarray()

    def test_expression_features_are_var_columns(self):
        """Each pair's expression features sit beside its summary in var, so they
        are read by name like any other column and can never be out of step with
        the layers' columns (what uns['lrfeatures'] was)."""
        lr = self.lr
        features = lr.var[FEATURES]
        self.assertFalse(features.isna().any().any())
        for name in lr.var_names:
            with self.subTest(lr=name):
                expr = self._genes(name)
                nonzero_medians = [
                    np.quantile(gene[gene > 0], 0.5, method="nearest")
                    for gene in expr.T
                ]
                self.assertEqual(
                    features.loc[name, "nonzero-median"], np.mean(nonzero_medians)
                )
                self.assertEqual(features.loc[name, "zero-prop"], (expr == 0).mean())
        # The ranks run over the tested pairs, 0 the lowest expressed; so the most
        # significant pair, LigA_RecA, with its co-expressed corner, ranks top.
        for rank in ("median_rank", "prop_rank"):
            self.assertCountEqual(features[rank], range(lr.n_vars))
        np.testing.assert_array_equal(
            features["mean_rank"], features[["median_rank", "prop_rank"]].mean(axis=1)
        )
        self.assertEqual(features["mean_rank"].idxmax(), "LigA_RecA")

    def test_expression_quantiles_are_in_varm(self):
        """The quantiles of each pair's ligand and receptor expression, which the
        test matched its background genes on, are a DataFrame in varm: a row per
        pair in the table's order, 'L_q' columns then 'R_q'."""
        lr = self.lr
        quantiles = lr.varm[QUANTILES]
        self.assertIsInstance(quantiles, pd.DataFrame)
        self.assertTrue(quantiles.index.equals(lr.var_names))
        expected_columns = [f"L_{float(q)}" for q in RUN_QUANTILES]
        expected_columns += [f"R_{float(q)}" for q in RUN_QUANTILES]
        self.assertEqual(list(quantiles.columns), expected_columns)
        self.assertFalse(quantiles.isna().any().any())
        for name in lr.var_names:
            with self.subTest(lr=name):
                expr = self._genes(name)
                expected = np.quantile(expr, RUN_QUANTILES, axis=0, method="nearest")
                np.testing.assert_array_equal(
                    quantiles.loc[name].to_numpy(), expected.T.ravel()
                )
        # Not all zero: the fixture's co-expressed corner expresses LigA at 6.
        self.assertEqual(quantiles.loc["LigA_RecA", "L_1.0"], 6)

    def test_layers_are_sparse_with_the_tested_spots(self):
        """Each layer is a CSC matrix with an entry for each spot scored for an
        LR, so the table of a large sample stays small; the other spots have
        the result's default, which lr_result fills in: a score of 0, a p-value
        of 1."""
        lr = self.lr
        scores = lr.layers["lr_scores"]
        self.assertIsInstance(scores, scipy.sparse.csc_matrix)
        self.assertTrue((scores.data > 0).all())
        np.testing.assert_array_equal(lr.var["n_spots"], np.diff(scores.indptr))
        for key in LR_RESULTS:
            with self.subTest(key=key):
                layer = lr.layers[key]
                self.assertIsInstance(layer, scipy.sparse.csc_matrix)
                np.testing.assert_array_equal(layer.indptr, scores.indptr)
                np.testing.assert_array_equal(layer.indices, scores.indices)
                dense = lr_result(lr, key)
                self.assertEqual(dense.shape, lr.shape)
                untested = lr_result(lr, "lr_scores") == 0
                default = 1.0 if key in ("p_vals", "p_adjs") else 0.0
                self.assertTrue((dense[untested] == default).all())
                for j, name in enumerate(lr.var_names):
                    np.testing.assert_array_equal(lr_result(lr, key, name), dense[:, j])
                    np.testing.assert_array_equal(lr_result(lr, key, j), dense[:, j])

    def test_expression_table_keeps_only_what_is_per_spot(self):
        self.assertEqual(sorted(self.adata.obsm), ["cci_het", "spatial"])
        self.assertEqual(list(self.adata.obsp), ["spot_neighbours"])
        self.assertFalse(set(self.adata.uns) & set(OLD_UNS))

    def test_without_lr_results(self):
        with self.assertRaisesRegex(ValueError, "lr_results"):
            st.pl.lr_result_plot(self.adata, show_plot=False)
        with self.assertRaisesRegex(ValueError, "lr_results"):
            run_cci(self.adata.copy())

    def test_adj_pvals_reorders_the_table(self):
        """adj_pvals re-ranks the LR pairs; each pair's columns move with it."""
        lr = self.lr.copy()
        lr.layers["mine"] = lr.layers["lr_scores"].copy()  # moves with the pairs too
        # Move the significance, so the ranking changes: the top pair's p-values
        # to 1 and the last pair's to 0.001, in the layer's sparse entries.
        pvals = lr.layers["p_vals"].copy()
        pvals.data[pvals.indptr[0] : pvals.indptr[1]] = 1.0
        pvals.data[pvals.indptr[-2] : pvals.indptr[-1]] = 0.001
        lr.layers["p_vals"] = pvals
        st.tl.cci.adj_pvals(lr, correct_axis=None)
        self.assertEqual(lr.var_names[0], self.lr.var_names[-1])
        self.assertNotEqual(list(lr.var_names), list(self.lr.var_names))
        np.testing.assert_array_equal(
            lr.layers["mine"].toarray(), lr_result(lr, "lr_scores")
        )
        self.assertTrue((np.diff(lr.var["n_spots_sig"]) <= 0).all())
        np.testing.assert_array_equal(lr_result(lr, "p_adjs"), lr_result(lr, "p_vals"))
        for name in lr.var_names:
            with self.subTest(lr=name):
                np.testing.assert_array_equal(
                    lr_result(lr, "lr_scores", name),
                    lr_result(self.lr, "lr_scores", name),
                )
        self.assertEqual(list(lr.var.columns), SUMMARY)
        pd.testing.assert_frame_equal(
            lr.var[FEATURES], self.lr.var.loc[lr.var_names, FEATURES]
        )
        pd.testing.assert_frame_equal(
            lr.varm[QUANTILES], self.lr.varm[QUANTILES].loc[lr.var_names]
        )

    def test_adj_pvals_drops_run_cci_columns_but_keeps_features(self):
        lr = self.with_cci.copy()
        self.assertTrue(set(lr.var.columns) - set(SUMMARY), "run_cci added columns")
        st.tl.cci.adj_pvals(lr)
        self.assertEqual(list(lr.var.columns), SUMMARY)
        pd.testing.assert_frame_equal(
            lr.var[FEATURES], self.lr.var.loc[lr.var_names, FEATURES]
        )

    def test_plots(self):
        """Every plot of LR results draws from the table given as lr_results=."""
        adata = self.adata.copy()
        lr = self.with_cci.copy()
        lr.uns["lr_go"] = GO_TERMS
        top = lr.var_names[0]
        plots = {
            "lr_result_plot": lambda: st.pl.lr_result_plot(
                adata, lr_results=lr, use_lr=top, show_plot=False
            ),
            "lr_plot": lambda: st.pl.lr_plot(
                adata, top, lr_results=lr, use_label="cell_type", show_image=False
            ),
            "lr_plot stat": lambda: st.pl.lr_plot(
                adata, top, lr_results=lr, use_label="p_adjs", show_image=False
            ),
            "lr_summary": lambda: st.pl.lr_summary(lr, show=False),
            "lr_n_spots": lambda: st.pl.lr_n_spots(lr, show=False),
            "lr_diagnostics": lambda: st.pl.lr_diagnostics(lr, show=False),
            "lr_go": lambda: st.pl.lr_go(lr, show=False),
            "cci_check": lambda: st.pl.cci_check(
                adata, "cell_type", lr_results=lr, show=False
            ),
            "ccinet_plot": lambda: st.pl.ccinet_plot(adata, "cell_type", lr_results=lr),
            "ccinet_plot lr": lambda: st.pl.ccinet_plot(
                adata, "cell_type", lr_results=lr, lr=top
            ),
            "cci_map": lambda: st.pl.cci_map(adata, "cell_type", lr_results=lr),
            "lr_cci_map": lambda: st.pl.lr_cci_map(
                adata, "cell_type", lr_results=lr, n_top_lrs=1
            ),
            "lr_chord_plot": lambda: st.pl.lr_chord_plot(
                adata, "cell_type", lr_results=lr, min_ints=0, show=False
            ),
            "lr_chord_plot empty": lambda: st.pl.lr_chord_plot(
                adata, "cell_type", lr_results=lr, min_ints=10**9, show=False
            ),
        }
        for name, plot in plots.items():
            with self.subTest(plot=name), mock.patch("matplotlib.pyplot.show"):
                plot()
                plt.close("all")

    def test_interactive_plots(self):
        """The Bokeh apps are built from the table given as lr_results=."""
        from stlearn.pl.classes_bokeh import BokehLRPlot, BokehSpatialCciPlot

        adata = self.adata.copy()
        # The CCI app colours cell types by these; clustering usually sets them.
        adata.uns["cell_type_colors"] = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        BokehLRPlot(adata, lr_results=self.lr)
        BokehSpatialCciPlot(adata, lr_results=self.with_cci)


if __name__ == "__main__":
    unittest.main()
