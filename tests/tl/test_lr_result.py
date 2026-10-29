#!/usr/bin/env python

import unittest
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd
from anndata import AnnData

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stlearn
from stlearn.tl.cci import LR_RESULTS


def lr_table() -> AnnData:
    """An LR table as st.tl.cci.run returns: 4 spots x 2 LR pairs, without
    neg_log10_p_adjs."""
    lr = AnnData(
        obs=pd.DataFrame(index=[f"spot{i}" for i in range(4)]),
        var=pd.DataFrame(
            {"n_spots": [3, 2], "n_spots_sig": [3, 0], "n_spots_sig_pval": [3, 0]},
            index=["A_B", "C_D"],
        ),
    )
    lr.layers["lr_scores"] = np.array([[1.0, 0.0], [2.0, 0.5], [0.0, 0.0], [3.0, 1.0]])
    lr.layers["p_vals"] = np.array([[0.01, 1.0], [0.02, 0.5], [1.0, 1.0], [0.001, 0.2]])
    lr.layers["p_adjs"] = np.array([[0.02, 1.0], [0.04, 0.5], [1.0, 1.0], [0.002, 0.4]])
    lr.layers["lr_sig_scores"] = np.where(
        lr.layers["p_adjs"] < 0.05, lr.layers["lr_scores"], 0
    )
    return lr


def expression_table() -> AnnData:
    """The table st.tl.cci.run ran on: the same 4 spots, with what the plots need."""
    adata = AnnData(np.zeros((4, 4)))
    adata.obs_names = [f"spot{i}" for i in range(4)]
    adata.var_names = ["A", "B", "C", "D"]
    adata.obs["imagecol"] = [10.0, 20.0, 10.0, 20.0]
    adata.obs["imagerow"] = [10.0, 10.0, 20.0, 20.0]
    adata.obsm["spatial"] = adata.obs[["imagecol", "imagerow"]].to_numpy()
    adata.uns["spatial"] = {
        "s": {
            "images": {"hires": np.zeros((30, 30, 3), dtype=np.uint8)},
            "scalefactors": {"tissue_hires_scalef": 1.0, "spot_diameter_fullres": 5.0},
            "use_quality": "hires",
        }
    }
    return adata


class TestLrResult(unittest.TestCase):
    def test_stored_matrices_are_returned(self):
        lr = lr_table()
        for key in ("lr_scores", "p_vals", "p_adjs", "lr_sig_scores"):
            np.testing.assert_array_equal(
                stlearn.tl.cci.lr_result(lr, key), lr.layers[key]
            )

    def test_log10_is_computed_when_not_stored(self):
        lr = lr_table()
        computed = stlearn.tl.cci.lr_result(lr, "neg_log10_p_adjs")
        np.testing.assert_allclose(computed, -np.log10(lr.layers["p_adjs"]))
        lr.layers["neg_log10_p_adjs"] = np.full((4, 2), 7.0)
        np.testing.assert_array_equal(
            stlearn.tl.cci.lr_result(lr, "neg_log10_p_adjs"),
            lr.layers["neg_log10_p_adjs"],
        )

    def test_unknown_key(self):
        for key in ("scores", "-log10(p_adjs)"):
            with self.subTest(key=key), self.assertRaises(KeyError):
                stlearn.tl.cci.lr_result(lr_table(), key)

    def test_spots_must_match(self):
        """The LR table must have the expression table's spots, in order."""
        with self.assertRaisesRegex(ValueError, "same order"):
            stlearn.pl.lr_result_plot(
                expression_table()[[3, 1]], lr_results=lr_table(), show_plot=False
            )

    def test_without_lr_results(self):
        with self.assertRaisesRegex(ValueError, "lr_results"):
            stlearn.tl.cci.lr_result(expression_table(), "lr_scores")

    def test_result_plot_without_the_stored_matrix(self):
        stlearn.pl.lr_result_plot(
            expression_table(),
            lr_results=lr_table(),
            use_lr="A_B",
            use_result="neg_log10_p_adjs",
            show_plot=False,
            show_image=False,
        )
        plt.close("all")

    def test_lr_plot_shows_the_result_asked_for(self):
        """lr_plot's use_label can be any LR result, and is what it draws."""
        for key in LR_RESULTS:
            with (
                self.subTest(use_label=key),
                mock.patch("stlearn.pl.cci_plot.lr_result_plot") as result_plot,
            ):
                stlearn.pl.lr_plot(
                    expression_table(),
                    "A_B",
                    lr_results=lr_table(),
                    sig_spots=False,
                    use_label=key,
                    show_image=False,
                )
                plt.close("all")
                result_plot.assert_called_once()
                self.assertEqual(result_plot.call_args.kwargs["use_result"], key)

    def test_lr_plot_arrows_need_lr_results(self):
        """The arrows join significant spots, which only the LR results give."""
        with self.assertRaisesRegex(ValueError, "lr_results"):
            stlearn.pl.lr_plot(
                expression_table(),
                "A_B",
                sig_spots=False,
                show_image=False,
                show_arrows=True,
            )
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
