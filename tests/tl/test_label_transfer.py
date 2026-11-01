"""Tests for st.tl.label.run_label_transfer, with R replaced by a stand-in."""

import importlib.util
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from anndata import AnnData

import stlearn as st
from stlearn.tl.label import label as label_module

N_GENES, N_SPOTS, N_CELLS = 300, 60, 80


def fake_label_transfer(st_expr, sc_expr, sc_labels):
    """Seurat's scores as R returns them, labels by spots: spot i is "A" when i
    is even and "B" when odd."""
    a_score = np.where(np.arange(st_expr.shape[1]) % 2 == 0, 0.9, 0.1)
    return pd.DataFrame(
        [a_score, 1 - a_score], index=["A", "B"], columns=st_expr.columns
    )


def fake_r():
    """rpy2's robjects, with identity conversions and the R function above."""
    ro = mock.MagicMock()
    ro.globalenv = {"label_transfer": fake_label_transfer}
    ro.conversion.py2rpy = lambda x: x
    ro.conversion.rpy2py = lambda x: x
    ro.StrVector = lambda x: x
    return ro


@unittest.skipUnless(
    importlib.util.find_spec("skmisc"), "seurat_v3 gene selection needs scikit-misc"
)
class TestRunLabelTransfer(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        means = rng.gamma(0.5, 4.0, N_GENES)
        self.st_data = AnnData(rng.poisson(means, (N_SPOTS, N_GENES)).astype(float))
        self.st_data.obs_names = [f"spot{i}" for i in range(N_SPOTS)]
        self.st_data.var_names = [f"gene{i}" for i in range(N_GENES)]
        self.sc_data = AnnData(rng.poisson(means, (N_CELLS, N_GENES)).astype(float))
        self.sc_data.var_names = self.st_data.var_names
        self.sc_data.obs["cell_type"] = ["A", "B"] * (N_CELLS // 2)

    def run_label_transfer(self, **kwargs):
        rhs = label_module.rhs
        with (
            mock.patch.object(rhs, "rpy2_setup"),
            mock.patch.object(rhs, "ro", fake_r()),
            mock.patch.object(rhs, "pandas2ri", mock.MagicMock()),
            mock.patch.object(rhs, "localconverter", mock.MagicMock()),
        ):
            st.tl.label.run_label_transfer(
                self.st_data,
                self.sc_data,
                "cell_type",
                "/fake/R",
                n_highly_variable=100,
                **kwargs,
            )

    def test_labels_added_to_st_data(self):
        """The labels and scores are on the object passed in, by spot, and
        nothing else about it changes: its genes and expression stay as they were."""
        before = self.st_data.copy()
        self.run_label_transfer()

        self.assertIn("cell_type", self.st_data.obs)
        expected = np.where(np.arange(N_SPOTS) % 2 == 0, "A", "B")
        np.testing.assert_array_equal(self.st_data.obs["cell_type"], expected)
        self.assertIsInstance(self.st_data.obs["cell_type"].dtype, pd.CategoricalDtype)

        self.assertIn("cell_type", self.st_data.uns)
        scores = self.st_data.uns["cell_type"]
        self.assertEqual(list(scores.index), list(self.st_data.obs_names))
        self.assertEqual(list(scores.columns), ["A", "B"])

        self.assertEqual(self.st_data.shape, before.shape)
        np.testing.assert_array_equal(self.st_data.X, before.X)
        self.assertNotIn("highly_variable", self.st_data.var)

    def test_st_label_col(self):
        self.run_label_transfer(st_label_col="predictions")
        self.assertIn("predictions", self.st_data.obs)
        self.assertIn("predictions", self.st_data.uns)
        self.assertNotIn("cell_type", self.st_data.obs)


if __name__ == "__main__":
    unittest.main()
