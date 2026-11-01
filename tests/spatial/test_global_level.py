#!/usr/bin/env python

"""Tests for the global level of the spatial trajectory."""

import tracemalloc
import unittest

import numpy as np
import pandas as pd
from anndata import AnnData

from stlearn.spatial.trajectory.global_level import (
    ge_distance_matrix,
    spatial_distance_matrix,
)
from tests.utils import read_test_data

# Between bands of 10 array rows, each split into subclusters of 20 array
# columns: the expression and spatial distance matrices as stLearn gave
# them. Rows and columns are the subclusters in the order they first appear.
EXPECTED = {
    ("3", "4"): {
        "ge": [
            [0.928844740115, 0.92744453454, 0.997526342041],
            [0.92537001119, 0.926212911616, 1.0],
            [0.885270913822, 0.915736243515, 0.909586629981],
        ],
        "spatial": [
            [0.29043376798, 0.235578409461, 0.811260597552],
            [0.148467325869, 0.233914123995, 0.60878256659],
            [0.464226179209, 0.353863103607, 1.0],
        ],
    },
    ("5", "6"): {
        "ge": [
            [0.999632529796, 0.976485354747, 0.97486783178],
            [0.941670232211, 0.973194820082, 0.906531950847],
            [1.0, 0.994640467688, 0.952466270314],
        ],
        "spatial": [
            [0.615463077187, 0.419881299355, 0.552782076074],
            [1.0, 0.656947780798, 0.297323044985],
            [0.29360113577, 0.486640901222, 0.922861874912],
        ],
    },
}


def _bands():
    """The test data with bands of 10 array rows as clusters, split into
    subclusters of 20 array columns, and expression standing in for PCA."""
    adata = read_test_data()
    adata.obsm["X_pca"] = np.asarray(adata.X[:, :20], dtype=np.float64)
    band = (adata.obs["array_row"] // 10).astype(str)
    subclusters = band + "_" + (adata.obs["array_col"] // 20).astype(str)
    adata.obs["band"] = band.astype("category")
    adata.obs["sub_cluster_labels"] = subclusters.astype("category")
    centroids = adata.obs[["imagecol", "imagerow"]].groupby(subclusters).mean()
    adata.uns["centroid_dict"] = {
        label: centroid.to_numpy() for label, centroid in centroids.iterrows()
    }
    return adata


class TestGlobalLevel(unittest.TestCase):
    def test_distance_matrices(self):
        adata = _bands()
        for (first, second), expected in EXPECTED.items():
            with self.subTest(clusters=(first, second)):
                np.testing.assert_allclose(
                    ge_distance_matrix(adata, first, second, "band", "X_pca", 20),
                    expected["ge"],
                    rtol=1e-9,
                )
                np.testing.assert_allclose(
                    spatial_distance_matrix(adata, first, second, "band"),
                    expected["spatial"],
                    rtol=1e-9,
                )

    def test_distance_memory(self):
        """Two subclusters of 6,000 spots: pairwise distances alone
        took 288 MB."""
        n = 6000
        adata = AnnData(
            obs=pd.DataFrame(
                {
                    "leiden": pd.Categorical(["a"] * n + ["b"] * n),
                    "sub_cluster_labels": pd.Categorical(["0"] * n + ["1"] * n),
                },
                index=[str(i) for i in range(2 * n)],
            ),
            obsm={"X_pca": np.random.default_rng(0).normal(size=(2 * n, 20))},
        )
        tracemalloc.start()
        ge_distance_matrix(adata, "a", "b", "leiden", "X_pca", 20)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertLess(peak, 50 * 1024**2)
