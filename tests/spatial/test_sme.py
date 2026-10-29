#!/usr/bin/env python

"""Tests for `stlearn` package."""

import shutil
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

import stlearn as st
from stlearn.spatial.sme._adjust import adjust_by_blocks
from stlearn.spatial.sme._weighting_matrix import (
    pseudo_spot_weights,
    row_col_by_platform,
)
from tests.utils import read_test_data

# For each weighting, on sample(): the sum of |values| sme_normalize and
# sme_impute0 give, and the sum of each spot's largest weight.
EXPECTED = {
    "weights_matrix_all": (
        11173.897611551678,
        11872.972061122415,
        278.0260033789748,
    ),
    "weights_matrix_pd_gd": (
        11146.370136407313,
        14181.215509849124,
        166.66404913516106,
    ),
    "weights_matrix_pd_md": (
        11177.652979995997,
        13732.040492215894,
        186.85585584538546,
    ),
    "weights_matrix_gd_md": (
        11188.82846632614,
        12775.914289217273,
        227.3451369108932,
    ),
    "gene_expression_correlation": (
        11150.190054192946,
        14930.363167899792,
        127.05504500112053,
    ),
    "physical_distance": (
        11178.109831611568,
        17163.65576902194,
        52.83333333333333,
    ),
    "morphological_distance": (
        11227.298395641634,
        14567.498920601507,
        143.0498124085304,
    ),
}
# For pseudo-spots half a step right of each spot: the sum of |imputed values|
# and of each pseudo-spot's largest weight, as stLearn gave them.
PSEUDO_SPOT_EXPECTED = {
    "weights_matrix_all": (9745.812478213109, 219.59094983692614),
    "morphological_distance": (10995.437931344459, 114.4610657528433),
    "physical_distance": (11156.624894271832, 98.66666666666666),
}


def sample():
    """The test data, with stand-ins that don't depend on the image model or a
    PCA solver: the scaled expression of its first 20 genes for PCA and of the
    next 30 for morphology. The positive part is the expression, so half of it
    is zeros for sme_impute0 to fill."""
    adata = read_test_data()
    adata.layers["scaled"] = np.asarray(adata.X, dtype=np.float64)
    adata.obsm["X_pca"] = adata.layers["scaled"][:, :20]
    adata.obsm["X_morphology"] = adata.layers["scaled"][:, 20:50]
    adata.X = np.clip(adata.X, 0, None)
    return adata


def total(matrix):
    """Sum of |values|, dense or sparse."""
    return float(abs(matrix).sum())


class TestSME(unittest.TestCase):
    """Tests for `stlearn` package."""

    def setUp(self):
        self.adata = read_test_data()
        self.tiling_dir = "./tiling"

    def tearDown(self):
        if Path(self.tiling_dir).exists():
            shutil.rmtree(self.tiling_dir)

    def test_sme(self):
        sc.pp.pca(self.adata)
        st.pp.tiling(self.adata, self.tiling_dir)
        st.pp.extract_feature(self.adata)
        self.assertIn("X_tile_feature", self.adata.obsm)
        self.assertIn("X_morphology", self.adata.obsm)
        self.assertEqual(self.adata.obsm["X_pca"].shape, (316, 50))
        self.assertEqual(self.adata.obsm["X_tile_feature"].shape, (316, 2048))
        self.assertEqual(self.adata.obsm["X_morphology"].shape, (316, 50))
        data_sme = self.adata.copy()
        # apply stSME to normalise log transformed data
        st.spatial.sme.sme_normalize(data_sme, use_data="raw")

    def test_values(self):
        self._check_values()

    def test_values_in_chunks(self):
        """The same values when spots are weighted a few at a time."""
        with mock.patch.multiple(
            "stlearn.spatial.sme._weighting_matrix",
            _CHUNK_SPOTS=7,
            _BLOCK_BYTES=8 * 316 * 5,
        ):
            self._check_values()

    def _check_values(self):
        for weights, expected in EXPECTED.items():
            with self.subTest(weights):
                adata = sample()
                st.spatial.sme.sme_normalize(adata, weights=weights)
                largest = adata.obsm["top_weights"][:, -1].sum()
                st.spatial.sme.sme_impute0(adata, weights=weights)
                found = (
                    total(adata.obsm["raw_SME_normalized"]),
                    total(adata.obsm["raw_SME_imputed"]),
                    largest,
                )
                np.testing.assert_allclose(found, expected, rtol=1e-9)
                # keep spots x spots matrix in uns for each weighting.
                self.assertEqual(set(adata.uns) & set(EXPECTED), set())

    def test_pseudo_spot_values(self):
        adata = sample()
        reg_col, reg_row, _ = row_col_by_platform(adata, "Visium")
        step = np.hypot(reg_row.coef_[0], reg_col.coef_[0])
        pseudo = AnnData(obs=adata.obs[["imagecol", "imagerow"]] + [step / 2, 0])
        pseudo.obs_names = [f"pseudo_{name}" for name in adata.obs_names]
        pseudo.obsm["X_morphology"] = adata.layers["scaled"][:, 50:80]
        for weights, expected in PSEUDO_SPOT_EXPECTED.items():
            with self.subTest(weights):
                neighbours, top_weights = pseudo_spot_weights(
                    adata, pseudo, weights=weights
                )
                found = (total(neighbours @ adata.X), top_weights[:, -1].sum())
                np.testing.assert_allclose(found, expected, rtol=1e-9)

    def test_pseudo_spot(self):
        """Runs end to end on dense and sparse expression, with tiling and
        feature extraction (which need the image model) replaced by stand-in
        morphology."""

        def features(pseudo, *args, **kwargs):
            pseudo.obsm["X_morphology"] = np.random.default_rng(1).normal(
                size=(pseudo.n_obs, 30)
            )

        for is_sparse in (False, True):
            with self.subTest(sparse=is_sparse):
                adata = sample()
                if is_sparse:
                    adata.X = csr_matrix(adata.X)
                before = adata.copy()
                with (
                    mock.patch("stlearn.pp.tiling"),
                    mock.patch("stlearn.pp.extract_feature", side_effect=features),
                ):
                    pseudo = st.spatial.sme.pseudo_spot(adata, use_data="raw")
                    combined = st.spatial.sme.pseudo_spot(
                        adata, use_data="raw", copy="combined_adata"
                    )
                self.assertGreater(pseudo.n_obs, 0)
                self.assertEqual(pseudo.n_vars, adata.n_vars)
                self.assertTrue((np.asarray(pseudo.X.sum(axis=1)).ravel() > 0).all())
                self.assertEqual(combined.n_obs, adata.n_obs + pseudo.n_obs)
                self.assertNotIn("weights_matrix_all", pseudo.uns)
                # The input is only read.
                self.assertEqual(total(adata.X - before.X), 0)
                self.assertEqual(list(adata.obs.columns), list(before.obs.columns))

    def test_sparse_data_stays_sparse(self):
        self._check_sparse_and_dense()

    def test_values_in_blocks(self):
        """The same when imputed values are combined a few spots at a time."""
        with mock.patch.object(adjust_by_blocks, "__defaults__", (7,)):
            self._check_sparse_and_dense()

    def _check_sparse_and_dense(self):
        expected = EXPECTED["weights_matrix_all"][:2]
        for is_sparse in (False, True):
            with self.subTest(sparse=is_sparse):
                adata = sample()
                if is_sparse:
                    adata.X = csr_matrix(adata.X)
                st.spatial.sme.sme_normalize(adata)
                st.spatial.sme.sme_impute0(adata)
                results = [
                    adata.obsm["raw_SME_normalized"],
                    adata.obsm["raw_SME_imputed"],
                ]
                for result in results:
                    self.assertEqual(issparse(result), is_sparse)
                    self.assertEqual(result.dtype, np.float64)
                found = [total(result) for result in results]
                np.testing.assert_allclose(found, expected, rtol=1e-9)
                self.assertNotIn("imputed_data", adata.obsm)
