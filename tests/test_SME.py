#!/usr/bin/env python

"""Tests for `stlearn` package."""

import shutil
import unittest
from pathlib import Path

import scanpy as sc

import stlearn as st

from .utils import read_test_data


class TestSME(unittest.TestCase):
    """Tests for `stlearn` package."""

    def setUp(self):
        self.adata = read_test_data()
        self.tiling_dir = "./tiling"

    def tearDown(self):
        if Path(self.tiling_dir).exists():
            shutil.rmtree(self.tiling_dir)

    def test_SME(self):
        sc.pp.pca(self.adata)
        st.pp.tiling(self.adata, self.tiling_dir)
        st.pp.extract_feature(self.adata)
        self.assertIn("X_tile_feature", self.adata.obsm)
        self.assertIn("X_morphology", self.adata.obsm)
        self.assertEqual(self.adata.obsm["X_pca"].shape, (316, 50))
        self.assertEqual(self.adata.obsm["X_tile_feature"].shape, (316, 2048))
        self.assertEqual(self.adata.obsm["X_morphology"].shape, (316, 50))
        data_SME = self.adata.copy()
        # apply stSME to normalise log transformed data
        st.spatial.SME.SME_normalize(data_SME, use_data="raw")
