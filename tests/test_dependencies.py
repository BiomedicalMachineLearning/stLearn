#!/usr/bin/env python

"""Tests that optional code paths stLearn relies on are installed with it."""

import unittest

import numpy as np
import scanpy as sc
from anndata import AnnData


class TestDependencies(unittest.TestCase):
    def test_seurat_v3_variable_genes(self):
        """stlearn.tl.label selects genes with flavor="seurat_v3", which scanpy
        implements with scikit-misc and imports only when that flavour is used."""
        rng = np.random.default_rng(0)
        adata = AnnData(rng.poisson(1.0, size=(200, 60)).astype(np.float32))

        sc.pp.highly_variable_genes(adata, n_top_genes=10, flavor="seurat_v3")

        self.assertEqual(int(adata.var["highly_variable"].sum()), 10)
