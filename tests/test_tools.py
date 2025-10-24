#!/usr/bin/env python

"""Simple tests for clustering tools."""

import unittest

import stlearn as st

from .utils import read_test_data


class TestTools(unittest.TestCase):

    def setUp(self):
        self.adata = read_test_data()
        st.em.run_pca(self.adata, n_comps=50, random_state=0)
        st.pp.neighbors(self.adata, n_neighbors=25, use_rep="X_pca", random_state=0)

    def test_imports(self):
        self.assertTrue(callable(st.tl.clustering.annotate_interactive))
        self.assertTrue(callable(st.tl.clustering.kmeans))
        self.assertTrue(callable(st.tl.clustering.leiden))
        self.assertTrue(callable(st.tl.clustering.louvain))

    def test_kmeans(self):
        st.tl.clustering.kmeans(self.adata)
        self.assertIn("kmeans", self.adata.obs.columns)

    def test_louvain_runs(self):
        st.tl.clustering.louvain(self.adata, resolution=1.0)
        self.assertIn("louvain", self.adata.obs.columns)

    def test_leiden_runs(self):
        st.tl.clustering.leiden(self.adata, resolution=1.0)
        self.assertIn("leiden", self.adata.obs.columns)
