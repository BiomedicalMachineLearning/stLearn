#!/usr/bin/env python

"""Tests for `stlearn` package."""

import unittest

import numpy as np

import stlearn as st

from tests.utils import read_test_data

class TestPSTS(unittest.TestCase):
    """Tests for `stlearn` package."""

    @classmethod
    def setUpClass(cls):
        cls._base_adata = read_test_data()

    def test_psts(self):
        self.adata = self.__class__._base_adata.copy()
        st.em.run_pca(self.adata)
        print("Done PCA!")
        st.pp.neighbors(self.adata)
        print("Done KNN!")
        st.tl.clustering.leiden(self.adata, resolution=0.6)
        print("Done leiden!")
        self.adata.uns["iroot"] = np.flatnonzero(self.adata.obs["leiden"] == "0")[0]
        st.spatial.trajectory.pseudotime(
            self.adata, eps=100, use_rep="X_pca", use_sme=False, use_label="leiden"
        )
        st.spatial.trajectory.pseudotimespace_global(
            self.adata, use_label="leiden", list_clusters=[0, 1]
        )
        st.spatial.trajectory.detect_transition_markers_clades(
            self.adata, clade=0, use_raw_count=False, cutoff_spearman=0.3
        )
        print("Done PSTS!")
