#!/usr/bin/env python

"""Tests for `stlearn` package."""


import unittest

import stlearn as st
import scanpy as sc
from .utils import read_test_data
import numpy as np

global adata
adata = read_test_data()


class TestPSTS(unittest.TestCase):
    """Tests for `stlearn` package."""

    def test_PSTS(self):
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, resolution=0.6)
        sc.tl.louvain(adata)

        adata.uns["iroot"] = np.flatnonzero(adata.obs["leiden"] == "0")[0]
        st.spatial.trajectory.pseudotime(
            adata, eps=100, use_rep="X_pca", use_sme=False, use_label="leiden"
        )
        st.spatial.trajectory.pseudotimespace_global(
            adata, use_label="leiden", list_clusters=[0, 1]
        )
        st.spatial.trajectory.detect_transition_markers_clades(
            adata, clade=0, use_raw_count=False, cutoff_spearman=0.3
        )
