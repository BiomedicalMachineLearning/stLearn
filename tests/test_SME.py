#!/usr/bin/env python

"""Tests for `stlearn` package."""


import unittest

import stlearn as st
import scanpy as sc
from .utils import read_test_data

global adata
adata = read_test_data()


class TestSME(unittest.TestCase):
    """Tests for `stlearn` package."""

    def test_SME(self):
        sc.pp.pca(adata)
        st.pp.tiling(adata, "./tiling")
        st.pp.extract_feature(adata)
        import shutil

        shutil.rmtree("./tiling")
        data_SME = adata.copy()
        # apply stSME to normalise log transformed data
        st.spatial.SME.SME_normalize(data_SME, use_data="raw")
