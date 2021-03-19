#!/usr/bin/env python

"""Tests for `stlearn` package."""


import unittest
from click.testing import CliRunner

import stlearn as st
import scanpy as sc
from .utils import read_test_data

global adata
adata = read_test_data()    

class TestCCI(unittest.TestCase):
    """Tests for `stlearn` package."""

    def test_cci(self):
        adata.uns["lr"] = ['CROCC_PTAFR']
        st.tl.cci.lr(adata=adata)
        st.tl.cci.permutation(adata,n_pairs=1)