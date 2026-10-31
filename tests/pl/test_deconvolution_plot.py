#!/usr/bin/env python

"""Tests for deconvolution_plot."""

import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd

import stlearn as st
from tests.utils import path_for_test_data, read_test_data


class TestDeconvolutionPlot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        adata = read_test_data()[:40].copy()
        proportions = pd.read_csv(
            f"{path_for_test_data()}/"
            "v1_human_breast_cancer_block_a_section_1_proportions.csv",
            index_col=0,
        )
        adata.obsm["deconvolution"] = proportions.reindex(adata.obs_names).fillna(0)
        cls._base_adata = adata

    def setUp(self):
        self.adata = self.__class__._base_adata.copy()

    def tearDown(self):
        plt.close("all")

    def test_default_colours(self):
        """With no colours given, one colour per cell type comes from the colormap."""
        with patch("matplotlib.pyplot.show"):
            st.pl.deconvolution_plot(self.adata)

    def test_named_colormap(self):
        with patch("matplotlib.pyplot.show"):
            st.pl.deconvolution_plot(self.adata, cmap="tab10")
