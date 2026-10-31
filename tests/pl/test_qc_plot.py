#!/usr/bin/env python

"""Tests for qc_plot."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt

import stlearn as st
from tests.utils import read_test_data


class TestQCPlot(unittest.TestCase):
    """qc_plot must run with its default arguments and label its size legend."""

    @classmethod
    def setUpClass(cls):
        cls._base_adata = read_test_data()

    def setUp(self):
        self.adata = self.__class__._base_adata.copy()

    def tearDown(self):
        plt.close("all")

    def _legend_labels(self) -> list[str]:
        legend = plt.gcf().axes[0].get_legend()
        self.assertIsNotNone(legend)
        return [text.get_text() for text in legend.get_texts()]

    def test_default_arguments(self):
        """The size legend is on by default; building it must not raise."""
        with tempfile.TemporaryDirectory() as out, patch("matplotlib.pyplot.show"):
            st.pl.qc_plot(self.adata, name="qc.png", output=out)
            self.assertTrue((Path(out) / "qc.png").is_file())

        labels = self._legend_labels()
        self.assertEqual(len(labels), 4)
        values = [int(label) for label in labels]
        self.assertEqual(values, sorted(values))

    def test_output_without_size_legend(self):
        with tempfile.TemporaryDirectory() as out, patch("matplotlib.pyplot.show"):
            st.pl.qc_plot(self.adata, name="qc.png", output=out, show_size_legend=False)
            self.assertTrue((Path(out) / "qc.png").is_file())
        self.assertIsNone(plt.gcf().axes[0].get_legend())
