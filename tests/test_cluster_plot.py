#!/usr/bin/env python

"""Tests for ClusterPlot."""

import unittest
from unittest.mock import MagicMock, patch

import matplotlib.colors
import numpy as np
import pandas as pd

from stlearn.pl.classes import ClusterPlot

from .utils import read_test_data

global adata
adata = read_test_data()


class TestClusterPlot(unittest.TestCase):
    """Tests for ClusterPlot."""

    def setUp(self):
        """Set up test data with known clusters."""
        self.adata = adata.copy()

        # Create test clustering data
        n_spots = len(self.adata.obs)
        cluster_labels = np.random.choice(
            ["Cluster_0", "Cluster_1", "Cluster_2"], n_spots
        )
        self.adata.obs["test_clusters"] = pd.Categorical(cluster_labels)

        # Ensure we have a clean slate
        if "test_clusters_colors" in self.adata.uns:
            del self.adata.uns["test_clusters_colors"]

    def test_color_generation_first_call(self):
        """Test that colors are generated correctly on first call."""
        with (
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch.object(ClusterPlot, "_plot_clusters") as _,
            patch.object(ClusterPlot, "_add_image"),
        ):
            # Mock matplotlib components
            mock_fig, mock_ax = MagicMock(), MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            # Create ClusterPlot
            label_name = "test_clusters"
            plot = ClusterPlot(
                adata=self.adata,
                use_label=label_name,
                show_image=False,
                show_color_bar=False,
            )

            # Check that colors were generated
            colors = plot.adata[0].uns[f"{label_name}_colors"]
            self.assertIsNotNone(colors)
            self.assertEqual(len(colors), 3)  # 3 clusters

            # Check that all colors are valid hex colors
            for color in colors:
                self.assertTrue(matplotlib.colors.is_color_like(color))
                self.assertTrue(color.startswith("#"))
                self.assertEqual(len(color), 7)  # #RRGGBB format

    def test_multiple_calls_same_adata(self):
        """Test that multiple calls with same adata work correctly."""
        with (
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch.object(ClusterPlot, "_plot_clusters") as _,
            patch.object(ClusterPlot, "_add_image"),
        ):
            mock_fig, mock_ax = MagicMock(), MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            label_name = "test_clusters"

            # First call
            plot1 = ClusterPlot(
                adata=self.adata,
                use_label=label_name,
                show_image=False,
                show_color_bar=False,
            )

            # Second call with same adata
            plot2 = ClusterPlot(
                adata=self.adata,
                use_label=label_name,
                show_image=False,
                show_color_bar=False,
            )

            # Both should succeed and generate consistent colors
            colors1 = plot1.adata[0].uns[f"{label_name}_colors"]
            colors2 = plot2.adata[0].uns[f"{label_name}_colors"]

            self.assertEqual(len(colors1), len(colors2))
            self.assertEqual(colors1, colors2)

    def test_insufficient_existing_colors_extended(self):
        """Test that insufficient existing colors are extended."""
        # Pre-populate adata with insufficient colors (only 2 colors for 3 clusters)
        existing_colors = ["#FF0000", "#00FF00"]
        label_name = "test_clusters"
        self.adata.uns[f"{label_name}_colors"] = existing_colors

        with (
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch.object(ClusterPlot, "_plot_clusters") as _,
            patch.object(ClusterPlot, "_add_image"),
        ):
            mock_fig, mock_ax = MagicMock(), MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            plot = ClusterPlot(
                adata=self.adata,
                use_label=label_name,
                show_image=False,
                show_color_bar=False,
            )

            # Should extend existing colors
            colors = plot.adata[0].uns[f"{label_name}_colors"]
            self.assertEqual(len(colors), 3)
            self.assertNotEqual(colors[:2], existing_colors)

    def tearDown(self):
        """Clean up after each test."""
        # Clear any test artifacts
        if hasattr(self, "adata") and "test_clusters_colors" in self.adata.uns:
            del self.adata.uns["test_clusters_colors"]
