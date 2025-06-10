
# !/usr/bin/env python

"""Tests for tiling function."""

import unittest
import time
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import os
from PIL import Image
from unittest.mock import patch, MagicMock
import filecmp

import scanpy as sc
import stlearn as st

from .utils import read_test_data

global adata
adata = read_test_data()


class TestTiling(unittest.TestCase):
    """Tests for `stlearn` package."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = adata.copy()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir_orig = tempfile.mkdtemp(suffix="_orig")

        # Ensure we have required spatial data
        if "spatial" not in self.test_data.uns:
            self.skipTest("Test data missing spatial information")

        # Add imagerow/imagecol if missing (for testing)
        if "imagerow" not in self.test_data.obs:
            # Create synthetic coordinates for testing
            n_spots = len(self.test_data)
            self.test_data.obs["imagerow"] = np.random.randint(50, 450, n_spots)
            self.test_data.obs["imagecol"] = np.random.randint(50, 450, n_spots)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        if os.path.exists(self.temp_dir_orig):
            shutil.rmtree(self.temp_dir_orig)

    def test_directory_creation(self):
        """Test directory creation behavior."""

        # Test nested directory creation
        nested_path = Path(self.temp_dir) / "level1" / "level2" / "tiles"
        data = self.test_data.copy()

        st.pp.tiling(data, nested_path)

        self.assertTrue(nested_path.exists(), "Nested directories not created")
        self.assertGreater(len(list(nested_path.glob("*"))), 0,
                           "No files in nested directory")

    def test_quality_parameter(self):
        """Test JPEG quality parameter."""
        data = self.test_data[:3].copy()  # Small subset

        # Test different quality settings
        for quality in [50, 95]:
            temp_quality = tempfile.mkdtemp(suffix=f"_q{quality}")
            test_data = data.copy()

            st.pp.tiling(test_data, temp_quality, img_fmt="JPEG", quality=quality)

            # Verify files exist
            jpeg_files = list(Path(temp_quality).glob("*.jpeg"))
            self.assertEqual(len(jpeg_files), len(data))

            shutil.rmtree(temp_quality)