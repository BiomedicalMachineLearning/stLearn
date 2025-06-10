#!/usr/bin/env python

import unittest
import numpy as np
import tempfile
import shutil
import os

import scanpy as sc
import stlearn as st

from .utils import read_test_data

global adata
adata = read_test_data()


class TestFeatureExtractionPerformance(unittest.TestCase):
    """Comprehensive tests for feature extraction."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = adata.copy()
        self.temp_dir = tempfile.mkdtemp()
        sc.pp.pca(self.test_data)
        st.pp.tiling(self.test_data, self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_deterministic_behavior(self):
        """Test that results are deterministic with same seed."""
        data1 = self.test_data.copy()
        data2 = self.test_data.copy()

        st.pp.extract_feature(data1, seeds=42, batch=16)
        st.pp.extract_feature(data2, seeds=42, batch=32)

        np.testing.assert_array_equal(
            data1.obsm["X_morphology"],
            data2.obsm["X_morphology"],
            err_msg="Results should be deterministic with same seed"
        )


    def test_copy_behavior(self):
        """Test copy=True vs copy=False behavior."""
        original_data = self.test_data.copy()

        # Test copy=True
        result_copy = st.pp.extract_feature(original_data, copy=True)
        self.assertIsNotNone(result_copy)
        self.assertNotIn("X_morphology", original_data.obsm)
        self.assertIn("X_morphology", result_copy.obsm)

        # Test copy=False
        result_inplace = st.pp.extract_feature(original_data, copy=False)
        self.assertIsNone(result_inplace)
        self.assertIn("X_morphology", original_data.obsm)

