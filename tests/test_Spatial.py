#!/usr/bin/env python

"""Tests for `stlearn` package."""

import unittest

import numpy.testing as npt

from stlearn.classes import Spatial

from .utils import read_test_data

global adata
adata = read_test_data()


class TestSpatial(unittest.TestCase):
    """Tests for `stlearn` package."""

    def test_setup_Spatial(self):
        spatial = Spatial(adata)
        self.assertIsNotNone(spatial)
        self.assertEqual("V1_Breast_Cancer_Block_A_Section_1", spatial.library_id)
        self.assertEqual("hires", spatial.img_key)
        self.assertEqual(177.4829519178534, spatial.spot_size)
        self.assertEqual(True, spatial.crop_coord)
        self.assertEqual(False, spatial.use_raw)
        npt.assert_array_almost_equal(
            [896.782, 1370.627, 1483.498, 1178.713, 1584.901],
            spatial.imagecol[:5],
            decimal=3,
        )
        npt.assert_array_almost_equal(
            [1549.092, 1158.003, 1040.594, 1373.267, 1021.205],
            spatial.imagerow[:5],
            decimal=3,
        )
