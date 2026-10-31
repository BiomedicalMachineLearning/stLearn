#!/usr/bin/env python

"""Tests for the SME weighting matrix."""

import unittest

import numpy as np
import pandas as pd
from anndata import AnnData

from stlearn.spatial.sme._weighting_matrix import row_col_by_platform


class TestRowColByPlatform(unittest.TestCase):
    def test_visium(self):
        obs = pd.DataFrame(
            {
                "array_row": [0, 1],
                "array_col": [0, 2],
                "imagerow": [0.0, 10.0],  # 10 px per array_row step
                "imagecol": [0.0, 3.0],  # 1.5 px per array_col step
            },
            index=["a", "b"],
        )
        reg_col, reg_row, rate = row_col_by_platform(AnnData(obs=obs), "Visium")

        self.assertAlmostEqual(reg_row.coef_[0], 10.0)
        self.assertAlmostEqual(reg_col.coef_[0], 1.5)
        self.assertEqual(rate, 3)

    def test_visium_spot_without_array_row(self):
        """A spot with no array_row (NaN) is left out of the fit, rather than
        failing with "Input X contains NaN" (#310)."""
        obs = pd.DataFrame(
            {
                "array_row": [0, 1, np.nan],
                "array_col": [0, 2, 4],
                "imagerow": [0.0, 10.0, 20.0],
                "imagecol": [0.0, 3.0, 6.0],
            },
            index=["a", "b", "c"],
        )
        reg_col, reg_row, _ = row_col_by_platform(AnnData(obs=obs), "Visium")

        self.assertAlmostEqual(reg_row.coef_[0], 10.0)
        self.assertAlmostEqual(reg_col.coef_[0], 1.5)

    def test_visium_no_spot_with_array_row(self):
        """With every array_row missing (NaN), the error says so."""
        obs = pd.DataFrame(
            {
                "array_row": [np.nan, np.nan],
                "array_col": [0, 2],
                "imagerow": [0.0, 10.0],
                "imagecol": [0.0, 3.0],
            },
            index=["a", "b"],
        )
        with self.assertRaisesRegex(ValueError, "No spot has both array_row and"):
            row_col_by_platform(AnnData(obs=obs), "Visium")
