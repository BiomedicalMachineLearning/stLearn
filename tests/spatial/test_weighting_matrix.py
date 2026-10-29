#!/usr/bin/env python

"""Tests for the SME weighting matrix."""

import unittest

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
