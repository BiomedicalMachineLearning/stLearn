"""Tests for add.parsing (old Spatial Transcriptomics spot coordinates)."""

import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd
from anndata import AnnData

import stlearn as st

# st_spot_detector output: array x, y, adjusted x, y, pixel x, y.
COORDINATES = """\
x\ty\tnew_x\tnew_y\tpixel_x\tpixel_y
10\t12\t10.05\t12.10\t1005.5\t1210.25
11\t12\t11.02\t12.03\t1102.0\t1203.0
12\t13\t12.00\t13.00\t1200.0\t1300.5
"""


class TestParsing(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.coordinates_file = pathlib.Path(self._tmp.name) / "spots.tsv"
        self.coordinates_file.write_text(COORDINATES)
        counts = pd.DataFrame(
            [[1, 0, 2], [0, 0, 3], [4, 0, 0], [5, 7, 0]],
            index=["12x13", "10x12", "11x12", "99x99"],
            columns=["geneA", "geneB", "geneC"],
        )
        self.adata = AnnData(counts.astype(np.float32))

    def tearDown(self):
        self._tmp.cleanup()

    def _check(self, adata):
        self.assertEqual(list(adata.obs_names), ["12x13", "10x12", "11x12"])
        expected = np.array([[1200.0, 1300.5], [1005.5, 1210.25], [1102.0, 1203.0]])
        np.testing.assert_allclose(adata.obsm["spatial"], expected)
        np.testing.assert_allclose(adata.obs[["imagecol", "imagerow"]], expected)
        # geneB only had counts in the dropped spot.
        self.assertEqual(list(adata.var_names), ["geneA", "geneC"])

    def test_copy(self):
        parsed = st.add.parsing(self.adata, self.coordinates_file)
        self._check(parsed)
        self.assertEqual(self.adata.n_obs, 4)

    def test_in_place(self):
        result = st.add.parsing(self.adata, self.coordinates_file, copy=False)
        self.assertIsNone(result)
        self._check(self.adata)


if __name__ == "__main__":
    unittest.main()
