"""Tests for read_merfish."""

import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

import stlearn as st
from stlearn.tl.cci.neighbours import calc_distance


class TestReadMerfish(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        folder = pathlib.Path(self._tmp.name)
        # Genes by cells, with numeric cell IDs as in the MERFISH spreadsheets.
        counts = pd.DataFrame(
            [[1, 0, 3, 2], [0, 5, 1, 0]],
            index=["Gad1", "Slc17a6"],
            columns=["101", "102", "103", "104"],
        )
        self.count_file = folder / "counts.csv"
        counts.to_csv(self.count_file)
        # Only three of the cells have coordinates, one of them negative.
        coordinates = pd.DataFrame(
            {"x": [-20.0, 150.0, 300.0], "y": [40.0, 80.0, 400.0]},
            index=pd.Index([103, 101, 104], name="cell"),
        )
        self.spatial_file = folder / "coordinates.xlsx"
        coordinates.to_excel(self.spatial_file)

    def tearDown(self):
        self._tmp.cleanup()

    def test_read_merfish(self):
        adata = st.io.read_merfish(self.count_file, self.spatial_file, scale=2.0)

        self.assertEqual(list(adata.obs_names), ["103", "101", "104"])
        self.assertEqual(list(adata.var_names), ["Gad1", "Slc17a6"])
        np.testing.assert_allclose(adata.X, [[3, 1], [1, 0], [2, 0]])
        # Negative coordinates are shifted by |min| + 100.
        expected = np.array([[100.0, 160.0], [270.0, 200.0], [420.0, 520.0]])
        np.testing.assert_allclose(adata.obsm["spatial"], expected)
        np.testing.assert_allclose(adata.obs[["imagecol", "imagerow"]], expected * 2)

        spatial = adata.uns["spatial"]["MERSEQ"]
        self.assertEqual(spatial["use_quality"], "hires")
        self.assertEqual(spatial["scalefactors"]["tissue_hires_scalef"], 2.0)
        self.assertEqual(spatial["images"]["hires"].shape[2], 3)
        # No blank controls in this matrix.
        self.assertNotIn("blank", adata.obsm)
        self.assertNotIn("blank_fraction", adata.obs)

    def test_scale_in_microns(self):
        adata = st.io.read_merfish(self.count_file, self.spatial_file, scale=2.0)

        scalefactors = adata.uns["spatial"]["MERSEQ"]["scalefactors"]
        self.assertEqual(scalefactors["microns_per_pixel"], 1.0)
        self.assertEqual(scalefactors["spot_diameter_fullres"], 10)
        # The default CCI neighbourhood is twice the cell diameter: 20 microns,
        # 40 pixels of the 2-pixels-per-micron image.
        self.assertAlmostEqual(calc_distance(adata, None), 40.0)


if __name__ == "__main__":
    unittest.main()
