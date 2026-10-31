#!/usr/bin/env python

"""Tests for create_stlearn."""

import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd
import PIL.Image
import scanpy as sc

import stlearn as st


class TestCreateStlearn(unittest.TestCase):
    def setUp(self):
        self.count = pd.DataFrame(
            [[1.0, 3.0], [2.0, 6.0]], index=["s0", "s1"], columns=["GeneA", "GeneB"]
        )
        self.spatial = pd.DataFrame(
            {"imagecol": [0.0, 300.0], "imagerow": [0.0, 300.0]}, index=["s0", "s1"]
        )

    def test_spots_matched_by_barcode(self):
        """spatial's rows are matched to count's by barcode, not by position."""
        shuffled = self.spatial.iloc[::-1]
        adata = st.io.create_stlearn(self.count, shuffled, "lib", scale=1)

        self.assertEqual(adata.obs.loc["s0", ["imagecol", "imagerow"]].tolist(), [0, 0])
        np.testing.assert_array_equal(adata.obsm["spatial"][0], [0.0, 0.0])

    def test_x_is_writable_copy(self):
        """In-place steps such as normalize_total work on X (under pandas 3 a
        DataFrame's values are a read-only view) and leave count unchanged."""
        adata = st.io.create_stlearn(self.count, self.spatial, "lib", scale=1)
        sc.pp.normalize_total(adata, target_sum=4)

        np.testing.assert_allclose(adata.X, [[1.0, 3.0], [1.0, 3.0]])
        self.assertEqual(self.count.loc["s1", "GeneB"], 6.0)

    def image(self) -> pathlib.Path:
        """A 200 x 200 px white image, deleted after the test."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = pathlib.Path(folder.name) / "tissue.png"
        PIL.Image.new("RGB", (200, 200), "white").save(path)
        return path

    def test_image_coordinates_scaled_once(self):
        """With an image, obsm["spatial"] keeps the given coordinates and
        imagecol/imagerow are them times scale, once."""
        adata = st.io.create_stlearn(
            self.count, self.spatial, "lib", image_path=self.image(), scale=0.5
        )

        np.testing.assert_array_equal(adata.obsm["spatial"], self.spatial.to_numpy())
        np.testing.assert_array_equal(
            adata.obs[["imagecol", "imagerow"]], self.spatial.to_numpy() * 0.5
        )
        library = adata.uns["spatial"]["lib"]
        self.assertEqual(library["scalefactors"]["tissue_hires_scalef"], 0.5)
        self.assertEqual(library["images"]["hires"].shape, (200, 200, 3))

    def test_image_without_scale_is_not_scaled(self):
        """With an image and no scale, the coordinates are its pixels."""
        adata = st.io.create_stlearn(
            self.count, self.spatial, "lib", image_path=self.image()
        )

        np.testing.assert_array_equal(
            adata.obs[["imagecol", "imagerow"]], self.spatial.to_numpy()
        )
        scalefactors = adata.uns["spatial"]["lib"]["scalefactors"]
        self.assertEqual(scalefactors["tissue_hires_scalef"], 1)


if __name__ == "__main__":
    unittest.main()
