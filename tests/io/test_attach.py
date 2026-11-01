#!/usr/bin/env python

"""Tests for attach_spatial_sdata: cell coordinates in full-resolution pixels."""

import tempfile
import unittest

import anndata
import numpy as np
import spatialdata
from spatialdata.transformations import Identity, set_transformation

from stlearn.io.spatialdata import attach_spatial_sdata
from tests.utils import ALIGNMENT, MARKERS, PIXEL_SIZE, he_image, table


class TestAttach(unittest.TestCase):
    """Calling attach_spatial_sdata again on the same table, as re-running a
    notebook cell does, doesn't convert the coordinates twice."""

    def setUp(self):
        self.he_xy = np.array(list(MARKERS), dtype=float)
        self.microns = (
            self.he_xy @ ALIGNMENT[:2, :2].T + ALIGNMENT[:2, 2]
        ) * PIXEL_SIZE
        self.sdata = spatialdata.SpatialData(images={"he": he_image(False)})

    def attach(self, adata, **kwargs):
        kwargs = {"image_key": "he", "max_px": 40, **kwargs}
        return attach_spatial_sdata(adata, self.sdata, "lib", PIXEL_SIZE, **kwargs)

    def test_wrong_unit(self):
        adata = table(self.microns)
        with self.assertRaisesRegex(ValueError, "spatial_units must be "):
            self.attach(adata, spatial_units="foo")

    def test_twice(self):
        for image_key in ("he", None):
            with self.subTest(image_key=image_key):
                once = self.attach(table(self.microns), image_key=image_key)
                twice = self.attach(
                    self.attach(table(self.microns), image_key=image_key),
                    image_key=image_key,
                )
                np.testing.assert_allclose(twice.obsm["spatial"], once.obsm["spatial"])
                np.testing.assert_allclose(
                    twice.obs[["imagecol", "imagerow"]],
                    once.obs[["imagecol", "imagerow"]],
                )
                self.assertEqual(
                    twice.uns["spatial"]["lib"]["scalefactors"],
                    once.uns["spatial"]["lib"]["scalefactors"],
                )

    def test_change_image(self):
        """Re-attaching with another image or size starts from the cells' positions,
        as attaching the original table would."""
        adata = self.attach(table(self.microns))
        no_image = self.attach(adata, image_key=None)
        np.testing.assert_allclose(
            no_image.obsm["spatial"],
            self.attach(table(self.microns), image_key=None).obsm["spatial"],
        )
        self.assertNotIn("image_key", no_image.uns["spatial"]["lib"]["metadata"])

        again = self.attach(no_image, image_key="he", max_px=20)
        np.testing.assert_allclose(again.obsm["spatial"], self.he_xy, atol=1e-9)

    def test_after_saving(self):
        """The transform is kept when the table is written and read back."""
        adata = self.attach(table(self.microns))
        with tempfile.TemporaryDirectory() as tmp:
            adata.write_zarr(f"{tmp}/table.zarr")
            read = anndata.read_zarr(f"{tmp}/table.zarr")
        self.attach(read)
        np.testing.assert_allclose(read.obsm["spatial"], self.he_xy, atol=1e-9)

    def test_attached_without_transform(self):
        """A table attached before the transform was kept can't be converted back,
        so it's an error rather than converting again."""
        adata = self.attach(table(self.microns))
        adata.uns["spatial"]["lib"]["metadata"].pop(
            "fullres_to_coordinate_system", None
        )
        with self.assertRaisesRegex(ValueError, "adata was attached before"):
            self.attach(adata)

    def test_other_coordinate_system(self):
        """Cells put in pixels from 'global' can't be re-attached in 'aligned', as
        that would read them as 'aligned' positions: an error, though a fresh table
        attaches in 'aligned' fine."""
        image = he_image(False)
        set_transformation(image, Identity(), "aligned")
        self.sdata = spatialdata.SpatialData(images={"he": image})
        self.attach(table(self.microns), coordinate_system="aligned")

        adata = self.attach(table(self.microns))
        with self.assertRaisesRegex(ValueError, "attached in 'global'"):
            self.attach(adata, coordinate_system="aligned")


if __name__ == "__main__":
    unittest.main()
