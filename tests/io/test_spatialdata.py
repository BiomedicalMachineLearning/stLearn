#!/usr/bin/env python

"""Tests for attach_spatial_sdata: cell coordinates in full-resolution pixels."""

import unittest

import numpy as np
import spatialdata

from stlearn.io.spatialdata import attach_spatial_sdata
from tests.utils import ALIGNMENT, MARKERS, PIXEL_SIZE, he_image, table


class TestSpatialData(unittest.TestCase):
    def test_he_image(self):
        """Cells in microns end up on the H&E pixels they were placed at."""
        he_xy = np.array([*MARKERS, (-10, 70)])  # the last cell is outside the image
        microns = (he_xy @ ALIGNMENT[:2, :2].T + ALIGNMENT[:2, 2]) * PIXEL_SIZE

        for multiscale in (True, False):
            with self.subTest(multiscale=multiscale):
                adata = table(microns)
                sdata = spatialdata.SpatialData(images={"he": he_image(multiscale)})
                attach_spatial_sdata(
                    adata, sdata, "lib", PIXEL_SIZE, image_key="he", max_px=40
                )

                np.testing.assert_allclose(adata.obsm["spatial"], he_xy, atol=1e-9)

                spatial = adata.uns["spatial"]["lib"]
                scalefactors = spatial["scalefactors"]
                self.assertEqual(scalefactors["tissue_hires_scalef"], 0.5)  # 80 -> 40
                self.assertAlmostEqual(
                    scalefactors["microns_per_pixel"], 2 * PIXEL_SIZE
                )
                self.assertAlmostEqual(
                    scalefactors["spot_diameter_fullres"], 10 / (2 * PIXEL_SIZE)
                )
                metadata = spatial["metadata"]
                self.assertEqual(
                    set(metadata),
                    {"coordinate_system", "image_key", "fullres_to_coordinate_system"},
                )
                self.assertEqual(metadata["coordinate_system"], "global")
                self.assertEqual(metadata["image_key"], "he")
                np.testing.assert_allclose(
                    metadata["fullres_to_coordinate_system"], ALIGNMENT
                )

                # Each marker's cell sits on that marker's colour in the small image.
                image = spatial["images"]["hires"]
                self.assertEqual(image.shape, (30, 40, 3))
                self.assertEqual(image.dtype, np.uint8)
                for i, colour in enumerate(MARKERS.values()):
                    col, row = adata.obs[["imagecol", "imagerow"]].iloc[i]
                    np.testing.assert_allclose(
                        image[int(row), int(col)], np.array(colour), atol=2
                    )

    def test_no_image(self):
        """Without an image: morphology pixels and a white image from the origin."""
        morphology = np.array([[0.0, 0.0], [399.0, 199.0]])

        for xy, units in [
            (morphology * PIXEL_SIZE, "micron"),
            (morphology, "coordinate_system"),
        ]:
            with self.subTest(spatial_units=units):
                adata = table(xy)
                attach_spatial_sdata(
                    adata,
                    spatialdata.SpatialData(),
                    "lib",
                    PIXEL_SIZE,
                    spatial_units=units,
                    max_px=40,
                )

                np.testing.assert_allclose(adata.obsm["spatial"], morphology)

                spatial = adata.uns["spatial"]["lib"]
                scalefactors = spatial["scalefactors"]
                self.assertEqual(scalefactors["tissue_hires_scalef"], 0.1)  # 400 -> 40
                self.assertEqual(scalefactors["microns_per_pixel"], PIXEL_SIZE)
                np.testing.assert_array_equal(
                    spatial["images"]["hires"],
                    np.full((20, 40, 3), 255, dtype=np.uint8),
                    strict=True,
                )

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "negative"):
            # The blank image starts at the origin, so this cell would be off it.
            attach_spatial_sdata(
                table([[-1.0, 5.0]]), spatialdata.SpatialData(), "lib", PIXEL_SIZE
            )

        sdata = spatialdata.SpatialData(images={"he": he_image(multiscale=False)})
        with self.assertRaisesRegex(ValueError, "no transformation to 'aligned'"):
            attach_spatial_sdata(
                table([[1.0, 5.0]]),
                sdata,
                "lib",
                PIXEL_SIZE,
                image_key="he",
                coordinate_system="aligned",
            )


if __name__ == "__main__":
    unittest.main()
