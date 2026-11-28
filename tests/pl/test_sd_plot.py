#!/usr/bin/env python

"""Tests for sdata_plot's `image`: "auto", the name of an image, or None."""

import dataclasses
import unittest

import anndata
import geopandas
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import spatialdata
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations import Identity, Scale

from stlearn._compat import spatial_elements
from stlearn.pl import sdata_plot

TABLE = "square_008um"
FULL_RES = 2000  # pixels across the full-resolution image the bins are placed in
# Each image's coordinate system, and its pixels per full-resolution pixel.
IMAGES = {
    "hires_image": ("downscaled_hires", 0.1),
    "lowres_image": ("downscaled_lowres", 0.025),
}


def visium_hd(
    images: tuple[str, ...] = tuple(IMAGES),
    bins_in: tuple[str, ...] = tuple(IMAGES),
) -> spatialdata.SpatialData:
    """A 6 x 6 grid of bins with a `leiden` column, laid out like spatialdata-io's
    Visium HD: the bins are in full-resolution pixels, scaled into the coordinate
    system of each of the images named in `bins_in`.
    """
    corners = 400 + 100 * np.indices((6, 6)).reshape(2, -1).T
    bins = geopandas.GeoDataFrame(
        geometry=[shapely.box(x, y, x + 100, y + 100) for x, y in corners],
        index=[f"bin{i}" for i in range(len(corners))],
    )
    obs = pd.DataFrame(
        {
            "leiden": pd.Categorical((np.arange(len(bins)) % 3).astype(str)),
            "region": pd.Categorical(["bins"] * len(bins)),
            "instance_id": bins.index,
        },
        index=bins.index,
    )

    def image(name: str):
        coordinate_system, scale = IMAGES[name]
        side = int(FULL_RES * scale)
        return Image2DModel.parse(
            np.full((3, side, side), 200, dtype=np.uint8),
            dims=("c", "y", "x"),
            c_coords=["r", "g", "b"],
            transformations={coordinate_system: Identity()},
        )

    return spatialdata.SpatialData(
        images={name: image(name) for name in images},
        shapes={
            "bins": ShapesModel.parse(
                bins,
                transformations={
                    coordinate_system: Scale([scale, scale], axes=("x", "y"))
                    for coordinate_system, scale in map(IMAGES.__getitem__, bins_in)
                },
            )
        },
        tables={
            TABLE: TableModel.parse(
                anndata.AnnData(np.ones((len(bins), 1), dtype=np.float32), obs=obs),
                region="bins",
                region_key="region",
                instance_key="instance_id",
            )
        },
    )


class TestSdataPlotImage(unittest.TestCase):
    def plot(self, sdata: spatialdata.SpatialData, **kwargs) -> matplotlib.axes.Axes:
        """Plot the bins as vector shapes, so the only raster drawn is an image."""
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        return sdata_plot(
            sdata, "leiden", table_key=TABLE, method="matplotlib", ax=ax, **kwargs
        )

    def assert_image_width(self, ax: matplotlib.axes.Axes, width: float | None):
        """One image `width` across was drawn, or none if `width` is None."""
        extents = [image.get_extent() for image in ax.images]
        widths = [abs(right - left) for left, right, _, _ in extents]
        self.assertEqual(widths, [] if width is None else [width])

    def test_auto_chooses_an_image(self):
        """
        The default draws the smallest image, and "hires" the largest.

        The low-res image is 2000 * 0.025 = 50px, the hi-res image is *0.1 = 200px.
        """
        self.assert_image_width(self.plot(visium_hd()), 50)
        self.assert_image_width(self.plot(visium_hd(), image="auto"), 50)
        self.assert_image_width(self.plot(visium_hd(), resolution="hires"), 200)

    def test_name_uses_that_image(self):
        self.assert_image_width(self.plot(visium_hd(), image="hires_image"), 200)

    def test_unknown_name_is_an_error(self):
        with self.assertRaisesRegex(
            ValueError, r"No image 'he'; choose from \['hires_image', 'lowres_image'\]"
        ):
            self.plot(visium_hd(), image="he")

    def test_none_draws_no_image(self):
        """Without an image the bins are where they are with the image."""
        sdata = visium_hd()
        for resolution in ("lowres", "hires"):
            with self.subTest(resolution=resolution):
                ax = self.plot(sdata, image=None, resolution=resolution)

                self.assert_image_width(ax, None)
                with_image = spatial_elements(sdata, TABLE, resolution)
                self.assertEqual(
                    spatial_elements(sdata, TABLE, resolution, None),
                    dataclasses.replace(with_image, image=None),
                )

    def test_none_needs_no_image(self):
        """With no image to match, the bins' only coordinate system is used."""
        sdata = visium_hd(images=(), bins_in=("hires_image",))

        self.assert_image_width(self.plot(sdata, image=None), None)
        self.assertEqual(
            spatial_elements(sdata, TABLE, image=None).coordinate_system,
            "downscaled_hires",
        )

    def test_none_with_several_coordinate_systems(self):
        """With no image to match, the coordinate system has to be given."""
        sdata = visium_hd(images=())

        with self.assertRaisesRegex(ValueError, "pass `coordinate_system`"):
            self.plot(sdata, image=None)
        ax = self.plot(sdata, image=None, coordinate_system="downscaled_lowres")
        self.assert_image_width(ax, None)

    def test_auto_without_an_image_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "pass `image=None`"):
            self.plot(visium_hd(images=()))


if __name__ == "__main__":
    unittest.main()
