"""Tests for st.utils.microns_to_pixels and st.utils.pixels_to_microns."""

import unittest

import numpy as np
from anndata import AnnData

import stlearn as st
from stlearn.tl.cci.neighbours import calc_neighbours


def _spots(microns_per_pixel=0.5, scalef=0.25, quality="hires", spacing_um=10.0):
    """A 5 by 5 square lattice of spots spacing_um apart, in full-resolution pixels
    of microns_per_pixel, on an image scalef of full resolution."""
    xy_um = spacing_um * np.array([[x, y] for x in range(5) for y in range(5)], float)
    fullres = xy_um / microns_per_pixel
    image = fullres * (1.0 if quality == "fulres" else scalef)
    adata = AnnData(np.zeros((len(xy_um), 1)))
    adata.obsm["spatial"] = fullres
    adata.obs["imagecol"], adata.obs["imagerow"] = image.T
    scalefactors = {"microns_per_pixel": microns_per_pixel}
    if quality != "fulres":
        scalefactors[f"tissue_{quality}_scalef"] = scalef
    adata.uns["spatial"] = {
        "lib": {"use_quality": quality, "scalefactors": scalefactors}
    }
    return adata


class TestMicronsToPixels(unittest.TestCase):
    def test_value(self):
        """0.5 µm per full-resolution pixel on a quarter-size image is 2 µm per
        image pixel."""
        adata = _spots(microns_per_pixel=0.5, scalef=0.25)
        self.assertAlmostEqual(st.utils.microns_to_pixels(adata, 100), 50)
        self.assertAlmostEqual(st.utils.pixels_to_microns(adata, 50), 100)

    def test_full_resolution(self):
        adata = _spots(microns_per_pixel=0.5, quality="fulres")
        self.assertAlmostEqual(st.utils.microns_to_pixels(adata, 100), 200)

    def test_matches_imagecol(self):
        """The spots are 10 µm apart, so 10 µm is their spacing in imagecol."""
        for microns_per_pixel, scalef in ((0.5, 0.25), (0.2125, 0.1), (0.2738, 1.0)):
            with self.subTest(microns_per_pixel=microns_per_pixel, scalef=scalef):
                adata = _spots(microns_per_pixel, scalef)
                spacing = np.diff(np.unique(adata.obs["imagecol"]))
                np.testing.assert_allclose(
                    spacing, st.utils.microns_to_pixels(adata, 10.0)
                )

    def test_cci_neighbours(self):
        """A distance in microns finds the same neighbours at any pixel size: just
        over the spacing gives the 4 adjacent spots, and over the diagonal 8."""
        for microns_per_pixel, scalef in ((0.5, 0.25), (0.2125, 0.1)):
            adata = _spots(microns_per_pixel, scalef, spacing_um=10.0)
            centre = 12  # the middle of the 5 by 5 lattice
            for distance_um, n_expected in ((10.5, 4), (14.5, 8)):
                with self.subTest(
                    microns_per_pixel=microns_per_pixel, distance_um=distance_um
                ):
                    distance = st.utils.microns_to_pixels(adata, distance_um)
                    neighbours = calc_neighbours(adata, distance, verbose=False)
                    self.assertEqual(len(neighbours[centre]), n_expected)

    def test_arrays_and_round_trip(self):
        adata = _spots(microns_per_pixel=0.2125, scalef=0.1)
        microns = np.array([0.0, 10.0, 52.0, 270.0])
        np.testing.assert_allclose(
            st.utils.pixels_to_microns(
                adata, st.utils.microns_to_pixels(adata, microns)
            ),
            microns,
        )

    def test_microns_per_pixel(self):
        """The argument overrides the scale factor, and is needed without one."""
        adata = _spots(microns_per_pixel=0.5, scalef=0.25)
        self.assertAlmostEqual(
            st.utils.microns_to_pixels(adata, 100, microns_per_pixel=1.0), 25
        )
        del adata.uns["spatial"]["lib"]["scalefactors"]["microns_per_pixel"]
        with self.assertRaisesRegex(ValueError, "microns_per_pixel"):
            st.utils.microns_to_pixels(adata, 100)
        self.assertAlmostEqual(
            st.utils.microns_to_pixels(adata, 100, microns_per_pixel=0.5), 50
        )


if __name__ == "__main__":
    unittest.main()
