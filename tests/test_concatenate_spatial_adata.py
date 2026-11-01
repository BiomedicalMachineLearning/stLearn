"""Tests for concatenate_spatial_adata."""

import unittest

import numpy as np
from anndata import AnnData

import stlearn as st
from stlearn.utils import as_uint8_image

RED, BLUE, GREEN, YELLOW = (1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0)
# Mid-tones, 51 and 153 of 255, exact in uint8 and uint16 (x 257).
GREY, TEAL = (0.2, 0.2, 0.2), (0.0, 0.6, 0.6)
WHITE = 255  # The tiled image is uint8, as stLearn stores images.


def _u8(colour):
    """A colour in [0, 1] as the tiled image's uint8."""
    return np.array(colour) * 255


def _sample(
    name, shape, blocks, scalef, microns_per_pixel=None, n=60, seed=0, dtype=None
):
    """Cells on an image of colour blocks, split in half along `blocks`' axis; the
    image is float32 in [0, 1], or an integer dtype over its range up to 65535."""
    rng = np.random.default_rng(seed)
    height, width = shape
    image = np.empty((height, width, 3), dtype=np.float32)
    axis, (first, second) = blocks
    half = (height if axis == "rows" else width) // 2
    if axis == "rows":
        image[:half], image[half:] = first, second
    else:
        image[:, :half], image[:, half:] = first, second
    if dtype is not None:
        top = min(np.iinfo(dtype).max, 65535)
        image = (image * top).astype(dtype)
    # Image pixels away from the edges and the boundary between the blocks.
    image_xy = rng.uniform([10, 10], [width - 10, height - 10], size=(n, 2))
    along = image_xy[:, 1] if axis == "rows" else image_xy[:, 0]
    image_xy = image_xy[np.abs(along - half) > 10]

    adata = AnnData(np.ones((len(image_xy), 2), dtype=np.float32))
    adata.obs_names = [f"{name}{i}" for i in range(len(image_xy))]
    adata.obsm["spatial"] = image_xy / scalef
    adata.obs["imagecol"], adata.obs["imagerow"] = image_xy[:, 0], image_xy[:, 1]
    scalefactors = {"tissue_hires_scalef": scalef, "spot_diameter_fullres": 20.0}
    if microns_per_pixel is not None:
        scalefactors["microns_per_pixel"] = microns_per_pixel
    adata.uns["spatial"] = {
        name: {
            "images": {"hires": image},
            "use_quality": "hires",
            "scalefactors": scalefactors,
        }
    }
    return adata


def _colours(adata):
    """The tiled image's colour under each cell."""
    image = adata.uns["spatial"]["A"]["images"]["hires"]
    xy = adata.obs[["imagecol", "imagerow"]].to_numpy().astype(int)
    return image[xy[:, 1], xy[:, 0]]


class TestConcatenateSpatialAdata(unittest.TestCase):
    def setUp(self):
        # A is square, B is wider than tall, with different scale factors.
        self.a = _sample("A", (400, 400), ("columns", (RED, BLUE)), 0.5)
        self.b = _sample("B", (200, 300), ("rows", (GREEN, YELLOW)), 0.25, seed=1)

    def _expected_colours(self, adata, image_xy):
        image = adata.uns["spatial"][next(iter(adata.uns["spatial"]))]["images"][
            "hires"
        ]
        xy = image_xy.astype(int)
        return as_uint8_image(image)[xy[:, 1], xy[:, 0]]

    def test_tiles_keep_aspect_and_cells_stay_on_their_tissue(self):
        concat = st.concatenate_spatial_adata(
            [self.a, self.b], ncols=2, fixed_size=(200, 200)
        )
        image = concat.uns["spatial"]["A"]["images"]["hires"]
        self.assertEqual(image.shape, (200, 400, 3))
        # B is 200 x 300, scaled by 2/3 to 133 x 200: white below it.
        np.testing.assert_allclose(image[150:, 200:], WHITE)
        np.testing.assert_allclose(image[60, 300], _u8(GREEN))

        a = concat[concat.obs["batch"] == "0"]
        b = concat[concat.obs["batch"] == "1"]
        a_xy = self.a.obsm["spatial"] * 0.5
        b_xy = self.b.obsm["spatial"] * 0.25
        np.testing.assert_allclose(a.obs[["imagecol", "imagerow"]], a_xy * 0.5)
        np.testing.assert_allclose(
            b.obs[["imagecol", "imagerow"]], b_xy * 2 / 3 + [200, 0]
        )
        np.testing.assert_allclose(_colours(a), self._expected_colours(self.a, a_xy))
        np.testing.assert_allclose(_colours(b), self._expected_colours(self.b, b_xy))

        # obsm["spatial"] is in full-resolution units of A's scale factor.
        scalefactors = concat.uns["spatial"]["A"]["scalefactors"]
        self.assertEqual(scalefactors["tissue_hires_scalef"], 0.5)
        np.testing.assert_allclose(
            concat.obsm["spatial"] * 0.5, concat.obs[["imagecol", "imagerow"]]
        )
        self.assertAlmostEqual(scalefactors["spot_diameter_fullres"], 20.0 * 0.5)
        self.assertNotIn("microns_per_pixel", scalefactors)

    def test_samples_share_a_micron_scale(self):
        # 4 microns per image pixel each: A is 1600 um across, B 1200 x 800 um.
        a = _sample("A", (400, 400), ("columns", (RED, BLUE)), 0.5, 2.0)
        b = _sample("B", (200, 300), ("rows", (GREEN, YELLOW)), 0.25, 1.0, seed=1)
        concat = st.concatenate_spatial_adata([a, b], ncols=2, fixed_size=(200, 200))

        scalefactors = concat.uns["spatial"]["A"]["scalefactors"]
        # 8 um per tiled pixel, so A fills its tile; that's 4 um per obsm unit.
        self.assertAlmostEqual(scalefactors["microns_per_pixel"], 4.0)
        image = concat.uns["spatial"]["A"]["images"]["hires"]
        np.testing.assert_allclose(image[20, 300], _u8(GREEN))
        np.testing.assert_allclose(image[80, 300], _u8(YELLOW))
        np.testing.assert_allclose(image[120:, 200:], WHITE)  # B is 100 x 150
        np.testing.assert_allclose(image[:, 350:], WHITE)

        for sample, batch in ((a, "0"), (b, "1")):
            part = concat[concat.obs["batch"] == batch]
            image_xy = (
                sample.obsm["spatial"]
                * sample.uns["spatial"][next(iter(sample.uns["spatial"]))][
                    "scalefactors"
                ]["tissue_hires_scalef"]
            )
            np.testing.assert_allclose(
                _colours(part), self._expected_colours(sample, image_xy)
            )
            # Distances in microns are kept.
            microns = sample.uns["spatial"][next(iter(sample.uns["spatial"]))][
                "scalefactors"
            ]["microns_per_pixel"]
            before = np.linalg.norm(np.diff(sample.obsm["spatial"][:2], axis=0))
            after = np.linalg.norm(np.diff(part.obsm["spatial"][:2], axis=0))
            self.assertAlmostEqual(before * microns, after * 4.0, places=6)

    def test_blank_tile_and_mixed_scales(self):
        c = _sample("C", (100, 100), ("rows", (BLUE, RED)), 1.0, 1.0, seed=2)
        with self.assertWarnsRegex(UserWarning, "microns_per_pixel"):
            concat = st.concatenate_spatial_adata(
                [c, self.a, self.b], ncols=2, fixed_size=(100, 100)
            )
        image = concat.uns["spatial"]["C"]["images"]["hires"]
        self.assertEqual(image.shape, (200, 200, 3))
        np.testing.assert_allclose(image[100:, 100:], WHITE)  # no fourth sample
        self.assertNotIn(
            "microns_per_pixel", concat.uns["spatial"]["C"]["scalefactors"]
        )
        self.assertEqual(list(concat.obs["batch"].cat.categories), ["0", "1", "2"])

    def test_stored_as_uint8(self):
        """The tiled image is uint8, as stLearn stores images, from images of any
        dtype; 16-bit images, and those PIL opens as 32-bit, keep their colours,
        mid-tones included."""
        for dtype in (np.float32, np.uint8, np.uint16, np.int32):
            with self.subTest(dtype=dtype.__name__):
                b = _sample(
                    "B",
                    (200, 300),
                    ("rows", (GREY, TEAL)),
                    0.25,
                    seed=1,
                    dtype=None if dtype is np.float32 else dtype,
                )
                concat = st.concatenate_spatial_adata(
                    [self.a, b], ncols=2, fixed_size=(200, 200)
                )
                image = concat.uns["spatial"]["A"]["images"]["hires"]
                self.assertEqual(image.dtype, np.uint8)
                np.testing.assert_allclose(image[20, 300], _u8(GREY))
                np.testing.assert_allclose(image[120, 300], _u8(TEAL))

    def test_too_many_columns(self):
        with self.assertRaisesRegex(ValueError, "out of bound"):
            st.concatenate_spatial_adata([self.a, self.b], ncols=3)


if __name__ == "__main__":
    unittest.main()
