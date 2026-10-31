"""Tests for reading tissue images as uint8: as_uint8_image, read_image, and the
readers that use them."""

import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd
import PIL.Image
import tifffile
from anndata import AnnData

import stlearn as st
from stlearn.utils import as_uint8_image
from tests.io.test_microns_per_pixel import VISIUM, write_spaceranger

# An 8 x 8 greyscale ramp over the 16-bit range, and the same ramp as uint8.
RAMP = np.arange(64).reshape(8, 8) * 1000
RAMP_UINT8 = np.rint(RAMP * 255 / 65535).astype(np.uint8)


def write_int32(path: pathlib.Path) -> None:
    """RAMP as a 32-bit integer image, which PIL opens in mode "I", as some
    Pillow versions open 16-bit PNGs. PIL reads the format from the file's
    contents, so this can be written where a reader expects a PNG."""
    tifffile.imwrite(path, RAMP.astype(np.int32))
    with PIL.Image.open(path) as image:
        assert image.mode == "I"


class TestImages(unittest.TestCase):
    def setUp(self):
        """A Space Ranger output directory, also used to write other images in."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = pathlib.Path(tmp.name)
        write_spaceranger(self.path, VISIUM)
        self.hires = self.path / "spatial" / "tissue_hires_image.png"

    def assertRamp(self, image):
        """image is RAMP as uint8: its full range kept, not clipped to 0-255."""
        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image, RAMP_UINT8)

    def test_as_uint8_image_signed_integers(self):
        """Signed integers, as PIL opens some 16-bit images, are scaled as uint8 if
        none is above 255, otherwise as uint16"""
        for dtype in (np.int32, np.int64):
            with self.subTest(dtype=dtype.__name__):
                self.assertRamp(as_uint8_image(RAMP.astype(dtype)))
        np.testing.assert_array_equal(
            as_uint8_image(np.array([-1, 0, 65535, 70000])), [0, 0, 255, 255]
        )
        # 8-bit values, as numpy's default int64 holds them, stay 8-bit.
        np.testing.assert_array_equal(
            as_uint8_image(np.array([0, 128, 255])), [0, 128, 255]
        )

    def test_as_uint8_image_other_dtypes(self):
        self.assertRamp(as_uint8_image(RAMP.astype(np.uint16)))
        np.testing.assert_array_equal(
            as_uint8_image(np.array([0.0, 0.5, 1.0, 2.0])), [0, 128, 255, 255]
        )
        np.testing.assert_array_equal(as_uint8_image(np.array([True, False])), [255, 0])
        with self.assertRaises(TypeError):
            as_uint8_image(np.zeros(4, dtype=complex))

    def test_read_image_16_and_32_bit_greyscale(self):
        PIL.Image.fromarray(RAMP.astype(np.uint16)).save(self.path / "16.png")
        write_int32(self.path / "32.tif")
        for name in ("16.png", "32.tif"):
            with self.subTest(name):
                self.assertRamp(st.utils.read_image(self.path / name))

    def test_read_image_colour_modes(self):
        """RGB stays RGB; palette and grey-with-alpha images become RGBA."""
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        # In PIL's web palette, so making a palette image doesn't change them.
        rgb[..., 0], rgb[..., 2] = 204, 51
        PIL.Image.fromarray(rgb).save(self.path / "rgb.png")
        PIL.Image.fromarray(rgb).convert("P").save(self.path / "palette.png")
        PIL.Image.fromarray(np.full((4, 4), 90, np.uint8)).convert("LA").save(
            self.path / "la.png"
        )
        np.testing.assert_array_equal(st.utils.read_image(self.path / "rgb.png"), rgb)
        np.testing.assert_array_equal(
            st.utils.read_image(self.path / "palette.png")[..., :3], rgb
        )
        np.testing.assert_array_equal(
            st.utils.read_image(self.path / "la.png"),
            np.full((4, 4, 4), [90, 90, 90, 255]),
        )

    def test_read_10x(self):
        """The Space Ranger image, and a full-resolution image given by path."""
        write_int32(self.hires)
        adata = st.io.read_10x(self.path)
        self.assertRamp(adata.uns["spatial"]["lib"]["images"]["hires"])

        tifffile.imwrite(self.path / "fullres.tif", RAMP.astype(np.uint16))
        adata = st.io.read_10x(
            self.path, quality="fulres", image_path=self.path / "fullres.tif"
        )
        self.assertRamp(adata.uns["spatial"]["lib"]["images"]["fulres"])

    def test_read_10x_errors(self):
        """Each error says what's wrong: why the file couldn't be read, or that
        it's missing."""
        self.hires.write_bytes(b"not an image")
        with self.assertRaisesRegex(
            OSError, "Could not read .*tissue_hires_image"
        ) as e:
            st.io.read_10x(self.path)
        self.assertIn("cannot identify image file", str(e.exception))

        self.hires.unlink()
        with self.assertRaisesRegex(OSError, "Could not find .*tissue_hires_image"):
            st.io.read_10x(self.path)

    def test_add_image(self):
        """add.image keeps a 32-bit image's range, where matplotlib's imread
        clipped it to 0-255, a white image."""
        write_int32(self.path / "image.tif")
        adata = AnnData(np.ones((2, 1)))
        adata.obs = pd.DataFrame(
            {"imagecol": [1.0, 2.0], "imagerow": [3.0, 4.0]}, index=["a", "b"]
        )
        st.add.image(adata, self.path / "image.tif", "lib")
        self.assertRamp(adata.uns["spatial"]["lib"]["images"]["hires"])

    def test_add_image_errors(self):
        """Each error says what's wrong: no file, or why it couldn't be read."""
        adata = AnnData(np.ones((1, 1)))
        with self.assertRaisesRegex(ValueError, "No image file"):
            st.add.image(adata, self.path / "missing.png", "lib")
        (self.path / "bad.png").write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "cannot identify image file"):
            st.add.image(adata, self.path / "bad.png", "lib")


if __name__ == "__main__":
    unittest.main()
