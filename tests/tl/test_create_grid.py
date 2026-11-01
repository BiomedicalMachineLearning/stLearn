"""Tests for st.tl.cci.grid and st.tl.cci.create_grid."""

import unittest

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

import stlearn as st


def _section(microns_per_pixel=0.5, scalef=0.5):
    """Cells every 10 µm over a 1000 by 1660 µm section, in full-resolution pixels
    of microns_per_pixel, on an image scalef of full resolution."""
    rng = np.random.default_rng(0)
    xy_um = np.array(
        [[x, y] for x in range(0, 1001, 10) for y in range(0, 1661, 10)], float
    )
    adata = AnnData(scipy.sparse.csr_matrix(rng.poisson(1.0, (len(xy_um), 3))))
    adata.obs_names = [f"cell{i}" for i in range(len(xy_um))]
    fullres = xy_um / microns_per_pixel
    adata.obsm["spatial"] = fullres
    adata.obs["imagecol"], adata.obs["imagerow"] = (fullres * scalef).T
    adata.obs["cell_type"] = pd.Categorical(rng.choice(["A", "B", "C"], len(xy_um)))
    adata.uns["spatial"] = {
        "lib": {
            "use_quality": "hires",
            "scalefactors": {
                "tissue_hires_scalef": scalef,
                "microns_per_pixel": microns_per_pixel,
            },
        }
    }
    return adata


class TestCreateGrid(unittest.TestCase):
    """create_grid picks the rows and columns from the tissue's size in microns."""

    @staticmethod
    def _shape(grid):
        return len(grid.uns["grid_yedges"]) - 1, len(grid.uns["grid_xedges"]) - 1

    def test_square_bins_of_bin_um(self):
        """1000 by 1660 µm in 52 µm squares is 32 rows by 19 columns."""
        grid = st.tl.cci.create_grid(_section(), bin_um=52, verbose=False)
        self.assertEqual(self._shape(grid), (32, 19))
        np.testing.assert_allclose(grid.uns["grid_bin_um"], [1000 / 19, 1660 / 32])

    def test_same_as_grid(self):
        """It is grid() with those rows and columns."""
        adata = _section()
        sized = st.tl.cci.create_grid(
            adata, bin_um=52, use_label="cell_type", verbose=False
        )
        counted = st.tl.cci.grid(
            adata, n_row=32, n_col=19, use_label="cell_type", verbose=False
        )
        np.testing.assert_allclose(sized.X, counted.X)
        pd.testing.assert_frame_equal(sized.obs, counted.obs)
        pd.testing.assert_frame_equal(sized.uns["cell_type"], counted.uns["cell_type"])
        np.testing.assert_allclose(sized.obsm["spatial"], counted.obsm["spatial"])

    def test_pixel_size_does_not_change_grid(self):
        """The squares are sized in microns, whatever the pixels and image scale."""
        for microns_per_pixel, scalef in ((0.5, 0.5), (0.2125, 0.1), (0.2738, 0.3)):
            with self.subTest(microns_per_pixel=microns_per_pixel, scalef=scalef):
                grid = st.tl.cci.create_grid(
                    _section(microns_per_pixel, scalef), bin_um=52, verbose=False
                )
                self.assertEqual(self._shape(grid), (32, 19))
                np.testing.assert_allclose(grid.uns["grid_bin_um"], 52, rtol=0.02)

    def test_microns_per_pixel(self):
        """The argument overrides the scale factor, and is needed without one."""
        adata = _section()
        grid = st.tl.cci.create_grid(
            adata, bin_um=52, microns_per_pixel=1.0, verbose=False
        )
        self.assertEqual(self._shape(grid), (64, 38))  # 2000 by 3320 µm

        del adata.uns["spatial"]["lib"]["scalefactors"]["microns_per_pixel"]
        with self.assertRaisesRegex(ValueError, "microns_per_pixel"):
            st.tl.cci.create_grid(adata, verbose=False)
        grid = st.tl.cci.create_grid(
            adata, bin_um=52, microns_per_pixel=0.5, verbose=False
        )
        self.assertEqual(self._shape(grid), (32, 19))

    def test_get_shape(self):
        """st.utils.get_shape is the (columns, rows) create_grid grids into, as
        Python ints, honouring microns_per_pixel like create_grid does."""
        adata = _section()
        shape = st.utils.get_shape(adata, 52)
        self.assertEqual(shape, (19, 32))
        self.assertTrue(all(type(n) is int for n in shape))
        self.assertEqual(st.utils.get_shape(adata, 52, microns_per_pixel=1.0), (38, 64))

    def test_at_least_one_square(self):
        adata = _section()
        grid = st.tl.cci.create_grid(adata, bin_um=5000, verbose=False)
        self.assertEqual(self._shape(grid), (1, 1))
        self.assertEqual(grid.obs["n_cells"].sum(), adata.n_obs)

    def test_bin_um_must_be_positive(self):
        for bin_um in (0, -52, float("nan")):
            with self.subTest(bin_um=bin_um), self.assertRaises(ValueError):
                st.tl.cci.create_grid(_section(), bin_um=bin_um, verbose=False)


if __name__ == "__main__":
    unittest.main()
