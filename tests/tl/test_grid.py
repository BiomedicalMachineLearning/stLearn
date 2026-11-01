"""Tests for st.tl.cci.grid and st.tl.cci.create_grid."""

import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stlearn as st
from stlearn.classes import Spatial
from tests.utils import read_test_data


class TestGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        adata = read_test_data()
        st.em.run_pca(adata)
        st.pp.neighbors(adata)
        st.tl.clustering.leiden(adata, resolution=0.6)
        cls.adata = adata
        cls.grid = st.tl.cci.grid(
            adata, n_row=6, n_col=6, use_label="leiden", verbose=False
        )

    def test_full_resolution_coordinates(self):
        """obsm["spatial"] is in full-resolution units, like the cells', so the
        image scale factor maps it onto imagecol/imagerow."""
        scalef = self.adata.uns["spatial"]["V1_Breast_Cancer_Block_A_Section_1"][
            "scalefactors"
        ]["tissue_hires_scalef"]
        np.testing.assert_allclose(
            self.grid.obsm["spatial"] * scalef,
            self.grid.obs[["imagecol", "imagerow"]],
        )
        cells = np.asarray(self.adata.obsm["spatial"], dtype=float)
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        self.assertTrue((self.grid.obsm["spatial"] >= lo - 1 / scalef).all())
        self.assertTrue((self.grid.obsm["spatial"] <= hi + 1 / scalef).all())

    def test_plotted_at_grid_centres(self):
        """The plots draw obsm["spatial"] times the scale factor."""
        np.testing.assert_allclose(
            Spatial(self.grid).imagecol, self.grid.obs["imagecol"]
        )
        st.pl.cluster_plot(self.grid, use_label="leiden", show_plot=False)
        offsets = np.vstack(
            [c.get_offsets() for c in plt.gca().collections if len(c.get_offsets())]
        )
        plt.close("all")
        expected = self.grid.obs[["imagecol", "imagerow"]].to_numpy()
        self.assertEqual(len(offsets), len(expected))
        np.testing.assert_allclose(np.sort(offsets, axis=0), np.sort(expected, axis=0))


if __name__ == "__main__":
    unittest.main()
