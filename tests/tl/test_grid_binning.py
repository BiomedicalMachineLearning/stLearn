"""Tests for st.tl.cci.grid and st.tl.cci.create_grid."""

import unittest

import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

matplotlib.use("Agg")

import stlearn as st


def _lattice(sparse):
    """Cells on integer coordinates, many of them on the grid's bin edges."""
    rng = np.random.default_rng(0)
    xy = np.array([[x, y] for x in range(0, 101, 5) for y in range(0, 81, 4)], float)
    X = rng.poisson(2.0, size=(len(xy), 3)).astype(np.float32)
    adata = AnnData(scipy.sparse.csr_matrix(X) if sparse else X)
    adata.obs_names = [f"cell{i}" for i in range(len(xy))]
    adata.obs["imagecol"], adata.obs["imagerow"] = xy[:, 0], xy[:, 1]
    adata.obs["cell_type"] = pd.Categorical(rng.choice(["A", "B", "C"], len(xy)))
    adata.obsm["spatial"] = xy
    adata.uns["spatial"] = {
        "lib": {"use_quality": "hires", "scalefactors": {"tissue_hires_scalef": 1.0}}
    }
    return adata, X


class TestGridBinning(unittest.TestCase):
    """Every cell is in one grid square, binned as numpy.histogram2d bins it."""

    def _check(self, sparse, n_row, n_col):
        adata, X = _lattice(sparse)
        grid = st.tl.cci.grid(
            adata, n_row=n_row, n_col=n_col, use_label="cell_type", verbose=False
        )
        xs, ys = adata.obs["imagecol"], adata.obs["imagerow"]

        def histogram(weights=None):
            counts, _, _ = np.histogram2d(xs, ys, [n_col, n_row], weights=weights)
            return counts.ravel()  # square n = column * n_row + row

        counts = histogram()
        occupied = np.flatnonzero(counts > 0)
        self.assertEqual(grid.obs["n_cells"].sum(), adata.n_obs)
        np.testing.assert_array_equal(grid.obs["n_cells"], counts[occupied])
        np.testing.assert_array_equal(grid.obs_names, [f"grid_{n}" for n in occupied])
        for gene in range(X.shape[1]):
            np.testing.assert_allclose(
                grid.X[:, gene], histogram(X[:, gene])[occupied], err_msg=gene
            )
        proportions = grid.uns["cell_type"]
        for cell_type in proportions.columns:
            is_type = (adata.obs["cell_type"] == cell_type).to_numpy(float)
            np.testing.assert_allclose(
                proportions[cell_type],
                histogram(is_type)[occupied] / counts[occupied],
                err_msg=cell_type,
            )

        # Squares are centred between their edges.
        _, xedges, yedges = np.histogram2d(xs, ys, [n_col, n_row])
        column, row = np.divmod(occupied, n_row)
        np.testing.assert_allclose(
            grid.obs["imagecol"], (xedges[column] + xedges[column + 1]) / 2
        )
        np.testing.assert_allclose(
            grid.obs["imagerow"], (yedges[row] + yedges[row + 1]) / 2
        )

    def test_sparse(self):
        for n_row, n_col in ((4, 5), (5, 5), (10, 4)):
            with self.subTest(n_row=n_row, n_col=n_col):
                self._check(True, n_row, n_col)

    def test_dense(self):
        self._check(False, 5, 4)


if __name__ == "__main__":
    unittest.main()
