#!/usr/bin/env python

import unittest

import numpy as np

import stlearn as st
from stlearn.tl.cci.neighbours import calc_neighbours, get_neighbourhoods
from tests.utils import CCI_LRS, cci_table


class TestNeighbourhoods(unittest.TestCase):
    """get_neighbourhoods reads the neighbours st.tl.cci.run stores in obsp."""

    @classmethod
    def setUpClass(cls):
        cls.distance = 1.5
        adata = cci_table()
        st.tl.cci.run(
            adata,
            CCI_LRS,
            min_spots=2,
            distance=cls.distance,
            n_pairs=100,
            verbose=False,
        )
        cls.adata = adata
        graph = calc_neighbours(adata, cls.distance, verbose=False)
        cls.found = np.split(graph.indices, graph.indptr[1:-1])

    def test_matches_calc_neighbours(self):
        """Each spot's index, barcode and neighbours are those calc_neighbours
        finds, in its order."""
        bcs, indices = get_neighbourhoods(self.adata)
        names = self.adata.obs_names.to_numpy()
        self.assertEqual(len(bcs), self.adata.n_obs)
        self.assertEqual(len(indices), self.adata.n_obs)
        for i in range(self.adata.n_obs):
            self.assertEqual(indices[i][0], i)
            np.testing.assert_array_equal(indices[i][1], self.found[i])
            self.assertEqual(bcs[i][0], names[i])
            np.testing.assert_array_equal(bcs[i][1], names[self.found[i]])

    def test_subset_keeps_neighbours_still_in_it(self):
        """Subsetting slices obsp on both axes, so a spot keeps only the
        neighbours still in the subset, at their positions in it."""
        keep = np.arange(self.adata.n_obs) % 3 != 0
        subset = self.adata[keep].copy()
        bcs, indices = get_neighbourhoods(subset)
        names = self.adata.obs_names.to_numpy()
        self.assertEqual(len(bcs), subset.n_obs)
        for row, i in enumerate(np.flatnonzero(keep)):
            expected = names[[n for n in self.found[i] if keep[n]]]
            self.assertEqual(bcs[row][0], names[i])
            self.assertEqual(sorted(bcs[row][1]), sorted(expected))
            np.testing.assert_array_equal(
                subset.obs_names[indices[row][1]], bcs[row][1]
            )

    def test_without_neighbours(self):
        """A table without obsp["spot_neighbours"] says to run st.tl.cci.run."""
        adata = self.adata.copy()
        del adata.obsp["spot_neighbours"]
        with self.assertRaisesRegex(ValueError, r"st\.tl\.cci\.run"):
            get_neighbourhoods(adata)


if __name__ == "__main__":
    unittest.main()
