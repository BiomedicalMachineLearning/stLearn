#!/usr/bin/env python

"""Tests for perform_spot_testing: per-spot p-values against each LR's background."""

import unittest

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

from stlearn.tl.cci._results import to_dense
from stlearn.tl.cci.neighbours import calc_neighbours
from stlearn.tl.cci.permutation import gene_quantiles, perform_spot_testing
from stlearn.tl.cci.scoring import get_lrs_scores

LRS = np.array(["Lig0_Rec0", "Lig1_Rec1", "Lig2_Rec2"])
N_PAIRS = 100


def _make_adata(n_side=10, n_extra_genes=60, seed=0):
    """Sparse counts on a grid. Lig0 and Rec0 are high together in one corner, so
    some spots are significant; zeros elsewhere leave some spots unscored."""
    rng = np.random.default_rng(seed)
    n_spots = n_side * n_side
    lr_genes = ["Lig0", "Rec0", "Lig1", "Rec1", "Lig2", "Rec2"]
    var_names = lr_genes + [f"gene{i}" for i in range(n_extra_genes)]

    X = rng.poisson(0.3, size=(n_spots, len(var_names))).astype(np.float32)
    rows, cols = np.divmod(np.arange(n_spots), n_side)
    corner = (rows < 4) & (cols < 4)
    X[corner, :2] = 6

    obs = pd.DataFrame(
        {"imagerow": rows.astype(float), "imagecol": cols.astype(float)},
        index=[f"spot{i}" for i in range(n_spots)],
    )
    var = pd.DataFrame(index=var_names)
    return AnnData(X=scipy.sparse.csr_matrix(X), obs=obs, var=var)


class TestPermutation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adata = _make_adata()
        neighbours = calc_neighbours(cls.adata, distance=1.5, verbose=False)
        het_vals = np.ones(len(cls.adata))
        # Columns of scores, and of everything perform_spot_testing returns,
        # follow lrs: the order get_lrs_scores gives them.
        cls.scores, cls.lrs = get_lrs_scores(cls.adata, LRS, neighbours, het_vals, 0)
        cls.results = perform_spot_testing(
            cls.adata,
            cls.scores,
            cls.lrs,
            N_PAIRS,
            neighbours,
            het_vals,
            0,
            verbose=False,
            save_bg=True,
        )

    def test_pvals_from_background(self):
        """A scored spot's p-value is the share of its background scores at or
        above its own score, at least 1 / n_pairs. Unscored spots keep 1: they
        are not in the sparse p-values, whose entries are the scored spots."""
        sparse_pvals = self.results.pvals
        self.assertTrue(scipy.sparse.issparse(sparse_pvals))
        self.assertEqual(sparse_pvals.shape, self.scores.shape)
        np.testing.assert_array_equal(sparse_pvals.toarray() != 0, self.scores > 0)
        pvals = to_dense(sparse_pvals, 1.0)
        for j, lr in enumerate(self.lrs):
            background = self.results.backgrounds[lr]
            spots = self.results.bg_spot_indices[lr]
            np.testing.assert_array_equal(spots, np.flatnonzero(self.scores[:, j] > 0))
            self.assertTrue(0 < len(spots) < len(self.adata), msg=lr)
            self.assertEqual(background.shape, (len(spots), N_PAIRS), msg=lr)

            for spot_i, spot in enumerate(spots):
                n_greater = np.count_nonzero(background[spot_i] >= self.scores[spot, j])
                expected = max(n_greater, 1) / N_PAIRS
                self.assertEqual(pvals[spot, j], expected, msg=f"{lr} {spot}")

            unscored = np.setdiff1d(np.arange(len(self.adata)), spots)
            np.testing.assert_array_equal(pvals[unscored, j], 1)

    def test_some_spots_significant(self):
        """The co-expressed corner gives Lig0_Rec0 p-values at the floor."""
        j = list(self.lrs).index("Lig0_Rec0")
        self.assertEqual(to_dense(self.results.pvals, 1.0)[:, j].min(), 1 / N_PAIRS)

    def test_lr_features_cover_every_lr(self):
        self.assertCountEqual(self.results.lr_features.index, self.lrs)

    def test_leaves_adata_alone(self):
        """Results are returned; st.tl.cci.run is what stores them."""
        for key in ("lr_summary", "lrfeatures", "lrs_to_bg", "lr_spot_indices"):
            self.assertNotIn(key, self.adata.uns)
        self.assertNotIn("p_vals", self.adata.obsm)

    def test_no_backgrounds_unless_asked(self):
        adata = _make_adata()
        neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
        het_vals = np.ones(len(adata))
        scores, lrs = get_lrs_scores(adata, LRS, neighbours, het_vals, 0)
        results = perform_spot_testing(
            adata, scores, lrs, N_PAIRS, neighbours, het_vals, 0, verbose=False
        )
        self.assertIsNone(results.backgrounds)
        self.assertIsNone(results.bg_spot_indices)
        np.testing.assert_array_equal(
            results.pvals.toarray(), self.results.pvals.toarray()
        )

    def test_sparse_scores_test_the_same(self):
        """The sparse scores run passes in (the scores above 0) give the same
        p-values as the dense scores, entry for entry."""
        adata = _make_adata()
        neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
        het_vals = np.ones(len(adata))
        scores, lrs = get_lrs_scores(
            adata, LRS, neighbours, het_vals, 0, sparse_result=True
        )
        self.assertTrue(scipy.sparse.issparse(scores))
        results = perform_spot_testing(
            adata, scores, lrs, N_PAIRS, neighbours, het_vals, 0, verbose=False
        )
        np.testing.assert_array_equal(results.pvals.indptr, scores.indptr)
        np.testing.assert_array_equal(results.pvals.indices, scores.indices)
        np.testing.assert_array_equal(
            results.pvals.toarray(), self.results.pvals.toarray()
        )

    def test_homophilic_lr_is_tested(self):
        """A gene paired with itself gets a background like any LR: random pairs
        of two different genes similar to it, so the random pairs never pair a
        gene with itself (as before), but the LR itself is tested."""
        adata = _make_adata()
        neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
        het_vals = np.ones(len(adata))
        scores, lrs = get_lrs_scores(
            adata, np.array(["Lig0_Lig0"]), neighbours, het_vals, 0
        )
        self.assertEqual(list(lrs), ["Lig0_Lig0"])

        results = perform_spot_testing(
            adata,
            scores,
            lrs,
            N_PAIRS,
            neighbours,
            het_vals,
            0,
            verbose=False,
            save_bg=True,
        )

        spots = np.flatnonzero(scores[:, 0] > 0)
        pvals = to_dense(results.pvals, 1.0)
        self.assertEqual(results.backgrounds["Lig0_Lig0"].shape, (len(spots), N_PAIRS))
        self.assertTrue((pvals[spots, 0] < 1).any())
        self.assertEqual(pvals[:, 0].min(), 1 / N_PAIRS)  # the corner

    def test_unscored_lr_is_left_untested(self):
        """An LR with no scored spot has nothing to test: its p-values stay 1,
        its background is empty, and the other LRs are tested as before."""
        adata = _make_adata()
        neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
        het_vals = np.ones(len(adata))
        scores, lrs = get_lrs_scores(adata, LRS, neighbours, het_vals, 0)
        scores[:, 1] = 0  # as if Lig1_Rec1 never scored
        results = perform_spot_testing(
            adata,
            scores,
            lrs,
            N_PAIRS,
            neighbours,
            het_vals,
            0,
            verbose=False,
            save_bg=True,
        )
        pvals = to_dense(results.pvals, 1.0)
        np.testing.assert_array_equal(pvals[:, 1], 1)
        self.assertEqual(results.pvals[:, 1].nnz, 0)  # no entries: nothing tested
        self.assertEqual(results.backgrounds[lrs[1]].shape, (0, N_PAIRS))
        self.assertEqual(results.bg_spot_indices[lrs[1]].size, 0)
        np.testing.assert_array_equal(
            pvals[:, [0, 2]], to_dense(self.results.pvals, 1.0)[:, [0, 2]]
        )

    def test_matches_quantiles_per_gene(self):
        """Quantiles computed in chunks equal np.quantile gene by gene, for chunk
        sizes that split the genes unevenly, one at a time, or all at once."""
        adata = _make_adata()
        genes = adata.var_names.to_numpy()
        quantiles = np.array([0.5, 0.75, 0.9, 0.99, 1])
        expected = np.apply_along_axis(
            np.quantile, 0, adata.X.toarray(), q=quantiles, method="nearest"
        )
        for chunk_size in (1, 7, len(genes), 1000):
            np.testing.assert_array_equal(
                gene_quantiles(adata, genes, quantiles, chunk_size),
                expected,
                err_msg=f"{chunk_size=}",
            )
