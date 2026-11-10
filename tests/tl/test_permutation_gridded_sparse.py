#!/usr/bin/env python

"""Tests for perform_spot_testing: per-spot p-values against each LR's background,
under both nulls ("codetected", the default, and "random", the original), on a
gridded sample of well-covered spots and on sparse single cells without a grid."""

import unittest

import numpy as np
import itertools
from anndata import AnnData
from stlearn.tl.cci.ligand_receptors import LRPairs
from stlearn.tl.cci._results import to_dense
from stlearn.tl.cci.neighbours import calc_neighbours
from stlearn.tl.cci.permutation import (
    empirical_pvals,
    perform_spot_testing,
    gene_quantiles,
)
from stlearn.tl.cci.scoring import get_lrs_scores
from tests.utils import spot_table

# Lig0 and Rec0 are high together in a patch; the other two pairs are noise.
LRS = LRPairs.from_names(["Lig0_Rec0", "Lig1_Rec1", "Lig2_Rec2"], species="human")
LR_GENES = ["Lig0", "Rec0", "Lig1", "Rec1", "Lig2", "Rec2"]
NULLS = ("codetected", "random")


def _test(adata, distance, n_pairs, lrs=LRS, sparse_result=False, **kwargs):
    """Scores `lrs` and tests them; kwargs go to perform_spot_testing."""
    neighbours = calc_neighbours(adata, distance=distance, verbose=False)
    het_vals = np.ones(len(adata))
    scores, lrs = get_lrs_scores(
        adata, lrs, neighbours, het_vals, 0, sparse_result=sparse_result
    )
    if "zero_lr" in kwargs:  # as if this LR never scored
        scores[:, kwargs.pop("zero_lr")] = 0
    results = perform_spot_testing(
        adata, scores, lrs, n_pairs, neighbours, het_vals, 0, verbose=False, **kwargs
    )
    return scores, lrs, results


def _gridded_adata(n_side=10, n_extra_genes=60, seed=0):
    """Spots on a grid, each covering many cells, as Visium or gridded data: most
    genes are detected at most spots, so most random pairs score at a spot. Lig0
    and Rec0 are high together in one corner, so some spots are significant; the
    opposite corner has no LR gene, which leaves its spots unscored."""
    rng = np.random.default_rng(seed)
    n_spots = n_side * n_side
    X = rng.poisson(2.0, size=(n_spots, len(LR_GENES) + n_extra_genes))
    rows, cols = np.divmod(np.arange(n_spots), n_side)
    X[(rows < 4) & (cols < 4), :2] = 12
    X[(rows >= 8) & (cols >= 8), : len(LR_GENES)] = 0
    return spot_table(X, rows, cols, LR_GENES)


def _sparse_cells_adata(n_cells=900, n_extra_genes=150, seed=0):
    """Single cells scattered at one per unit area, as Xenium or Visium HD bins
    without gridding: a gene is detected in about a cell in ten, so few random
    pairs score at a cell. Lig0 and Rec0 are high together in the cells of one
    corner; the other LR genes are as sparse as the rest."""
    rng = np.random.default_rng(seed)
    side = np.sqrt(n_cells)
    rows, cols = rng.uniform(0, side, size=(2, n_cells))
    X = rng.poisson(0.1, size=(n_cells, len(LR_GENES) + n_extra_genes))
    patch = (rows < side * 0.3) & (cols < side * 0.3)
    X[patch, :2] = rng.poisson(4.0, size=(patch.sum(), 2))
    return spot_table(X, rows, cols, LR_GENES)


class Sample:
    """A sample tested under each null, with its backgrounds saved."""

    def __init__(self, adata: AnnData, distance: float, n_pairs: int):
        self.adata = adata
        self.n_pairs = n_pairs
        # Columns of scores, and of everything perform_spot_testing returns,
        # follow the scored pairs get_lrs_scores gives: their names() order.
        self.results = {}
        for null in NULLS:
            self.scores, self.lrs, self.results[null] = _test(
                adata, distance, n_pairs, save_bg=True, null=null
            )
        self.pvals = {null: to_dense(r.pvals, 1.0) for null, r in self.results.items()}
        self.tested = self.scores > 0

    def column(self, lr: str) -> int:
        return list(self.lrs.names()).index(lr)

    def codetected_share(self) -> np.ndarray:
        """Per tested spot and LR, the share of its random pairs scoring above 0."""
        backgrounds = self.results["random"].backgrounds
        return np.concatenate([(bg > 0).mean(1) for bg in backgrounds.values()])

    def called(self, null: str, lr: str) -> float:
        """The share of an LR's tested spots with p < 0.05."""
        j = self.column(lr)
        return (self.pvals[null][self.tested[:, j], j] < 0.05).mean()


class TestGriddedAndSparse(unittest.TestCase):
    """Both nulls on two samples. Gridded: well-covered spots, where nearly every
    random pair is detected at a spot, so the nulls are nearly the same test.
    Sparse: single cells without a grid, where few are, so the nulls part."""

    @classmethod
    def setUpClass(cls):
        # distance: the 8 surrounding spots; about 6 neighbouring cells.
        cls.gridded = Sample(_gridded_adata(), distance=1.5, n_pairs=100)
        cls.sparse = Sample(_sparse_cells_adata(), distance=1.4, n_pairs=1000)
        cls.samples = {"gridded": cls.gridded, "sparse": cls.sparse}

    # On both samples

    def test_pvals_from_background(self):
        """Each scored spot gets a p-value from its own background, as
        empirical_pvals counts it for the null. Unscored spots are left out of
        the sparse p-values, so read as 1."""
        for (name, sample), null in itertools.product(self.samples.items(), NULLS):
            results = sample.results[null]
            with self.subTest(sample=name, null=null):
                self.assertTrue(scipy.sparse.issparse(results.pvals))
                np.testing.assert_array_equal(
                    results.pvals.toarray() != 0, sample.tested
                )
            for j, lr in enumerate(sample.lrs.names()):
                with self.subTest(sample=name, null=null, lr=lr):
                    spots = results.bg_spot_indices[lr]
                    background = results.backgrounds[lr]
                    np.testing.assert_array_equal(
                        spots, np.flatnonzero(sample.tested[:, j])
                    )
                    self.assertTrue(0 < len(spots) < len(sample.adata))  # some unscored
                    self.assertEqual(background.shape, (len(spots), sample.n_pairs))
                    np.testing.assert_array_equal(
                        sample.pvals[null][spots, j],
                        empirical_pvals(sample.scores[spots, j], background, null),
                    )

    def test_nulls_share_the_background(self):
        """The null chooses which background scores count, not the random pairs
        or their scores."""
        for name, sample in self.samples.items():
            codetected, random = sample.results["codetected"], sample.results["random"]
            for lr in sample.lrs.names():
                with self.subTest(sample=name, lr=lr):
                    np.testing.assert_array_equal(
                        codetected.backgrounds[lr], random.backgrounds[lr]
                    )
                    np.testing.assert_array_equal(
                        codetected.bg_spot_indices[lr], random.bg_spot_indices[lr]
                    )

    def test_co_expressed_patch_is_significant(self):
        """Lig0 and Rec0, high together in a patch, are called under either
        null; under "random" at its floor."""
        for name, sample in self.samples.items():
            with self.subTest(sample=name):
                j = sample.column("Lig0_Rec0")
                self.assertEqual(sample.pvals["random"][:, j].min(), 1 / sample.n_pairs)
                self.assertLess(sample.pvals["codetected"][:, j].min(), 0.05)

    # Where the samples differ

    def test_share_of_random_pairs_codetected(self):
        """Nearly every random pair is detected at a gridded spot; few are at a
        sparse cell."""
        self.assertGreater(np.median(self.gridded.codetected_share()), 0.9)
        self.assertLess(np.median(self.sparse.codetected_share()), 0.25)

    def test_nulls_agree_on_gridded_spots(self):
        """With few zeros in the background the two nulls' p-values are close."""
        pvals = self.gridded.pvals
        self.assertLess(np.abs(pvals["codetected"] - pvals["random"]).max(), 0.1)

    def test_unrelated_genes_are_called_at_chance(self):
        """Lig1_Rec1 and Lig2_Rec2 are pairs of unrelated genes, so should be
        called at about the 5% chance gives. "codetected" is, on either sample;
        "random" only on gridded spots. At a sparse cell most random pairs score
        0, so against all of them any score above 0 is significant."""
        for lr in ("Lig1_Rec1", "Lig2_Rec2"):
            with self.subTest(lr=lr):
                self.assertLess(self.gridded.called("codetected", lr), 0.15)
                self.assertLess(self.sparse.called("codetected", lr), 0.15)
                self.assertLess(self.gridded.called("random", lr), 0.15)
                self.assertGreater(self.sparse.called("random", lr), 0.3)

    def test_patch_is_called_more_than_the_rest(self):
        """On sparse cells, under "codetected", the co-expressed patch is still
        what Lig0_Rec0 is called in."""
        sample = self.sparse
        j = sample.column("Lig0_Rec0")
        obs = sample.adata.obs
        side = np.sqrt(len(obs))
        patch = ((obs["imagerow"] < side * 0.3) & (obs["imagecol"] < side * 0.3)).values
        tested = sample.tested[:, j]
        significant = sample.pvals["codetected"][:, j] < 0.05
        self.assertGreater(significant[patch & tested].mean(), 0.5)
        self.assertLess(significant[~patch & tested].mean(), 0.15)


class TestGeneQuantiles(unittest.TestCase):
    def test_matches_quantiles_per_gene(self):
        """Quantiles computed in chunks equal np.quantile gene by gene, for chunk
        sizes that split the genes unevenly, one at a time, or all at once."""
        adata = _sparse_cells_adata(n_cells=100, n_extra_genes=60)
        genes = adata.var_names.to_numpy()
        quantiles = np.array([0.5, 0.75, 0.9, 0.99, 1])
        expected = np.apply_along_axis(
            np.quantile,
            0,
            adata.X.toarray(),
            q=quantiles,
            method="nearest",
        )
        for chunk_size in (1, 7, len(genes), 1000):
            np.testing.assert_array_equal(
                gene_quantiles(adata, genes, quantiles, chunk_size),
                expected,
                err_msg=f"{chunk_size=}",
            )


if __name__ == "__main__":
    unittest.main()
