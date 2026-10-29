#!/usr/bin/env python

"""Tests for stlearn.tl.cci.perm_utils: the random gene pairs of a background, and
the quantiles of sparse expression they are chosen by."""

import unittest

import numpy as np
import scipy.sparse

from stlearn.tl.cci.perm_utils import gen_rand_pairs, nonnegative, sparse_quantiles

QUANTILES = np.array([0.5, 0.9, 0.99, 1.0])


class TestPermUtils(unittest.TestCase):
    def test_unique_pairs_of_distinct_genes_from_the_candidates(self):
        """n_pairs pairs, no two the same, never a gene with itself, the ligand
        from genes1 and the receptor from genes2, which share some genes."""
        rng = np.random.default_rng(0)
        genes1 = rng.choice(500, 63, replace=False)
        genes2 = np.concatenate(
            [genes1[:20], rng.choice(np.arange(500, 900), 43, replace=False)]
        )

        pairs = gen_rand_pairs(genes1, genes2, 1000, seed=3)

        self.assertEqual(pairs.shape, (1000, 2))
        self.assertEqual(pairs.dtype, np.int64)
        self.assertEqual(len(np.unique(pairs, axis=0)), 1000)
        self.assertTrue((pairs[:, 0] != pairs[:, 1]).all())
        self.assertTrue(np.isin(pairs[:, 0], genes1).all())
        self.assertTrue(np.isin(pairs[:, 1], genes2).all())

    def test_same_pairs_as_the_name_based_generator_drew(self):
        """The pairs the previous generator, which drew gene names and joined
        them as 'L_R', drew for these candidates and seed: the same random
        stream and rejections, so backgrounds and p-values are unchanged. The
        candidates are unsorted and share two genes, so self-pairs and
        duplicates are rejected along the way."""
        genes1, genes2 = np.array([4, 0, 9, 2, 7]), np.array([2, 7, 1, 8])
        expected = [[9, 2], [0, 8], [4, 2], [4, 7], [2, 1], [2, 7], [0, 1], [4, 1]]
        np.testing.assert_array_equal(gen_rand_pairs(genes1, genes2, 8, 3), expected)

    def test_seed_reproduces_pairs(self):
        genes1, genes2 = np.arange(0, 63), np.arange(40, 103)
        first = gen_rand_pairs(genes1, genes2, 500, seed=7)
        np.testing.assert_array_equal(gen_rand_pairs(genes1, genes2, 500, 7), first)
        self.assertFalse(np.array_equal(gen_rand_pairs(genes1, genes2, 500, 8), first))

    def test_all_possible_pairs_when_asked_for_them(self):
        """Every pair but the self-pairs: 3 x 3 genes sharing 2 leaves 7."""
        genes1, genes2 = np.array([0, 1, 2]), np.array([1, 2, 3])
        pairs = gen_rand_pairs(genes1, genes2, 7, seed=0)
        expected = {(a, b) for a in genes1 for b in genes2 if a != b}
        self.assertEqual(set(map(tuple, pairs)), expected)

    def test_too_many_pairs_is_an_error(self):
        with self.assertRaises(ValueError):
            gen_rand_pairs(np.array([0, 1, 2]), np.array([1, 2, 3]), 8, seed=0)

    def test_repeated_gene_is_an_error(self):
        """Pairs are unique by position, so a gene listed twice is refused
        rather than letting the same pair in twice."""
        with self.assertRaises(ValueError):
            gen_rand_pairs(np.array([0, 1, 1]), np.array([2, 3, 4]), 2, seed=0)

    def test_matches_np_quantile(self):
        """Each column's quantiles as np.quantile's method="nearest" picks them,
        on sparse data with some explicit zeros stored."""
        X = scipy.sparse.random(
            500, 40, density=0.05, random_state=0, format="csc", dtype=np.float32
        )
        X.data[:5] = 0
        expected = np.quantile(X.toarray(), QUANTILES, axis=0, method="nearest")
        got = sparse_quantiles(X, QUANTILES)
        np.testing.assert_array_equal(got, expected)
        self.assertEqual(got.dtype, expected.dtype)

    def test_no_nonzeros(self):
        """A matrix with nothing stored is all zeros, not an error."""
        X = scipy.sparse.csc_matrix((5, 3), dtype=np.float32)
        np.testing.assert_array_equal(sparse_quantiles(X, QUANTILES), np.zeros((4, 3)))

    def test_any_sparse_format(self):
        """Columns are the genes whatever format the matrix comes in."""
        X = scipy.sparse.random(50, 7, density=0.2, random_state=1, format="csc")
        expected = sparse_quantiles(X, QUANTILES)
        for other in (X.tocsr(), X.tocoo()):
            with self.subTest(format=other.format):
                np.testing.assert_array_equal(
                    sparse_quantiles(other, QUANTILES), expected
                )

    def test_nonnegative_stored_values(self):
        """Every stored value at or above 0, including a stored 0; NaN is not;
        nothing stored is."""
        cases = [
            ([0.5, 2.0], True),
            ([0.0, 1.0], True),
            ([1.0, -0.1], False),
            ([1.0, np.nan], False),
            ([], True),
        ]
        for data, expected in cases:
            with self.subTest(data=data):
                coords = ([0] * len(data), list(range(len(data))))
                X = scipy.sparse.csc_matrix((np.array(data), coords), shape=(1, 3))
                self.assertEqual(nonnegative(X), expected)


if __name__ == "__main__":
    unittest.main()
