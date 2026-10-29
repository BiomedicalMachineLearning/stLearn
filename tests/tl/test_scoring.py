"""Tests for stlearn.tl.cci.scoring: LR scores against a plain-numpy reference."""

import unittest
from unittest import mock

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData
from numba.typed import List

from stlearn.tl.cci import scoring
from stlearn.tl.cci.neighbours import calc_neighbours
from stlearn.tl.cci.scoring import (
    LRScorer,
    LRPairs,
    flatten_neighbours,
    get_lrs_scores,
    get_scores,
    neighbourhood_means,
)


def _reference_scores(expr, pairs, neighbour_lists, het_vals, min_expr, spot_indices):
    """Plain-numpy oracle for get_scores (slow but obvious).

    For each pair: the (ligand, receptor) expression at the spot, against the
    (receptor, ligand) expression averaged over the spot's neighbours (0 with no
    neighbours). Each direction counts a gene at the spot when its partner's mean
    is above min_expr, and the partner's mean when the gene is above min_expr. The
    two directions are averaged and multiplied by the spot's het value.
    """
    scores = np.zeros((len(spot_indices), len(pairs)), np.float64)
    for j, (ligand, receptor) in enumerate(pairs):
        at_spot = expr[:, [ligand, receptor]][spot_indices]
        partner = expr[:, [receptor, ligand]]
        partner_mean = np.zeros((len(spot_indices), 2), np.float64)
        for i, si in enumerate(spot_indices):
            neigh = np.asarray(neighbour_lists[si], dtype=np.int64)
            if len(neigh) > 0:
                partner_mean[i] = partner[neigh].mean(axis=0)
        both = at_spot * (partner_mean > min_expr) + (at_spot > min_expr) * partner_mean
        scores[:, j] = het_vals[spot_indices] * both.sum(axis=1) / 2.0
    return scores


def _random_neighbours(rng, n_spots, max_neighbours=4):
    """Random neighbour sets of up to max_neighbours - 1 other spots, some empty."""
    neighbour_lists = []
    for spot in range(n_spots):
        others = np.delete(np.arange(n_spots), spot)
        k = rng.integers(0, max_neighbours)
        neighbour_lists.append(
            rng.choice(others, size=k, replace=False).astype(np.int32),
        )
    return neighbour_lists


class TestGetScores(unittest.TestCase):
    def test_worked_example(self):
        """Spot 0: ligand 1 with receptor mean (3 + 7) / 2 = 5 gives 1 + 5; its
        receptor is 0, so (6 + 0) / 2 = 3. Spot 1: ligand 5 with receptor mean 0
        gives 0; receptor 3 with ligand mean 1 gives 3 + 1; (0 + 4) / 2 = 2. Spot 2
        has no neighbours, so 0. Then each is multiplied by its het value."""
        expr = np.array([[1.0, 0.0], [5.0, 3.0], [0.0, 7.0]])  # ligand, receptor
        neighbours = List(
            [
                np.array([1, 2], dtype=np.int32),
                np.array([0], dtype=np.int32),
                np.array([], dtype=np.int32),
            ],
        )
        pairs = np.array([[0, 1]])
        spots = np.arange(3, dtype=np.int32)

        scores = get_scores(expr, pairs, neighbours, np.ones(3), 0.0, spots)
        self.assertEqual(scores.dtype, np.float64)
        np.testing.assert_array_equal(scores[:, 0], [3.0, 2.0, 0.0])

        het_vals = np.array([1, 2, 3])
        scores = get_scores(expr, pairs, neighbours, het_vals, 0.0, spots)
        np.testing.assert_array_equal(scores[:, 0], [3.0, 4.0, 0.0])

    def test_matches_reference_on_random_inputs(self):
        """get_scores agrees with the numpy oracle across shapes, thresholds, het
        values and subsets of spots. Integer-valued expression puts values exactly
        on min_expr, which is not above it."""
        rng = np.random.default_rng(0)
        for n_spots, n_genes, n_pairs, min_expr, counts in [
            (10, 2, 1, 0.0, False),
            (10, 4, 3, 0.5, False),
            (25, 6, 8, 0.0, True),
            (40, 5, 6, 1.0, True),
            (40, 3, 4, 2.0, True),
        ]:
            msg = f"{n_spots=} {n_genes=} {n_pairs=} {min_expr=} {counts=}"
            if counts:
                expr = rng.poisson(1.5, size=(n_spots, n_genes)).astype(np.float64)
            else:
                expr = rng.random((n_spots, n_genes))
            # Pairs of distinct genes; genes are shared between pairs.
            pairs = np.array(
                [rng.choice(n_genes, size=2, replace=False) for _ in range(n_pairs)],
            )
            neighbour_lists = _random_neighbours(rng, n_spots)
            het_vals = rng.integers(0, 4, size=n_spots)
            all_spots = np.arange(n_spots, dtype=np.int32)
            some_spots = np.sort(rng.choice(n_spots, size=n_spots // 2, replace=False))

            for spots in (all_spots, some_spots):
                got = get_scores(
                    np.asfortranarray(expr),
                    pairs,
                    List(neighbour_lists),
                    het_vals,
                    min_expr,
                    spots,
                )
                expected = _reference_scores(
                    expr,
                    pairs,
                    neighbour_lists,
                    het_vals,
                    min_expr,
                    spots,
                )
                self.assertEqual(got.shape, (len(spots), n_pairs), msg=msg)
                np.testing.assert_allclose(
                    got,
                    expected,
                    rtol=1e-12,
                    atol=1e-12,
                    err_msg=msg,
                )

    def test_means_once_per_gene(self):
        """Each gene's neighbourhood mean is computed once, for the scored spots
        only: genes by scored spots, 0 for a spot with no neighbours."""
        rng = np.random.default_rng(1)
        n_spots, n_genes = 30, 5
        expr = np.asfortranarray(rng.random((n_spots, n_genes)))
        neighbour_lists = _random_neighbours(rng, n_spots)
        spots = np.sort(rng.choice(n_spots, size=12, replace=False)).astype(np.int32)

        starts, flat = flatten_neighbours(List(neighbour_lists))
        means = neighbourhood_means(expr, starts, flat, spots)

        self.assertEqual(means.shape, (n_genes, len(spots)))
        self.assertEqual(means.dtype, np.float64)
        expected = np.zeros((n_genes, len(spots)))
        for s, si in enumerate(spots):
            if len(neighbour_lists[si]) > 0:
                expected[:, s] = expr[neighbour_lists[si]].mean(axis=0)
        np.testing.assert_allclose(means, expected, rtol=1e-12, atol=1e-12)

    def test_spots_scored_in_chunks(self):
        """Scores are the same however many chunks of spots the means are held in,
        including a chunk boundary inside the spots and a final short chunk."""
        rng = np.random.default_rng(2)
        n_spots, n_genes = 23, 4
        expr = np.asfortranarray(rng.random((n_spots, n_genes)))
        pairs = np.array([[0, 1], [2, 3], [1, 2]])
        neighbours = List(_random_neighbours(rng, n_spots))
        het_vals = np.ones(n_spots)
        spots = np.arange(n_spots, dtype=np.int32)

        whole = get_scores(expr, pairs, neighbours, het_vals, 0.0, spots)
        chunk_bytes = scoring.MEANS_CHUNK_BYTES
        try:
            # 8 bytes per mean, n_genes genes: 5 spots of means per chunk.
            scoring.MEANS_CHUNK_BYTES = 8 * n_genes * 5
            chunked = get_scores(expr, pairs, neighbours, het_vals, 0.0, spots)
        finally:
            scoring.MEANS_CHUNK_BYTES = chunk_bytes
        np.testing.assert_array_equal(chunked, whole)


def _make_adata(n_side=6, seed=0, sparse=True):
    rng = np.random.default_rng(seed)
    n_spots = n_side * n_side
    var_names = ["Lig0", "Rec0", "Lig1", "Rec1", "Rec2", "gene0"]
    X = rng.poisson(1.0, size=(n_spots, len(var_names))).astype(np.float32)
    rows, cols = np.divmod(np.arange(n_spots), n_side)
    obs = pd.DataFrame(
        {"imagerow": rows.astype(float), "imagecol": cols.astype(float)},
        index=[f"spot{i}" for i in range(n_spots)],
    )
    return AnnData(
        X=scipy.sparse.csr_matrix(X) if sparse else X,
        obs=obs,
        var=pd.DataFrame(index=var_names),
    )


class TestGetLrsScores(unittest.TestCase):
    def test_matches_reference_through_adata(self):
        """From named LR pairs on sparse or dense data: pairs with a gene missing
        from the data are dropped, the rest keep their order, and the scores match
        the oracle, including pairs that share a gene and a reversed pair."""
        lrs = LRPairs.from_names(
            ["Lig0_Rec0", "Lig1_Rec1", "Lig0_Missing", "Lig1_Rec2", "Rec0_Lig0"],
            species="human",
        )
        kept = ["Lig0_Rec0", "Lig1_Rec1", "Lig1_Rec2", "Rec0_Lig0"]
        for sparse in (True, False):
            adata = _make_adata(sparse=sparse)
            neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
            het_vals = np.random.default_rng(1).integers(1, 4, size=len(adata))

            scores, new_lrs = get_lrs_scores(adata, lrs, neighbours, het_vals, 0.5)

            self.assertEqual(list(new_lrs), kept)
            dense = adata.to_df()
            columns = {gene: i for i, gene in enumerate(dense.columns)}
            pairs = np.array([[columns[g] for g in lr.split("_")] for lr in kept])
            expected = _reference_scores(
                dense.to_numpy(dtype=np.float64),
                pairs,
                list(neighbours),
                het_vals,
                0.5,
                np.arange(len(adata)),
            )
            np.testing.assert_allclose(
                scores,
                expected,
                rtol=1e-6,
                err_msg=f"{sparse=}",
            )
            # Both directions are averaged, so a reversed pair scores the same.
            np.testing.assert_array_equal(scores[:, 0], scores[:, 3])

    def test_spot_subset_matches_rows_of_all_spots(self):
        """Scoring a subset of spots gives those spots' rows of the full scores."""
        adata = _make_adata()
        neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
        het_vals = np.ones(len(adata))
        lrs = LRPairs.from_names(["Lig0_Rec0", "Lig1_Rec1"], species="human")
        full, _ = get_lrs_scores(adata, lrs, neighbours, het_vals, 0)
        spots = np.flatnonzero(full[:, 0] > 0)  # as perform_spot_testing chooses them
        subset, _ = get_lrs_scores(
            adata,
            lrs,
            neighbours,
            het_vals,
            0,
            filter_pairs=False,
            spot_indices=spots,
        )
        np.testing.assert_array_equal(subset, full[spots])

    def test_homophilic_pair_scores_as_one_direction(self):
        """A pair of a gene with itself, as PECAM1_PECAM1 in connectomeDB, is
        kept and scored like any other: both directions are then the same, so
        the score is the gene at the spot with its own neighbourhood mean."""
        adata = _make_adata()
        neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
        het_vals = np.ones(len(adata))
        lrs = LRPairs.from_names(["Lig0_Lig0", "Lig0_Rec0"], species="human")

        scores, new_lrs = get_lrs_scores(adata, lrs, neighbours, het_vals, 0.5)

        self.assertEqual(list(new_lrs), ["Lig0_Lig0", "Lig0_Rec0"])
        dense = adata.to_df().to_numpy(dtype=np.float64)
        expected = _reference_scores(
            dense,
            [[0, 0]],
            list(neighbours),
            het_vals,
            0.5,
            np.arange(len(adata)),
        )
        np.testing.assert_allclose(scores[:, [0]], expected, rtol=1e-6)
        at_spot = dense[:, 0]
        own_mean = np.array([dense[n, 0].mean() if len(n) else 0.0 for n in neighbours])
        one_direction = at_spot * (own_mean > 0.5) + (at_spot > 0.5) * own_mean
        np.testing.assert_allclose(scores[:, 0], one_direction, rtol=1e-6)

    def test_no_pairs_in_data(self):
        """With every pair missing a gene there are no scores, but the shape
        still has a row per spot, as run checks new_lrs.size to fail fast."""
        adata = _make_adata()
        neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
        lrs = LRPairs.from_names(["Lig0_Nope", "Nope_Rec0"], species="human")
        scores, new_lrs = get_lrs_scores(
            adata,
            lrs,
            neighbours,
            np.ones(36),
            0,
        )
        self.assertEqual(scores.shape, (len(adata), 0))
        self.assertEqual(scores.dtype, np.float64)
        self.assertEqual(len(new_lrs), 0)

    def test_scores_accept_integer_expression(self):
        """Integer counts score as float64, the same as the float data."""
        adata = _make_adata(sparse=False)
        int_adata = adata.copy()
        int_adata.X = adata.X.astype(np.int64)
        neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
        het_vals = np.ones(len(adata))
        lrs = LRPairs.from_names(["Lig0_Rec0", "Lig1_Rec1"], species="human")

        scores, new_lrs = get_lrs_scores(int_adata, lrs, neighbours, het_vals, 0)
        float_scores, _ = get_lrs_scores(adata, lrs, neighbours, het_vals, 0)

        self.assertEqual(scores.dtype, np.float64)
        self.assertEqual(scores.shape, (len(adata), len(new_lrs)))
        np.testing.assert_allclose(scores, float_scores)


class TestLRScorer(unittest.TestCase):
    def test_reused_scorer_matches_one_off_calls(self):
        """One LRScorer scoring several sets of pairs at several subsets of spots,
        as perform_spot_testing does per LR, gives exactly what get_lrs_scores
        gives each call on its own, on sparse and dense data."""
        rng = np.random.default_rng(3)
        calls = [
            (LRPairs.from_names(["Lig0_Rec0", "Lig1_Rec1"], species="human"), None),
            (
                LRPairs.from_names(
                    ["gene0_Rec2", "Lig1_gene0", "Rec0_Lig0"],
                    species="human",
                ),
                np.array([5, 1, 30]),
            ),
            (
                LRPairs.from_names(["Lig0_Rec0"], species="human"),
                np.sort(rng.choice(36, 12, replace=False)),
            ),
            (
                LRPairs.from_names(["Rec2_Lig1"], species="human"),
                np.array([], dtype=np.int32),
            ),
        ]
        for sparse in (True, False):
            adata = _make_adata(sparse=sparse)
            neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
            het_vals = rng.integers(1, 4, size=len(adata))
            scorer = LRScorer(adata, neighbours)
            # Sparse data is kept by column; dense data is read as it is.
            self.assertEqual(scorer.columns is not None, sparse)

            for lrs, spots in calls:
                got, got_lrs = scorer.score(
                    lrs,
                    het_vals,
                    0.5,
                    filter_pairs=False,
                    spot_indices=spots,
                )
                expected, expected_lrs = get_lrs_scores(
                    adata,
                    lrs,
                    neighbours,
                    het_vals,
                    0.5,
                    False,
                    spots,
                )
                msg = f"{sparse=} {lrs=}"
                np.testing.assert_array_equal(got_lrs, expected_lrs, err_msg=msg)
                np.testing.assert_array_equal(got, expected, err_msg=msg)
                n_spots = len(adata) if spots is None else len(spots)
                self.assertEqual(got.shape, (n_spots, len(lrs)), msg=msg)

    def test_filters_pairs_like_get_lrs_scores(self):
        adata = _make_adata()
        neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
        lrs = LRPairs.from_names(
            ["Lig0_Rec0", "Lig0_Missing", "Lig1_Rec2"],
            species="human",
        )
        scorer = LRScorer(adata, neighbours)

        scores, new_lrs = scorer.score(lrs, np.ones(len(adata)), 0)
        expected, expected_lrs = get_lrs_scores(
            adata,
            lrs,
            neighbours,
            np.ones(len(adata)),
            0,
        )

        self.assertEqual(list(new_lrs), ["Lig0_Rec0", "Lig1_Rec2"])
        np.testing.assert_array_equal(new_lrs, expected_lrs)
        np.testing.assert_array_equal(scores, expected)

    def test_expression_is_the_genes_columns(self):
        """The genes asked for, by position and in that order: sparse data (kept
        by column, or not) as a CSR matrix, so a chunk of spots' rows are cheap
        to take out; dense data as it is, column-major."""
        for sparse, by_column in [(True, True), (True, False), (False, True)]:
            adata = _make_adata(sparse=sparse)
            neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
            genes = np.array(["Rec1", "Lig0", "gene0"])
            positions = adata.var_names.get_indexer(genes)

            expr = LRScorer(adata, neighbours, by_column).expression(positions)

            msg = f"{sparse=} {by_column=}"
            self.assertEqual(expr.shape, (len(adata), 3), msg=msg)
            if sparse:
                self.assertIsInstance(expr, scipy.sparse.csr_matrix, msg=msg)
                expr = expr.toarray()
            else:
                self.assertTrue(expr.flags.f_contiguous, msg=msg)
            np.testing.assert_array_equal(expr, adata[:, genes].to_df().to_numpy())

    def test_sparse_result_holds_the_scores_above_zero(self):
        """With sparse_result the scores are a canonical CSC matrix of the dense
        scores above 0, at every spot or a subset; a spot with het value 0
        scores 0 everywhere, so has no entries."""
        rng = np.random.default_rng(5)
        lrs = LRPairs.from_names(
            ["Lig0_Rec0", "Lig1_Rec1", "Rec2_Lig1", "gene0_Rec1"],
            species="human",
        )
        adata = _make_adata()
        neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
        het_vals = rng.integers(0, 3, size=len(adata))
        scorer = LRScorer(adata, neighbours)
        for spots in (None, np.sort(rng.choice(36, 20, replace=False))):
            dense, _ = scorer.score(lrs, het_vals, 0.5, spot_indices=spots)
            got, got_lrs = scorer.score(
                lrs,
                het_vals,
                0.5,
                spot_indices=spots,
                sparse_result=True,
            )
            msg = f"{spots=}"
            self.assertIsInstance(got, scipy.sparse.csc_matrix, msg=msg)
            self.assertTrue(got.has_canonical_format, msg=msg)
            self.assertTrue((got.data > 0).all(), msg=msg)
            np.testing.assert_array_equal(
                got.toarray(),
                np.where(dense > 0, dense, 0),
                err_msg=msg,
            )
            np.testing.assert_array_equal(
                got_lrs,
                ["Lig0_Rec0", "Lig1_Rec1", "Rec2_Lig1", "gene0_Rec1"],
            )

    def test_sparse_result_in_chunks(self):
        """The sparse scores are the same however many chunks of spots are
        scored, with a chunk boundary inside the spots and a short last chunk."""
        adata = _make_adata()
        neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
        lrs = LRPairs.from_names(
            ["Lig0_Rec0", "Lig1_Rec1", "Rec2_Lig1"],
            species="human",
        )
        whole, _ = get_lrs_scores(
            adata,
            lrs,
            neighbours,
            np.ones(36),
            0,
            sparse_result=True,
        )
        chunk_bytes = scoring.MEANS_CHUNK_BYTES
        try:
            scoring.MEANS_CHUNK_BYTES = 8 * 5 * 7  # 5 genes: 7 spots per chunk
            chunked, _ = get_lrs_scores(
                adata,
                lrs,
                neighbours,
                np.ones(36),
                0,
                sparse_result=True,
            )
        finally:
            scoring.MEANS_CHUNK_BYTES = chunk_bytes
        self.assertTrue(chunked.has_canonical_format)
        np.testing.assert_array_equal(chunked.indptr, whole.indptr)
        np.testing.assert_array_equal(chunked.indices, whole.indices)
        np.testing.assert_array_equal(chunked.data, whole.data)

    def test_only_a_chunks_rows_are_made_dense(self):
        """Scoring takes out of the expression only the rows a chunk needs: its
        spots and their neighbours."""
        adata = _make_adata()
        neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
        scorer = LRScorer(adata, neighbours)
        spots = np.array([0, 1, 35], dtype=np.int32)
        needed = set(spots) | {n for s in spots for n in neighbours[s]}
        taken = []
        original = scoring._dense_rows

        def recording(expr, rows):
            taken.append(np.asarray(rows))
            return original(expr, rows)

        with mock.patch.object(scoring, "_dense_rows", recording):
            scores = scorer.score_indices(
                np.array([[0, 1]]),
                np.ones(36),
                0.0,
                spot_indices=spots,
            )
        self.assertEqual(len(taken), 1)
        self.assertEqual(set(taken[0]), needed)
        self.assertLess(len(taken[0]), len(adata))
        expected = scorer.score_indices(np.array([[0, 1]]), np.ones(36), 0.0)[spots]
        np.testing.assert_array_equal(scores, expected)

    def test_scores_pairs_by_position(self):
        """Pairs as positions in var_names score exactly as the same pairs named,
        at every spot or a subset, as the random pairs of a background are."""
        adata = _make_adata()
        neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
        het_vals = np.random.default_rng(4).integers(1, 4, size=len(adata))
        lrs = LRPairs.from_names(
            ["Lig0_Rec0", "Rec2_Lig1", "gene0_Rec1"],
            species="human",
        )
        pairs = np.array([[0, 1], [4, 2], [5, 3]])
        scorer = LRScorer(adata, neighbours)
        for spots in (None, np.array([7, 0, 35])):
            named, _ = scorer.score(lrs, het_vals, 0.5, spot_indices=spots)
            by_position = scorer.score_indices(pairs, het_vals, 0.5, spots)
            np.testing.assert_array_equal(by_position, named)
            n_spots = len(adata) if spots is None else len(spots)
            self.assertEqual(by_position.shape, (n_spots, 3))

    def test_missing_gene_without_filtering_is_an_error(self):
        """Not filtering pairs promises every gene is present; a missing one is
        a KeyError naming it, never some other gene's column."""
        adata = _make_adata()
        neighbours = calc_neighbours(adata, distance=1.5, verbose=False)
        for by_column in (True, False):
            scorer = LRScorer(adata, neighbours, by_column=by_column)
            lrs = LRPairs.from_names(["Lig0_Rec0", "Lig0_Nope"], species="human")
            with self.assertRaises(KeyError, msg=f"{by_column=}") as raised:
                scorer.score(
                    lrs,
                    np.ones(36),
                    0,
                    False,
                )
            self.assertIn("Nope", str(raised.exception))
