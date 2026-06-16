#!/usr/bin/env python

import unittest

from numba.typed import List
import numpy as np

from stlearn.tl.cci.base import lr_core


def _reference_lr_core(spot_lr1, spot_lr2, neighbour_lists, min_expr, spot_indices):
    """Plain-numpy oracle for lr_core's intended semantics (slow but obvious)."""
    n_rows = len(spot_indices)
    n_cols = spot_lr2.shape[1]
    nb_lr2 = np.zeros((n_rows, n_cols), np.float64)
    for i, si in enumerate(spot_indices):
        neigh = np.asarray(neighbour_lists[si], dtype=np.int64)
        if len(neigh) > 0:
            nb_lr2[i] = spot_lr2[neigh].mean(axis=0)
    sl1 = spot_lr1[spot_indices]
    scores = sl1 * (nb_lr2 > min_expr) + (sl1 > min_expr) * nb_lr2
    return scores.sum(axis=1) / 2.0


class TestLrCore(unittest.TestCase):
    def test_matches_reference_on_random_inputs(self):
        """lr_core agrees with the numpy oracle across shapes and thresholds."""
        rng = np.random.default_rng(0)
        for n_spots, n_cols, min_expr in [
            (10, 2, 0.0),
            (10, 2, 0.5),
            (25, 4, 0.0),
            (40, 2, 1.0),
        ]:
            spot_lr1 = rng.random((n_spots, n_cols))
            spot_lr2 = rng.random((n_spots, n_cols))
            # Random neighbour sets, including some empty ones.
            neighbour_lists = []
            for _ in range(n_spots):
                k = rng.integers(0, 4)
                if k == 0:
                    neighbour_lists.append(np.array([], dtype=np.int32))
                else:
                    neighbour_lists.append(
                        rng.choice(n_spots, size=k, replace=False).astype(np.int32)
                    )
            spot_indices = np.arange(n_spots, dtype=np.int32)

            got = lr_core(
                spot_lr1,
                spot_lr2,
                neighbour_lists,
                min_expr,
                spot_indices,
            )
            expected = _reference_lr_core(
                spot_lr1, spot_lr2, neighbour_lists, min_expr, spot_indices
            )
            np.testing.assert_allclose(
                got,
                expected,
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"mismatch for {n_spots=} {n_cols=} {min_expr=}",
            )

    def test_returns_float64(self):
        spot_lr1 = np.array([[1.0, 0.0], [5.0, 3.0], [0.0, 7.0]], dtype=np.float64)
        spot_lr2 = np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 0.0]], dtype=np.float64)
        neighbour_lists = [
            np.array([1, 2], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([], dtype=np.int32),
        ]

        result = lr_core(
            spot_lr1,
            spot_lr2,
            List(neighbour_lists),
            0.0,
            np.array([0, 1, 2], dtype=np.int32),
        )
        self.assertEqual(result.dtype, np.float64)
