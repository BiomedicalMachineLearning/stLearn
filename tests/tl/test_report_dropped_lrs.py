"""Tests for the report of the LR pairs st.tl.cci.run drops before testing."""

import contextlib
import io
import unittest

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

from stlearn.tl.cci.analysis import _report_dropped_lrs


def _adata(sparse=False):
    """Four spots: A is 0.3 everywhere, E is never expressed, the rest are 1."""
    X = np.array([[0.3, 1.0, 1.0, 1.0, 0.0]] * 4, dtype=np.float32)
    return AnnData(
        scipy.sparse.csr_matrix(X) if sparse else X,
        var=pd.DataFrame(index=list("ABCDE")),
    )


def _report(adata, lrs, n_scored, min_spots=10, min_expr=0.0):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _report_dropped_lrs(
            adata, np.array(lrs), np.array(n_scored), min_spots, min_expr
        )
    return out.getvalue()


class TestReportDroppedLrs(unittest.TestCase):
    def test_pair_with_an_unexpressed_gene(self):
        """A pair dropped because a gene is never expressed names the gene."""
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                report = _report(_adata(sparse), ["E_B", "C_D"], [0, 15])
                self.assertIn(
                    "Dropping 1 LR pairs with a gene never expressed above "
                    "min_expr=0.0: E.",
                    report,
                )
                self.assertNotIn("or fewer spots", report)
                self.assertIn("Altogether 1 valid L-R pairs", report)

    def test_pair_scored_in_too_few_spots(self):
        report = _report(_adata(), ["C_D", "B_C"], [3, 15])
        self.assertIn("Dropping 1 LR pairs scored in 10 or fewer spots.", report)
        self.assertNotIn("never expressed", report)
        self.assertIn("Altogether 1 valid L-R pairs", report)

    def test_only_dropped_pairs_are_reported(self):
        """With min_expr above 0, a gene never above it can still score through
        its partner: A_B is kept, so it is not reported, and C_D's reason is."""
        report = _report(_adata(), ["A_B", "C_D"], [15, 3], min_expr=0.5)
        self.assertNotIn("never expressed", report)
        self.assertIn("Dropping 1 LR pairs scored in 10 or fewer spots.", report)
        self.assertIn("Altogether 1 valid L-R pairs", report)

    def test_nothing_dropped(self):
        report = _report(_adata(), ["A_B"], [15], min_expr=0.5)
        self.assertEqual(report, "Altogether 1 valid L-R pairs\n")

    def test_names_at_most_max_named_genes(self):
        genes = [f"G{i}" for i in range(12)]
        adata = AnnData(np.zeros((2, 12)), var=pd.DataFrame(index=genes))
        lrs = [f"G{i}_G{i + 1}" for i in range(11)]
        report = _report(adata, lrs, [0] * 11)
        self.assertIn("G0, G1, G10, G11, G2, G3, G4, G5, G6, G7 and 2 more.", report)
        self.assertIn("Altogether 0 valid L-R pairs", report)
