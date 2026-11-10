#!/usr/bin/env python

"""Runs st.tl.cci.run_cci on the test data with cell-type labels stored as a
categorical and as plain strings.
"""

import unittest
from typing import ClassVar

import numpy as np
import pandas as pd
from anndata import AnnData

import stlearn as st
from tests.utils import read_test_data

CELL_TYPE_LABELS = np.array(["CT1", "CT2", "CT3"])


def read_test_data_with_lr_pairs(n_pairs: int = 20):
    """Test data that ligand-receptor analysis can run on.

    The 100 genes in the test data contain no pair from the ligand-receptor
    database and the matrix is scaled, so `st.tl.cci.run` has nothing to test. This
    renames the first `2 * n_pairs` genes to database pairs and clips negative values
    to zero. It exercises the code path; the scores carry no biological meaning.
    """
    adata = read_test_data()
    lrs = st.tl.cci.lr_pairs("connectomeDB2020")

    genes: list[str] = []
    for ligand, receptor in zip(
            lrs.ligands().tolist(), lrs.receptors().tolist(), strict=True
    ):
        if ligand != receptor and ligand not in genes and receptor not in genes:
            genes += [ligand, receptor]
        if len(genes) == 2 * n_pairs:
            break

    names = list(adata.var_names)
    names[: len(genes)] = genes
    adata.var_names = names
    adata.X = np.clip(adata.X, 0, None)
    return adata, lrs


class TestCCILabels(unittest.TestCase):
    """run_cci with cell-type labels stored as a categorical and as plain strings.

    Under pandas 3, `Series.values.astype(str)` returns an Arrow-backed string array,
    which numba rejects, so run_cci failed for both dtypes.
    """

    adata: ClassVar[AnnData]
    lr_results: ClassVar[AnnData]

    @classmethod
    def setUpClass(cls):
        adata, lrs = read_test_data_with_lr_pairs()
        cls.lr_results = st.tl.cci.run(
            adata,
            lrs,
            min_spots=5,
            distance=None,
            n_pairs=200,
            n_cpus=1,
            verbose=False,
            null="random"
        )
        # Three bands of cell types across the tissue.
        bands = pd.qcut(adata.obs["imagecol"], 3, labels=CELL_TYPE_LABELS)
        adata.obs["cell_type"] = bands
        adata.obs["cell_type_str"] = bands.to_numpy().astype(str)
        cls.adata = adata

    def assert_run_cci_runs(self, use_label: str):
        adata = self.adata.copy()
        lr_results = self.lr_results.copy()
        st.tl.cci.run_cci(
            adata,
            use_label,
            lr_results=lr_results,
            min_spots=3,
            n_perms=20,
            n_cpus=1,
            verbose=False,
        )

        matrix = lr_results.uns[f"lr_cci_{use_label}"]
        self.assertEqual(list(matrix.index), CELL_TYPE_LABELS.tolist())
        self.assertEqual(list(matrix.columns), CELL_TYPE_LABELS.tolist())

    def test_categorical_labels(self):
        self.assert_run_cci_runs("cell_type")

    def test_string_labels(self):
        self.assert_run_cci_runs("cell_type_str")
