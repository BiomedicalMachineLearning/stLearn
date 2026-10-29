#!/usr/bin/env python

"""Tests for the ligand-receptor and cell-cell interaction functions in `stlearn`.

TestCCI holds unit tests of the neighbourhood and counting helpers on a toy
seven-spot neighbourhood. TestRunCCILabels runs st.tl.cci.run_cci on the test data
with cell-type labels stored as a categorical and as plain strings.
"""

import unittest
from typing import ClassVar

import numpy as np
import pandas as pd
from anndata import AnnData
from numba.typed import List

import stlearn as st
import stlearn.tl.cci.edges as edges
import stlearn.tl.cci.het as het
from tests.utils import read_test_data

# Per line - which cells to annotate
CELL_TYPE_ANNOTATIONS = ["CT1", "CT2", "CT3", "CT2", "CT1", "CT3", "CT2"]

# 3 cell types: A,E -> CT1, B,G -> CT2, C,F -> CT3
CELL_TYPE_LABELS = np.array(["CT1", "CT2", "CT3"])


def read_test_data_with_lr_pairs(n_pairs: int = 20):
    """Test data that ligand-receptor analysis can run on.

    The 100 genes in the test data contain no pair from the ligand-receptor
    database and the matrix is scaled, so `st.tl.cci.run` has nothing to test. This
    renames the first `2 * n_pairs` genes to database pairs and clips negative values
    to zero. It exercises the code path; the scores carry no biological meaning.
    """
    adata = read_test_data()
    lrs = st.tl.cci.lr_pairs("connectomeDB2020").names()

    genes: list[str] = []
    for lr in lrs:
        ligand, receptor = lr.split("_")
        if ligand != receptor and ligand not in genes and receptor not in genes:
            genes += [ligand, receptor]
        if len(genes) == 2 * n_pairs:
            break

    names = list(adata.var_names)
    names[: len(genes)] = genes
    adata.var_names = names
    adata.X = np.clip(adata.X, 0, None)
    return adata, lrs


class TestCCI(unittest.TestCase):
    """Tests for `stlearn` CCI capability."""

    def setUp(self) -> None:
        """Setup some basic test-cases as sanity checks."""

        # Unit neighbourhood, containing just 1 spot and 6 neighbours
        """
        * A is the middle spot, B/C/D/E/F/G are the neighbouring spots clock-
                                                  wise starting at the top-left.
        """
        spot_bcs = np.array(["A", "B", "C", "D", "E", "F", "G"])
        neigh_dict = {
            "A": ["B", "C", "D", "E", "F", "G"],
            "B": ["G", "A", "C"],
            "C": ["B", "A", "D"],
            "D": ["E", "A", "C"],
            "E": ["F", "A", "D"],
            "F": ["G", "A", "E"],
            "G": ["F", "A", "B"],
        }
        neighbourhood_bcs = List()
        neighbourhood_indices = List()
        for i, bc in enumerate(spot_bcs):
            neigh_bcs = np.array(neigh_dict[bc])
            neigh_indices = np.array(
                [np.where(spot_bcs == n_bc)[0][0] for n_bc in neigh_bcs]
            )
            neighbourhood_bcs.append((bc, neigh_bcs))
            neighbourhood_indices.append((i, neigh_indices))

        self.neighbourhood_bcs = neighbourhood_bcs
        self.neighbourhood_indices = neighbourhood_indices
        self.neigh_dict = neigh_dict

    # Important, granular tests related to LR scoring

    # Important, granular tests related to CCI counting
    def test_edge_retrieval_basic(self):
        """ Basic test of functionality to retrieve edges via \
                                                    get_between_spot_edge_array.
        """
        neighbourhood_bcs = self.neighbourhood_bcs
        neighbourhood_indices = self.neighbourhood_indices

        # Initialising the edge list #
        edge_list = edges.init_edge_list(neighbourhood_bcs)

        # Basic case, should populate with all edges
        neigh_bool = np.array([True] * len(neighbourhood_bcs))
        cell_data = np.array([1] * len(neighbourhood_bcs), dtype=np.float64)
        edges.get_between_spot_edge_array(
            edge_list, neighbourhood_bcs, neighbourhood_indices, neigh_bool, cell_data
        )
        edge_list = edge_list[1:]  # Remove initialisation edge

        # Getting all the edges in the neighbourhood #
        all_edges = []
        for bc in self.neigh_dict:
            neigh_bcs = self.neigh_dict[bc]
            for neigh_bc in neigh_bcs:
                if (bc, neigh_bc) not in all_edges and (neigh_bc, bc) not in all_edges:
                    all_edges.append((bc, neigh_bc))
        n_edges = len(all_edges)

        self.assertEqual(len(edge_list), n_edges)
        self.assertTrue(
            np.all([edge in all_edges or edge[::-1] in all_edges for edge in edge_list])
        )

        # Some neighbours not valid but no effect on edge list
        # No effect since though not a valid neighbour, still a valid spot #
        edge_list = edges.init_edge_list(neighbourhood_bcs)
        invalid_neighs = ["B", "E"]
        neigh_bool = np.array([bc not in invalid_neighs for bc in self.neigh_dict])
        edges.get_between_spot_edge_array(
            edge_list, neighbourhood_bcs, neighbourhood_indices, neigh_bool, cell_data
        )
        edge_list = edge_list[1:]  # Remove initialisation edge

        self.assertEqual(len(edge_list), n_edges)
        self.assertTrue(
            np.all([edge in all_edges or edge[::-1] in all_edges for edge in edge_list])
        )

        # Some neighbours not valid, effects the edge list
        # Two neighbouring spots no longer valid neighbours #
        edge_list = edges.init_edge_list(neighbourhood_bcs)
        invalid_neighs = ["B", "C"]
        neigh_bool = np.array([bc not in invalid_neighs for bc in self.neigh_dict])
        edges.get_between_spot_edge_array(
            edge_list, neighbourhood_bcs, neighbourhood_indices, neigh_bool, cell_data
        )
        edge_list = edge_list[1:]  # Remove initialisation edge

        sub_edges1 = [
            edge for edge in all_edges if edge not in [("B", "C"), ("C", "B")]
        ]
        n_subedges = len(sub_edges1)

        self.assertEqual(len(edge_list), n_subedges)
        self.assertTrue(
            np.all(
                [edge in sub_edges1 or edge[::-1] in sub_edges1 for edge in edge_list]
            )
        )

        # Middle spot not neighbour, cell type, or spot of interest
        # Removing the centre-spot as being the cell type of interest #
        neigh_bool = np.array([True] * len(neighbourhood_bcs))
        neigh_bool[0] = False
        cell_data = np.array([1] * len(neighbourhood_bcs), dtype=np.float64)
        cell_data[0] = 0
        neigh_bcs = neighbourhood_bcs[1:]
        neigh_indices = neighbourhood_indices[1:]

        edge_list = edges.init_edge_list(neighbourhood_bcs)
        edges.get_between_spot_edge_array(
            edge_list, neigh_bcs, neigh_indices, neigh_bool, cell_data
        )
        edge_list = edge_list[1:]  # Remove initialisation edge

        sub_edges2 = [edge for edge in all_edges if "A" not in edge]
        n_subedges2 = len(sub_edges2)

        self.assertEqual(len(edge_list), n_subedges2)
        self.assertTrue(
            np.all(
                [edge in sub_edges2 or edge[::-1] in sub_edges2 for edge in edge_list]
            )
        )

        # Corner spot valid neighbour, not cell type, not spot of interest
        neigh_bool = np.array([True] * len(neighbourhood_bcs))
        cell_data = np.array([1] * len(neighbourhood_bcs), dtype=np.float64)
        cell_data[1] = 0
        neigh_bcs, neigh_indices = List(), List()
        for i in range(len(cell_data)):
            if i != 1:
                neigh_bcs.append(neighbourhood_bcs[i])
                neigh_indices.append(neighbourhood_indices[i])

        edge_list = edges.init_edge_list(neighbourhood_bcs)
        edges.get_between_spot_edge_array(
            edge_list, neigh_bcs, neigh_indices, neigh_bool, cell_data
        )
        edge_list = edge_list[1:]  # Remove initialisation edge

        sub_edges3 = [edge for edge in all_edges if "B" not in edge]
        n_subedges3 = len(sub_edges3)

        self.assertEqual(len(edge_list), n_subedges3)
        self.assertTrue(
            np.all(
                [edge in sub_edges3 or edge[::-1] in sub_edges3 for edge in edge_list]
            )
        )

    def test_get_interaction_matrix(self):
        """Test getting the interaction matrix for cell type pairs."""

        # Create 0 matrix using the above annotations to create position.
        # i.e. CT1 = 0.
        cell_data = TestCCI.create_cci(CELL_TYPE_ANNOTATIONS, CELL_TYPE_LABELS)

        # Middle spot (A) is significant and expresses ligand
        sig_bool = np.array([True] + ([False] * 6))
        ligand_bool = sig_bool.copy()

        # Neighbors D, E, G express receptor
        receptor_bool = np.array([False] * len(CELL_TYPE_ANNOTATIONS))
        receptor_bool[[3, 4, 6]] = True

        # Get interaction matrix
        int_matrix = het.get_interaction_matrix(
            cell_data,
            self.neighbourhood_indices,
            CELL_TYPE_LABELS,
            sig_bool,
            ligand_bool,
            receptor_bool,
            cell_prop_cutoff=0.2,
        )

        # Expected: CT1 (A) -> CT2 (D,G): 2 interactions, CT1 -> CT1 (E): 1 interaction
        # Matrix is [CT1->CT1, CT1->CT2, CT1->CT3, CT2->CT1, ...]
        self.assertEqual(int_matrix.shape, (3, 3))
        self.assertEqual(int_matrix[0, 0], 1)  # CT1 -> CT1 (A->E)
        self.assertEqual(int_matrix[0, 1], 2)  # CT1 -> CT2 (A->D, A->G)
        self.assertEqual(int_matrix[0, 2], 0)  # CT1 -> CT3 None

    @staticmethod
    def create_cci(cell_annotations: list[str], unique_cell_type_labels):
        cell_data = np.zeros(
            (len(cell_annotations), len(unique_cell_type_labels)), dtype=np.float64
        )
        for i, annot in enumerate(cell_annotations):
            ct_index = np.where(unique_cell_type_labels == annot)[0][0]
            cell_data[i, ct_index] = 1
        return cell_data

    # TODO next things to test:
    #   1. Getting the LR scores.


class TestRunCCILabels(unittest.TestCase):
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
            adata, lrs, min_spots=5, distance=None, n_pairs=200, n_cpus=1, verbose=False
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
