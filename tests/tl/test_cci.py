#!/usr/bin/env python

"""Neighbourhood and counting helpers on a toy seven-spot neighbourhood."""

import unittest
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

import stlearn.tl.cci.het as het
from stlearn.tl.cci import get_edges

# Per line - which cells to annotate
CELL_TYPE_ANNOTATIONS = ["CT1", "CT2", "CT3", "CT2", "CT1", "CT3", "CT2"]

# 3 cell types: A,E -> CT1, B,G -> CT2, C,F -> CT3
CELL_TYPE_LABELS = np.array(["CT1", "CT2", "CT3"])


class TestCCI(unittest.TestCase):
    """Tests for `stlearn` CCI capability."""

    all_edges: list[Any]

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
        position = {bc: i for i, bc in enumerate(spot_bcs)}
        lengths = [len(neigh_dict[bc]) for bc in spot_bcs]
        self.indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        self.indices = np.array(
            [position[n] for bc in spot_bcs for n in neigh_dict[bc]], dtype=np.int64
        )
        self.adata = AnnData(
            X=np.zeros((len(spot_bcs), 1)), obs=pd.DataFrame(index=spot_bcs)
        )
        self.adata.obsp["spot_neighbours"] = scipy.sparse.csr_matrix(
            (np.ones(len(self.indices), np.int8), self.indices, self.indptr),
            shape=(len(spot_bcs), len(spot_bcs)),
        )
        self.spot_bcs = spot_bcs
        self.neigh_dict = neigh_dict
        # Every edge of the neighbourhood, each pair of spots once.
        self.all_edges = []
        for bc, neigh_bcs in neigh_dict.items():
            for neigh_bc in neigh_bcs:
                if (neigh_bc, bc) not in self.all_edges:
                    self.all_edges.append((bc, neigh_bc))

    def _bools(self, excluded=()):
        return np.array([bc not in excluded for bc in self.spot_bcs])

    def assertSameEdges(self, edges, expected):
        """The same pairs of spots, either way round."""
        self.assertEqual(len(edges), len(expected))
        self.assertTrue(all(e in expected or e[::-1] in expected for e in edges))

    # Important, granular tests related to LR scoring

    # Important, granular tests related to CCI counting
    def test_edges(self):
        """get_edges draws, forward, from each significant spot with the ligand
        to its neighbours with the receptor, and reverse the other way round,
        each pair of spots once."""
        everywhere = self._bools()
        forward, reverse = get_edges(self.adata, everywhere, everywhere, everywhere)
        self.assertSameEdges(forward, self.all_edges)
        self.assertSameEdges(reverse, self.all_edges)
        self.assertEqual(len(self.all_edges), 12)
        # Each edge once, the way round it was first found: A first, then B, ...
        self.assertEqual(forward[:6], [("A", n) for n in self.neigh_dict["A"]])
        self.assertEqual(forward[6], ("B", "G"))

        # B and E without the receptor: their edges still come from their own
        # neighbourhoods, where they are the significant ligand spot.
        forward, _ = get_edges(self.adata, everywhere, self._bools("BE"), everywhere)
        self.assertSameEdges(forward, self.all_edges)

        # Neighbouring B and C without the receptor: only their edge is lost.
        forward, _ = get_edges(self.adata, everywhere, self._bools("BC"), everywhere)
        self.assertSameEdges(
            forward, [e for e in self.all_edges if set(e) != {"B", "C"}]
        )

        # The middle spot neither significant nor with the receptor: the ring.
        no_a = self._bools("A")
        forward, _ = get_edges(self.adata, everywhere, no_a, no_a)
        self.assertSameEdges(forward, [e for e in self.all_edges if "A" not in e])

        # Forward and reverse differ when the ligand and receptor spots do.
        only_a, not_a = ~no_a, no_a
        forward, reverse = get_edges(self.adata, only_a, not_a, everywhere)
        self.assertEqual(forward, [("A", n) for n in self.neigh_dict["A"]])
        self.assertSameEdges(reverse, [e for e in self.all_edges if "A" in e])
        self.assertEqual(reverse[0], ("B", "A"))

        # Nothing significant: no edges, and lists all the same.
        self.assertEqual(
            get_edges(self.adata, everywhere, everywhere, ~everywhere), [[], []]
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
            self.indptr,
            self.indices,
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
