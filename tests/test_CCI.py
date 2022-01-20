#!/usr/bin/env python

"""Tests for `stlearn` package."""

import os

import unittest

import numpy as np
from numba.typed import List

import stlearn as st
import scanpy as sc
from tests.utils import read_test_data

import stlearn.tools.microenv.cci.het_helpers as het_hs
import stlearn.tools.microenv.cci.het as het

global adata
adata = read_test_data()


class TestCCI(unittest.TestCase):
    """Tests for `stlearn` CCI capability."""

    def setUp(self) -> None:
        """Setup some basic test-cases as sanity checks."""

        ##### Unit neighbourhood, containing just 1 spot and 6 neighbours ######
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

    ##### Basic tests #######
    def test_load_lrs(self):
        """Testing loading lr database."""
        sizes = [2293, 4071]  # lit lr db size, putative lr db size.
        lrs = st.tl.cci.load_lrs(["connectomeDB2020_lit"])
        self.assertEqual(len(lrs), sizes[0])

        lrs = st.tl.cci.load_lrs(["connectomeDB2020_put"])
        self.assertEqual(len(lrs), sizes[1])

        lrs = st.tl.cci.load_lrs(["connectomeDB2020_lit", "connectomeDB2020_put"])
        self.assertEqual(len(lrs), sizes[1])

        lrs = st.tl.cci.load_lrs()
        self.assertEqual(len(lrs), sizes[0])

        ### Testing loading mouse as species ####
        lrs = st.tl.cci.load_lrs(species="mouse")
        genes1 = [lr_.split("_")[0] for lr_ in lrs]
        genes2 = [lr_.split("_")[1] for lr_ in lrs]
        self.assertTrue(np.all([gene[0].isupper() for gene in genes1]))
        self.assertTrue(np.all([gene[1:] == gene[1:].lower() for gene in genes1]))
        self.assertTrue(np.all([gene[0].isupper() for gene in genes2]))
        self.assertTrue(np.all([gene[1:] == gene[1:].lower() for gene in genes2]))

    ####### Important, granular tests related to LR scoring #########

    ###### Important, granular tests related to CCI counting #######
    def test_edge_retrieval_basic(self):
        """ Basic test of functionality to retrieve edges via \
                                                    get_between_spot_edge_array.
        """
        neighbourhood_bcs = self.neighbourhood_bcs
        neighbourhood_indices = self.neighbourhood_indices

        # Initialising the edge list #
        edge_list = het_hs.init_edge_list(neighbourhood_bcs)

        ############# Basic case, should populate with all edges ###############
        neigh_bool = np.array([True] * len(neighbourhood_bcs))
        cell_data = np.array([1] * len(neighbourhood_bcs), dtype=np.float)
        het_hs.get_between_spot_edge_array(
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

        ########### Some neighbours not valid but no effect on edge list #######
        # No effect since though not a valid neighbour, still a valid spot #
        edge_list = het_hs.init_edge_list(neighbourhood_bcs)
        invalid_neighs = ["B", "E"]
        neigh_bool = np.array([bc not in invalid_neighs for bc in self.neigh_dict])
        het_hs.get_between_spot_edge_array(
            edge_list, neighbourhood_bcs, neighbourhood_indices, neigh_bool, cell_data
        )
        edge_list = edge_list[1:]  # Remove initialisation edge

        self.assertEqual(len(edge_list), n_edges)
        self.assertTrue(
            np.all([edge in all_edges or edge[::-1] in all_edges for edge in edge_list])
        )

        ########### Some neighbours not valid, effects the edge list ###########
        # Two neighbouring spots no longer valid neighbours #
        edge_list = het_hs.init_edge_list(neighbourhood_bcs)
        invalid_neighs = ["B", "C"]
        neigh_bool = np.array([bc not in invalid_neighs for bc in self.neigh_dict])
        het_hs.get_between_spot_edge_array(
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

        ######### Middle spot not neighbour, cell type, or spot of interest ####
        # Removing the centre-spot as being the cell type of interest #
        neigh_bool = np.array([True] * len(neighbourhood_bcs))
        neigh_bool[0] = False
        cell_data = np.array([1] * len(neighbourhood_bcs), dtype=np.float)
        cell_data[0] = 0
        neigh_bcs = neighbourhood_bcs[1:]
        neigh_indices = neighbourhood_indices[1:]

        edge_list = het_hs.init_edge_list(neighbourhood_bcs)
        het_hs.get_between_spot_edge_array(
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

        ### Corner spot valid neighbour, not cell type, not spot of interest ###
        neigh_bool = np.array([True] * len(neighbourhood_bcs))
        cell_data = np.array([1] * len(neighbourhood_bcs), dtype=np.float)
        cell_data[1] = 0
        neigh_bcs, neigh_indices = List(), List()
        for i in range(len(cell_data)):
            if i != 1:
                neigh_bcs.append(neighbourhood_bcs[i])
                neigh_indices.append(neighbourhood_indices[i])

        edge_list = het_hs.init_edge_list(neighbourhood_bcs)
        het_hs.get_between_spot_edge_array(
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

    def test_get_interactions(self):
        """Basic test of capability to get interactions betwen cell types when \
            considering spots of a particular cell type expressing a ligand, \
            and spots of another cell type expressing the receptor.
        """

        ####### Case 1 ######
        """ Middle spot only spot of interest.
            Cell type 1, 2, or 3.
            Middle spot expresses ligand.
            3 neighbours express receptor:
                * One is cell type 1, two are cell type 2.
        """
        cell_annots = [1, 2, 3, 2, 1, 3, 2]
        cell_data = np.zeros((len(cell_annots), 3), dtype=np.float)
        for i, annot in enumerate(cell_annots):
            cell_data[i, annot - 1] = 1
        all_set = np.array([str(i) for i in range(1, 4)])
        sig_bool = np.array([True] + ([False] * (len(cell_annots) - 1)))
        l_bool = sig_bool
        r_bool = np.array([False] * len(cell_annots))
        r_bool[[3, 4, 6]] = True

        # NOTE that format of output is an edge list for each celltype-celltype
        # interaction, where edge list represents interactions between:
        #   1-1, 1-2, 1-3, 2-1, 2-2, 2-3, 3-1, 3-2, 3-3
        n_ccis_comp = 3 * 3
        # one interaction between cell type one and itself,
        # two interactions between cell type one and 2, rest no interactions.
        expected_edges = [[("A", "E")], [("A", "D"), ("A", "G")]]
        [expected_edges.append([]) for i in range(n_ccis_comp - 2)]

        interaction_edges = het.get_interactions(
            cell_data,
            self.neighbourhood_bcs,
            self.neighbourhood_indices,
            all_set,
            sig_bool,
            l_bool,
            r_bool,
            0,
        )

        self.assertEqual(len(interaction_edges), len(expected_edges))
        for i in range(len(expected_edges)):
            expect_edgesi = expected_edges[i]
            observed_edgesi = interaction_edges[i]
            match_bool = [
                edge in observed_edgesi or edge[::-1] in observed_edgesi
                for edge in expect_edgesi
            ]
            self.assertEqual(len(observed_edgesi), len(expect_edgesi))
            self.assertTrue(np.all(match_bool))

    # TODO next things to test:
    #   1. Getting the interaction matrix.
    #   2. Getting the LR scores.
