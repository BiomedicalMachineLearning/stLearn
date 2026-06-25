import unittest
from types import SimpleNamespace

import networkx
import numpy
import pandas

from stlearn.spatial.trajectory.pseudotime import (
    node_pseudotime_summary,
    orient_by_pseudotime,
)


class TestPseudotime(unittest.TestCase):

    @staticmethod
    def new_graph(edges, n):
        graph = networkx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(edges)
        return graph

    @staticmethod
    def make_adata(labels, pseudotime):
        obs = pandas.DataFrame(
            {
                "leiden": pandas.Categorical([str(x) for x in labels]),
                "dpt_pseudotime": numpy.array(pseudotime, dtype=float),
            },
        )
        return SimpleNamespace(obs=obs)

    def test_inf_cluster_drops_its_edges(self):
        graph = TestPseudotime.new_graph([(0, 1), (1, 2)], 3)
        d = orient_by_pseudotime(graph, {0: 0.0, 1: 0.5, 2: float("inf")})
        assert networkx.is_directed_acyclic_graph(d)
        assert set(d.edges) == {(0, 1)}

    def test_chain_orients_lower_to_higher(self):
        graph = TestPseudotime.new_graph([(0, 1), (1, 2), (2, 3)], 4)
        directed_graph = orient_by_pseudotime(graph, {0: 0.0, 1: 0.25, 2: 0.5, 3: 1.0})
        assert networkx.is_directed_acyclic_graph(directed_graph)
        assert set(directed_graph.edges) == {(0, 1), (1, 2), (2, 3)}

    def test_tie_yields_single_arc_not_two_cycle(self):
        graph = TestPseudotime.new_graph([(0, 1)], 2)
        directed_graph = orient_by_pseudotime(graph, {0: 0.5, 1: 0.5})
        assert networkx.is_directed_acyclic_graph(directed_graph)
        assert directed_graph.number_of_edges() == 1

    def test_nan_cluster_drops_its_edges(self):
        graph = TestPseudotime.new_graph([(0, 1), (1, 2)], 3)
        new_graph = orient_by_pseudotime(graph, {0: 0.0, 1: 0.5, 2: float("nan")})
        assert networkx.is_directed_acyclic_graph(new_graph)
        assert set(new_graph.edges) == {(0, 1)}

    # Test never cycles with ties, NaN, etc.
    def test_orientation_is_always_acyclic(self):
        """Property test: ties + NaN + arbitrary connectivity must never cycle."""
        rng = numpy.random.default_rng(0)
        for _ in range(500):
            n = int(rng.integers(3, 8))
            random_node = rng.random((n, n))
            random_node = (random_node + random_node.T) / 2
            random_node[random_node < 0.5] = 0.0
            numpy.fill_diagonal(random_node, 0.0)
            graph = networkx.from_numpy_array(random_node)
            vals = rng.choice([0.0, 0.0, 0.5, 1.0, float("nan")], size=n)
            directed_graph = orient_by_pseudotime(
                graph, {i: float(vals[i]) for i in range(n)},
            )
            assert networkx.is_directed_acyclic_graph(directed_graph)

    # a stray-inf cluster survives orientation under the summary, vanishes under .max()
    def test_summary_keeps_cluster_that_broken_drops(self):
        adata = TestPseudotime.make_adata(
            ["0", "0", "1", "1", "2", "2"],
            [0.0, 0.1, 0.5, float("inf"), 0.9, 0.95],
        )
        chain = networkx.Graph()
        chain.add_nodes_from([0, 1, 2])
        chain.add_edges_from([(0, 1), (1, 2)])

        summary = node_pseudotime_summary(adata, chain, "dpt_pseudotime", "leiden")
        # Will be zero if broken.
        assert orient_by_pseudotime(chain, summary).number_of_edges() == 2
