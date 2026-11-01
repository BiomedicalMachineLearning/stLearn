#!/usr/bin/env python

"""Tests for `stlearn` package."""

import tempfile
import unittest
from collections.abc import Mapping
from unittest import mock

import anndata
import networkx
import numpy
from matplotlib import pyplot
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch

import stlearn as st
from stlearn.pl.utils import get_colors
from stlearn.utils import _read_graph
from tests.utils import read_test_data

FORMAT = "zarr"

# Cluster labels like the tutorial's annotations: commas, brackets and a slash.
DCIS = "Invasive Cancer, Fibrous Tissue (DCIS)"
IDC = "Invasive Cancer, Immune/lymphoid (IDC)"


def _non_string_keys(value, path="uns"):
    """Paths to every dict key below `value` that is not a str."""
    found = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}[{key!r}]"
            if not isinstance(key, str):
                found.append(child_path)
            found.extend(_non_string_keys(child, child_path))
    return found


def _weighted_edges(adata):
    graph = _read_graph(adata, "PTS_graph")
    return {(u, v): round(w, 10) for u, v, w in graph.edges(data="weight")}


def _write_and_read(adata, path):
    if path.endswith(".zarr"):
        adata.write_zarr(path)
        return anndata.read_zarr(path)
    adata.write_h5ad(path)
    return anndata.read_h5ad(path)


class TestPSTS(unittest.TestCase):
    """Pseudo-time-space trajectories on the test data.

    The fixture has two Leiden clusters, "0" and "1", with the root spot in "0".
    The shortest-path tests build their own pseudotimes rather than use the
    fixture's: which cluster holds the fixture's latest spot depends on library
    versions (it moved from "0" to "1" with umap-learn 0.5.12).
    """

    @classmethod
    def setUpClass(cls):
        adata = read_test_data()
        st.em.run_pca(adata)
        st.pp.neighbors(adata)
        st.tl.clustering.leiden(adata, resolution=0.6)
        adata.uns["iroot"] = numpy.flatnonzero(adata.obs["leiden"] == "0")[0]
        cls._clustered = adata.copy()

        st.spatial.trajectory.pseudotime(
            adata, eps=100, use_sme=False, use_label="leiden"
        )
        st.spatial.trajectory.pseudotimespace_global(
            adata, use_label="leiden", list_clusters=["0", "1"]
        )
        cls._trajectory = adata

        # The same clusters, renamed: every result should match the numbered run.
        named = cls._clustered.copy()
        named.obs["leiden"] = named.obs["leiden"].cat.rename_categories(
            {"0": DCIS, "1": IDC}
        )
        st.spatial.trajectory.pseudotime(
            named, eps=100, use_sme=False, use_label="leiden"
        )
        st.spatial.trajectory.pseudotimespace_global(
            named, use_label="leiden", list_clusters=[DCIS, IDC]
        )
        cls._named = named

    def test_psts(self):
        adata = self._trajectory.copy()
        st.spatial.trajectory.detect_transition_markers_clades(
            adata, clade="0", use_raw_count=False, cutoff_spearman=0.3
        )
        self.assertIn("clade_0", adata.uns)
        self.assertLessEqual(
            {"0_0_0", "0_0_1", "0_1_1"}, adata.uns["available_paths"].keys()
        )

    def test_trajectory_round_trip(self):
        adata = self._trajectory
        with tempfile.TemporaryDirectory() as tmp:
            loaded = _write_and_read(adata.copy(), f"{tmp}/trajectory.{FORMAT}")

            split_node = loaded.uns["split_node"]
            self.assertEqual(split_node.keys(), adata.uns["split_node"].keys())
            for key, members in adata.uns["split_node"].items():
                self.assertEqual(list(split_node[key]), list(members))
            for graph in ("global_graph", "PTS_graph"):
                numpy.testing.assert_array_equal(
                    loaded.uns[graph]["nodes"], adata.uns[graph]["nodes"]
                )
            self.assertEqual(
                loaded.uns["centroid_dict"].keys(),
                adata.uns["centroid_dict"].keys(),
            )
            for key, centroid in adata.uns["centroid_dict"].items():
                numpy.testing.assert_allclose(
                    loaded.uns["centroid_dict"][key], centroid
                )

    def test_uns_keys_are_strings(self):
        self.assertEqual(_non_string_keys(self._trajectory.uns), [])

    def test_ids_match_obs(self):
        adata = self._trajectory
        labels = set(adata.obs["leiden"].cat.categories)
        subclusters = set(adata.obs["sub_cluster_labels"].cat.categories)

        # split_node is keyed by the clusters' positions, never by their labels.
        positions = {str(i) for i in range(len(labels))}
        self.assertEqual(set(adata.uns["split_node"]), positions)
        self.assertNotIn("leiden_index_dict", adata.uns)
        self.assertEqual(set(adata.uns["centroid_dict"]), subclusters)
        for members in adata.uns["split_node"].values():
            self.assertLessEqual(set(members), subclusters)
        # available_paths is keyed by positions too, and holds cluster labels.
        self.assertIn("0_0_1", adata.uns["available_paths"])
        for path in adata.uns["available_paths"].values():
            self.assertLessEqual(set(path), labels)

    def test_graph_nodes_are_subcluster_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = _write_and_read(self._trajectory.copy(), f"{tmp}/t.{FORMAT}")
        for name, adata in (("in memory", self._trajectory), ("loaded", loaded)):
            with self.subTest(name):
                split_node = adata.uns["split_node"]
                kept = {sub for members in split_node.values() for sub in members}
                self.assertEqual(set(_read_graph(adata, "global_graph")), kept)

                # The trajectory's root points at the first cluster's subclusters.
                trajectory = _read_graph(adata, "PTS_graph")
                self.assertLessEqual(set(trajectory) - {"root"}, kept)
                self.assertEqual(
                    set(trajectory.successors("root")), set(split_node["0"])
                )

    def test_local_trajectory(self):
        adata = self._trajectory.copy()
        st.spatial.trajectory.pseudotimespace_local(
            adata, use_label="leiden", cluster="0", w=0.5
        )

        sizes = adata.obs["sub_cluster_labels"].value_counts()
        in_cluster = adata.obs.loc[adata.obs["leiden"] == "0", "sub_cluster_labels"]
        n = sum(sizes[s] > adata.uns["threshold_spots"] for s in in_cluster.unique())

        stdm = adata.uns["ST_distance_matrix"]
        dpt = adata.uns["nonabs_dpt_distance_matrix"]
        self.assertEqual(stdm.shape, (n, n))
        numpy.testing.assert_allclose(stdm, stdm.T)
        numpy.testing.assert_allclose(numpy.diag(stdm), 0)
        self.assertTrue(((stdm >= 0) & (stdm <= 1)).all())
        numpy.testing.assert_allclose(dpt, -dpt.T)

    def test_local_weight_search(self):
        adata = self._trajectory.copy()
        w = st.spatial.trajectory.weight_optimizing_local(
            adata, use_label="leiden", cluster="0"
        )
        self.assertTrue(0 <= w <= 1)
        self.assertIn("screening_result_local", adata.uns)

    def test_local_unknown_cluster(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            st.spatial.trajectory.pseudotimespace_local(
                self._trajectory.copy(), use_label="leiden", cluster="99", w=0.5
            )

    def test_local_needs_two_subclusters(self):
        adata = self._trajectory.copy()
        adata.uns["threshold_spots"] = adata.n_obs  # no subcluster is big enough
        with self.assertRaisesRegex(ValueError, "fewer than two subclusters"):
            st.spatial.trajectory.pseudotimespace_local(
                adata, use_label="leiden", cluster="0", w=0.5
            )

    def test_local_plot(self):
        adata = self._trajectory.copy()
        st.spatial.trajectory.pseudotimespace_local(
            adata, use_label="leiden", cluster="0", w=0.5
        )
        st.pl.trajectory.local_plot(
            adata, use_cluster="0", use_label="leiden", show_plot=False
        )
        fig = pyplot.gcf()
        fig.canvas.draw()  # the arrows are only projected when the figure is drawn

        ax = fig.axes[0]
        _, other_spots = ax.collections[:2]
        self.assertEqual(
            len(other_spots.get_offsets()), (adata.obs["leiden"] != "0").sum()
        )
        # One branch (arc) and one arrow for each pair of subclusters.
        n = adata.uns["ST_distance_matrix"].shape[0]
        pairs = n * (n - 1) // 2
        self.assertEqual(len(ax.lines), pairs)
        arrows = [a for a in ax.get_children() if isinstance(a, FancyArrowPatch)]
        self.assertEqual(len(arrows), pairs)
        pyplot.close(fig)

    def test_global_clusters_not_connected(self):
        adata = self._trajectory.copy()
        # Cut every edge between the two clusters' subclusters.
        graph = _read_graph(adata, "global_graph")
        first = set(adata.uns["split_node"]["0"])
        graph.remove_edges_from(
            [(u, v) for u, v in list(graph.edges) if (u in first) != (v in first)]
        )
        nodes = list(graph.nodes)
        adata.uns["global_graph"] = {
            "graph": networkx.to_scipy_sparse_array(graph, nodelist=nodes),
            "nodes": nodes,
        }
        with self.assertRaisesRegex(ValueError, "Please choose other path"):
            st.spatial.trajectory.pseudotimespace_global(
                adata, use_label="leiden", list_clusters=["0", "1"]
            )

    def test_global_cluster_without_subclusters(self):
        adata = self._trajectory.copy()
        # As if cluster "1" had no subcluster above threshold_spots.
        adata.uns["split_node"] = {**adata.uns["split_node"], "1": []}
        with self.assertRaisesRegex(ValueError, r"\['1'\] have no subclusters"):
            st.spatial.trajectory.global_level(
                adata, use_label="leiden", list_clusters=["0", "1"], w=0.5
            )

    def test_cluster_labels_not_positions(self):
        # Rename the clusters so labels and positions differ: "10" is at position 0.
        adata = self._clustered.copy()
        adata.obs["leiden"] = adata.obs["leiden"].cat.rename_categories(
            {"0": "10", "1": "11"}
        )
        st.spatial.trajectory.pseudotime(
            adata, eps=100, use_sme=False, use_label="leiden"
        )
        for clusters in (["10", "11"], ["11", "10"]):
            with self.subTest(list_clusters=clusters):
                result = adata.copy()
                st.spatial.trajectory.pseudotimespace_global(
                    result, use_label="leiden", list_clusters=clusters
                )
                self.assertEqual(
                    _weighted_edges(result), _weighted_edges(self._trajectory)
                )

    def test_cluster_ints_rejected(self):
        adata = self._trajectory.copy()
        trajectory, pl = st.spatial.trajectory, st.pl
        calls = {
            "set_root": lambda: trajectory.set_root(
                adata, use_label="leiden", cluster=0
            ),
            "pseudotimespace_global": lambda: trajectory.pseudotimespace_global(
                adata, use_label="leiden", list_clusters=[0, 1]
            ),
            "pseudotimespace_local": lambda: trajectory.pseudotimespace_local(
                adata, use_label="leiden", cluster=0, w=0.5
            ),
            "cluster_plot": lambda: pl.cluster_plot(
                adata, use_label="leiden", list_clusters=[0], show_plot=False
            ),
            "pseudotime_plot": lambda: pl.trajectory.pseudotime_plot(
                adata, use_label="leiden", list_clusters=[0], show_plot=False
            ),
            "subcluster_plot": lambda: pl.subcluster_plot(
                adata, use_label="leiden", cluster=0, show_plot=False
            ),
            "local_plot": lambda: pl.trajectory.local_plot(
                adata, use_cluster=0, use_label="leiden", show_plot=False
            ),
        }
        for name, call in calls.items():
            with self.subTest(name), self.assertRaisesRegex(TypeError, "'0' rather"):
                call()
        pyplot.close("all")

    def test_unknown_cluster_label(self):
        adata = self._trajectory.copy()
        calls = {
            "set_root": lambda: st.spatial.trajectory.set_root(
                adata, use_label="leiden", cluster="99"
            ),
            "pseudotimespace_global": lambda: (
                st.spatial.trajectory.pseudotimespace_global(
                    adata, use_label="leiden", list_clusters=["0", "99"]
                )
            ),
            "cluster_plot": lambda: st.pl.cluster_plot(
                adata, use_label="leiden", list_clusters=["99"], show_plot=False
            ),
            "pseudotime_plot": lambda: st.pl.trajectory.pseudotime_plot(
                adata, use_label="leiden", list_clusters=["99"], show_plot=False
            ),
            "subcluster_plot": lambda: st.pl.subcluster_plot(
                adata, use_label="leiden", cluster="99", show_plot=False
            ),
        }
        for name, call in calls.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, "not found"):
                call()
        pyplot.close("all")

    def test_transition_markers(self):
        # In the test data's trajectory, subcluster "0" leads to "3", so clade
        # "0" covers the same spots as the branch ["0", "3"].
        trajectory = st.spatial.trajectory
        with tempfile.TemporaryDirectory() as tmp:
            loaded = _write_and_read(self._trajectory.copy(), f"{tmp}/t.{FORMAT}")
        for name, adata in (("in memory", self._trajectory.copy()), ("loaded", loaded)):
            with self.subTest(name):
                trajectory.detect_transition_markers_clades(
                    adata, clade="0", cutoff_spearman=0.3
                )
                trajectory.detect_transition_markers_branches(
                    adata, branch=["0", "3"], cutoff_spearman=0.3
                )
                clade, branch = adata.uns["clade_0"], adata.uns["branch_0_3"]
                self.assertEqual(list(clade.columns), ["gene", "score", "p-value"])
                self.assertTrue(clade.equals(branch))

    def test_transition_markers_screening_genes(self):
        adata = self._trajectory.copy()
        genes = list(adata.var_names[:10])
        st.spatial.trajectory.detect_transition_markers_clades(
            adata, clade="0", cutoff_spearman=0.0, screening_genes=genes
        )
        self.assertLessEqual(set(adata.uns["clade_0"]["gene"]), set(genes))

    def test_transition_markers_subcluster_labels(self):
        adata = self._trajectory.copy()
        clades = st.spatial.trajectory.detect_transition_markers_clades
        branches = st.spatial.trajectory.detect_transition_markers_branches
        with self.assertRaisesRegex(TypeError, "'0' rather"):
            clades(adata, clade=0)
        with self.assertRaisesRegex(TypeError, "'0' rather"):
            branches(adata, branch=[0, 3])
        with self.assertRaisesRegex(ValueError, "not found"):
            clades(adata, clade="99")
        with self.assertRaisesRegex(ValueError, "not found"):
            branches(adata, branch=["0", "99"])
        # Subcluster "3" ends the trajectory: no clade starts there.
        with self.assertRaisesRegex(
            ValueError, r"Subcluster \'3\' does not(.*)[\'0\', \'2\']"
        ):
            clades(adata, clade="3")

    def test_cluster_plots_with_labels(self):
        for name, trajectory, clusters in (
            ("numbered", self._trajectory, ["0", "1"]),
            ("named", self._named, [DCIS, IDC]),
        ):
            with self.subTest(name):
                adata = trajectory.copy()
                st.pl.cluster_plot(
                    adata,
                    use_label="leiden",
                    list_clusters=clusters,
                    show_trajectories=True,
                    show_node=True,
                    show_plot=False,
                )
                for show_trajectories in (False, True):
                    st.pl.trajectory.pseudotime_plot(
                        adata,
                        use_label="leiden",
                        list_clusters=clusters,
                        show_trajectories=show_trajectories,
                        show_plot=False,
                    )
                st.pl.trajectory.tree_plot(adata, use_label="leiden", show_plot=False)
                st.pl.subcluster_plot(
                    adata, use_label="leiden", cluster=clusters[0], show_plot=False
                )
                pyplot.close("all")

    def test_trajectory_nodes_and_legend(self):
        # Nodes are solid dots in their cluster's colour, with no text on them;
        # the cluster labels are in a legend outside the image.
        for name, trajectory, clusters in (
            ("numbered", self._trajectory, ["0", "1"]),
            ("named", self._named, [DCIS, IDC]),
        ):
            adata = trajectory.copy()
            colors = get_colors(adata, "leiden")
            categories = list(adata.obs["leiden"].cat.categories)
            nodes, node_colors = [], []
            for cluster in clusters:
                position = categories.index(cluster)
                for subcluster in adata.uns["split_node"][str(position)]:
                    nodes.append(adata.uns["centroid_dict"][subcluster])
                    node_colors.append(to_rgba(colors[position]))

            for plot in ("cluster_plot", "pseudotime_plot"):
                with self.subTest(name, plot=plot):
                    pyplot.close("all")
                    fig, ax = pyplot.subplots()
                    if plot == "cluster_plot":
                        st.pl.cluster_plot(
                            adata,
                            use_label="leiden",
                            list_clusters=clusters,
                            show_trajectories=True,
                            show_node=True,
                            show_plot=False,
                            fig=fig,
                            ax=ax,
                        )
                    else:
                        st.pl.trajectory.pseudotime_plot(
                            adata,
                            use_label="leiden",
                            list_clusters=clusters,
                            show_trajectories=True,
                            show_plot=False,
                            ax=ax,
                        )
                    fig.canvas.draw()

                    # Everything is drawn on the given axes, in one figure.
                    self.assertEqual(pyplot.get_fignums(), [fig.number])

                    node_layer = ax.collections[-1]
                    numpy.testing.assert_allclose(node_layer.get_offsets(), nodes)
                    numpy.testing.assert_allclose(
                        node_layer.get_facecolors(), node_colors
                    )
                    self.assertEqual([text.get_text() for text in ax.texts], [])

                    legend = ax.get_legend()
                    labels = [text.get_text() for text in legend.get_texts()]
                    self.assertEqual(labels, clusters)
                    # Right of the image, and of the colour bar's tick labels.
                    left = legend.get_window_extent().x0
                    self.assertGreaterEqual(left, ax.get_window_extent().x1)
                    for colorbar in fig.axes[1:]:
                        self.assertGreaterEqual(left, colorbar.get_tightbbox().x1)
        pyplot.close("all")

    def test_cluster_labels_must_be_strings(self):
        adata = self._clustered.copy()
        adata.obs["leiden"] = adata.obs["leiden"].astype(int).astype("category")
        with self.assertRaisesRegex(TypeError, "string categories"):
            st.spatial.trajectory.set_root(adata, use_label="leiden", cluster="0")

    def test_named_clusters_match_numbered(self):
        named, numbered = self._named, self._trajectory
        labels = {"0": DCIS, "1": IDC}
        self.assertEqual(named.uns["split_node"], numbered.uns["split_node"])
        self.assertEqual(
            named.uns["available_paths"],
            {
                key: [labels[cluster] for cluster in path]
                for key, path in numbered.uns["available_paths"].items()
            },
        )
        self.assertEqual(_weighted_edges(named), _weighted_edges(numbered))

        # Given in the other order, the clusters are still ordered by pseudotime.
        reversed_order = named.copy()
        st.spatial.trajectory.pseudotimespace_global(
            reversed_order, use_label="leiden", list_clusters=[IDC, DCIS]
        )
        self.assertEqual(_weighted_edges(reversed_order), _weighted_edges(numbered))

    def test_shortest_path_same_start_and_end(self):
        # A pseudotime whose first and last spots are both in cluster "0": the
        # root stays at 0, every other spot is halved, and the latest spot of
        # cluster "0" is set to exactly 1.
        adata = self._trajectory.copy()
        obs = adata.obs
        pseudotime = obs["dpt_pseudotime"] / 2
        pseudotime.loc[obs.loc[obs["leiden"] == "0", "dpt_pseudotime"].idxmax()] = 1.0
        obs["ends_in_first"] = pseudotime
        with self.assertRaisesRegex(ValueError, "no path between clusters"):
            st.spatial.trajectory.shortest_path_spatial_paga(
                adata, use_label="leiden", key="ends_in_first"
            )

    def test_shortest_path_with_labels(self):
        # A pseudotime where every spot of cluster "1" comes after cluster "0":
        # the root stays at 0, and it is scaled so the latest spot is exactly 1,
        # as find_min_max_node needs. This tests label handling, not the choice
        # of first and last cluster.
        shortest = st.spatial.trajectory.shortest_path_spatial_paga
        results = []
        for adata in (self._trajectory.copy(), self._named.copy()):
            obs = adata.obs
            ordered = obs["leiden"].cat.codes + obs["dpt_pseudotime"]
            obs["ordered_pseudotime"] = ordered / ordered.max()
            results.append(
                shortest(adata, use_label="leiden", key="ordered_pseudotime")
            )
        numbered, named = results
        self.assertEqual(numbered, ["0", "1"])
        self.assertEqual(named, [DCIS, IDC])

    def test_named_clusters_round_trip(self):
        self.assertEqual(_non_string_keys(self._named.uns), [])
        with tempfile.TemporaryDirectory() as tmp:
            loaded = _write_and_read(self._named.copy(), f"{tmp}/named.{FORMAT}")
        self.assertEqual(_weighted_edges(loaded), _weighted_edges(self._named))

    def test_plots_do_not_copy_data(self):
        adata = self._trajectory.copy()
        with mock.patch.object(anndata.AnnData, "copy", autospec=True) as copy:
            st.pl.cluster_plot(
                adata,
                use_label="leiden",
                show_trajectories=True,
                list_clusters=["0", "1"],
                show_subcluster=True,
                show_plot=False,
            )
            st.pl.cluster_plot(adata, use_label="leiden", show_plot=False)
            st.pl.subcluster_plot(
                adata, use_label="leiden", cluster="0", show_plot=False
            )
        pyplot.close("all")
        self.assertEqual(copy.call_count, 0)
