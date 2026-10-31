#!/usr/bin/env python

"""Tests for the local level of the spatial trajectory."""

import unittest

import numpy as np

import stlearn as st
from tests.utils import read_test_data


class TestPSTSLocal(unittest.TestCase):
    """pseudotimespace_local on one cluster of the test section."""

    @classmethod
    def setUpClass(cls):
        cls._base_adata = read_test_data()

    def _run_pseudotime(self, eps=100):
        adata = self.__class__._base_adata.copy()
        st.em.run_pca(adata)
        st.pp.neighbors(adata)
        st.tl.clustering.leiden(adata, resolution=0.6)
        adata.uns["iroot"] = np.flatnonzero(adata.obs["leiden"] == "0")[0]
        st.spatial.trajectory.pseudotime(
            adata, eps=eps, use_sme=False, use_label="leiden"
        )
        return adata

    def test_psts_local(self):
        """The local level builds two square matrices over a cluster's sub-clusters."""
        adata = self._run_pseudotime()
        cluster = self._cluster_with_sub_clusters(adata, minimum=2)

        st.spatial.trajectory.pseudotimespace_local(
            adata, use_label="leiden", cluster=cluster, w=0.5
        )

        stdm = adata.uns["ST_distance_matrix"]
        dpt = adata.uns["nonabs_dpt_distance_matrix"]
        self.assertEqual(stdm.ndim, 2)
        self.assertEqual(stdm.shape[0], stdm.shape[1])
        self.assertGreaterEqual(stdm.shape[0], 2)
        self.assertEqual(dpt.shape, stdm.shape)
        # A distance matrix is symmetric with a zero diagonal and, being a weighted
        # sum of two matrices scaled to [0, 1], stays inside [0, 1].
        np.testing.assert_allclose(stdm, stdm.T)
        np.testing.assert_allclose(np.diag(stdm), 0)
        self.assertGreaterEqual(stdm.min(), 0)
        self.assertLessEqual(stdm.max(), 1)
        # Signed pseudotime differences are antisymmetric.
        np.testing.assert_allclose(dpt, -dpt.T)

    def test_psts_local_optimises_weight(self):
        """Without `w`, the weight is screened and the screening table is kept.

        Screening compares graphs across weights, which needs three sub-clusters:
        with two, every weight gives the same matrix. A smaller `eps` splits the
        test section finely enough.
        """
        adata = self._run_pseudotime(eps=50)
        cluster = self._cluster_with_sub_clusters(adata, minimum=3)

        st.spatial.trajectory.pseudotimespace_local(
            adata, use_label="leiden", cluster=cluster
        )

        self.assertGreaterEqual(adata.uns["ST_distance_matrix"].shape[0], 3)
        screening = adata.uns["screening_result_local"]
        self.assertFalse(screening["Dissmilarity_Score"].isna().any())

    @staticmethod
    def _cluster_with_sub_clusters(adata, minimum):
        """A cluster holding at least `minimum` sub-clusters big enough to be used."""
        sizes = adata.obs.groupby(
            ["leiden", "sub_cluster_labels"], observed=True
        ).size()
        usable = sizes[sizes > adata.uns["threshold_spots"]]
        per_cluster = usable.groupby(level="leiden", observed=True).size()
        candidates = per_cluster[per_cluster >= minimum]
        if candidates.empty:
            raise unittest.SkipTest(f"no cluster has {minimum} usable sub-clusters")
        return str(candidates.index[0])
