#!/usr/bin/env python

"""Tests for the local level of the spatial trajectory."""

import unittest
from typing import ClassVar
from unittest import mock

import anndata
import numpy
import scipy

import stlearn as st
from tests.utils import read_test_data


def _bands():
    adata = read_test_data()
    adata.obs["band"] = (adata.obs["array_row"] // 10).astype(str).astype("category")
    return adata


class TestSetRoot(unittest.TestCase):
    # Expected root spot (positional index into obs) for each band.
    # computed from X
    ROOT: ClassVar[dict[str, int]] = {
        "3": 40,
        "4": 27,
        "5": 204,
        "6": 69,
    }
    # computed from adata.raw
    RAW_ROOT: ClassVar[dict[str, int]] = {
        "3": 205,
        "4": 242,
        "5": 47,
        "6": 217,
    }

    def setUp(self):
        self.adata = _bands()

    def test_root_raw(self):
        positive = scipy.sparse.csr_matrix(numpy.clip(self.adata.X, 0, None))
        self.adata.raw = anndata.AnnData(
            positive, obs=self.adata.obs, var=self.adata.var
        )
        self.assertRoots(self.RAW_ROOT, use_raw=True)

    def assertRoots(self, expected, **kwargs):
        with numpy.errstate(divide="ignore", invalid="ignore"):
            for band, root in expected.items():
                with self.subTest(band=band):
                    got = st.spatial.trajectory.set_root(
                        self.adata, "band", band, **kwargs
                    )
                    self.assertEqual(got, root)
                    self.assertEqual(self.adata.obs["band"].iloc[got], band)

    def test_root_dense(self):
        self.assertRoots(self.ROOT)

    def test_root_sparse(self):
        self.adata.X = scipy.sparse.csr_matrix(self.adata.X)
        self.assertRoots(self.ROOT)

    def test_copies_nothing(self):
        with mock.patch.object(anndata.AnnData, "copy", autospec=True) as copy:
            st.spatial.trajectory.set_root(_bands(), use_label="band", cluster="3")
        self.assertEqual(copy.call_count, 0)
