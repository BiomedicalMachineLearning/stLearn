#!/usr/bin/env python

"""Tests for ClusterPlot."""

import unittest

from stlearn.adds import row_annotations
from tests.utils import read_test_data, test_data_path


class TestRowAnnotations(unittest.TestCase):
    """Tests for row annotations."""

    @classmethod
    def setUpClass(cls):
        cls._base_adata = read_test_data()
        cls.annotations_path = (
            f"{test_data_path()}/" + "v1_human_breast_cancer_block_a_section_1.csv"
        )


def setUp(self):
    """Set up test data with known clusters."""
    self.adata = self.__class__._base_adata.copy()


def test_add_row_annotations(self):
    row_annotations.row_annotations(self.adata, self.__class__.annotations_path, "ID")

    assert "annot_type" in self.adata.obs.columns
    assert "fine_annot_type" in self.adata.obs.columns

    # Check annotated the same number
    annotated = self.adata.obs["annot_type"].dropna()
    fine_annotated = self.adata.obs["fine_annot_type"].dropna()
    assert len(fine_annotated) == len(annotated)


def test_add_row_annotations_with_missing_column(self):
    with self.assertRaises(ValueError):
        row_annotations.row_annotations(
            self.adata,
            self.__class__.annotations_path,
            join_column="nonexistent_column",
        )
