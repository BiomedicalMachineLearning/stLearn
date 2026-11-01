#!/usr/bin/env python

"""Tests for ClusterPlot."""

import unittest

import pandas

from stlearn.adds import row_annotations
from tests.utils import path_for_test_data, read_test_data


class TestRowAnnotations(unittest.TestCase):
    """Tests for row annotations."""

    @classmethod
    def setUpClass(cls):
        cls._base_adata = read_test_data()
        cls.annotations_path = (
            f"{path_for_test_data()}/" + "v1_human_breast_cancer_block_a_section_1.csv"
        )
        cls.proportions_path = (
            f"{path_for_test_data()}/"
            + "v1_human_breast_cancer_block_a_section_1_proportions.csv"
        )

    def setUp(self):
        """Set up test data with known clusters."""
        self.adata = self.__class__._base_adata.copy()

    def test_add_row_annotations(self):
        row_annotations.row_annotations(
            self.adata, self.__class__.annotations_path, "ID"
        )

        # 315 of the 316 test spots are in the file.
        obs = self.adata.obs
        assert obs["annot_type"].notna().sum() == 315
        assert obs["fine_annot_type"].notna().sum() == 315
        assert obs["annot_type"]["AAAGGCTACGGACCAT-1"] == "Tumor"
        assert obs["fine_annot_type"]["AAAGGCTACGGACCAT-1"] == "DCIS/LCIS_4"

    def test_add_row_annotations_with_missing_column(self):
        with self.assertRaises(ValueError):
            row_annotations.row_annotations(
                self.adata,
                self.__class__.annotations_path,
                join_column="nonexistent_column",
            )

    def test_add_row_proportions(self):
        row_annotations.row_annotations_proportions(
            self.adata, self.__class__.proportions_path
        )

        assert "cell_type" in self.adata.obs.columns

        # Check annotated is correct
        print(len(self.adata.obs["cell_type"]))
        assert self.adata.obs["cell_type"]["AAGCTCTTTCATGGTG-1"] == "Plasmablasts"

    def _annotated_by_join_column(self):
        """Annotations added from the file, joined on its "ID" column."""
        adata = self.__class__._base_adata.copy()
        row_annotations.row_annotations(adata, self.__class__.annotations_path, "ID")
        return adata

    def test_add_row_annotations_from_indexed_dataframe(self):
        """A DataFrame already indexed by spot ID is joined on that index."""
        annotations = pandas.read_csv(self.__class__.annotations_path, index_col="ID")
        row_annotations.row_annotations(self.adata, annotations)

        expected = self._annotated_by_join_column()
        for column in ("annot_type", "fine_annot_type"):
            assert column in self.adata.obs.columns
            pandas.testing.assert_series_equal(
                self.adata.obs[column], expected.obs[column]
            )

    def test_add_row_annotations_from_path_without_join_column(self):
        """Without a join column, a file's first column holds the spot IDs."""
        row_annotations.row_annotations(self.adata, self.__class__.annotations_path)

        expected = self._annotated_by_join_column()
        for column in ("annot_type", "fine_annot_type"):
            pandas.testing.assert_series_equal(
                self.adata.obs[column], expected.obs[column]
            )

    def test_add_row_proportions_from_indexed_dataframe(self):
        """Proportions in a DataFrame indexed by spot ID give the same labels."""
        proportions = pandas.read_csv(self.__class__.proportions_path, index_col=0)
        row_annotations.row_annotations_proportions(self.adata, proportions)

        assert self.adata.obs["cell_type"]["AAGCTCTTTCATGGTG-1"] == "Plasmablasts"
