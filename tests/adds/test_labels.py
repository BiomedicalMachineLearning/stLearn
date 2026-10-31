"""Tests for st.add.labels: label transfer results matched to spots by barcode."""

import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd
from anndata import AnnData

import stlearn as st

LABELS = {"AAAC-1": "T cell", "CCCA-1": "B cell", "GGGT-1": "Tumour"}


def label_file(path: pathlib.Path, barcodes) -> str:
    """A Seurat label transfer table, rows in the order of barcodes."""
    predicted = [LABELS.get(barcode, "Stroma") for barcode in barcodes]
    table = pd.DataFrame(
        {
            "predicted.id": predicted,
            "prediction.score.T cell": [float(p == "T cell") for p in predicted],
            "prediction.score.Tumour": [float(p == "Tumour") for p in predicted],
            "prediction.score.max": 1.0,
        },
        index=list(barcodes),
    )
    table.to_csv(path / "labels.tsv", sep="\t")
    return str(path / "labels.tsv")


def spots() -> AnnData:
    adata = AnnData(np.ones((len(LABELS), 1)))
    adata.obs_names = list(LABELS)
    return adata


class TestLabels(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_rows_in_another_order(self):
        """The same spots as adata, in another order, land on their own spots."""
        adata = spots()
        st.add.labels(adata, label_file(self.path, ["GGGT-1", "AAAC-1", "CCCA-1"]))

        self.assertEqual(adata.obs["predictions"].astype(str).to_dict(), LABELS)
        scores = adata.uns["label_transfer"]
        self.assertEqual(list(scores.index), list(adata.obs_names))
        np.testing.assert_array_equal(
            scores["prediction.score.Tumour"], [0.0, 0.0, 1.0]
        )
        self.assertEqual(
            list(adata.obs["predictions"].cat.categories),
            ["B cell", "T cell", "Tumour"],
        )

    def test_extra_rows(self):
        """Rows for spots not in adata, e.g. ones filtered out, are left out."""
        adata = spots()
        rows = ["AAAC-1", "CCCA-1", "GGGT-1", "TTTT-1"]
        st.add.labels(adata, label_file(self.path, rows))

        self.assertEqual(adata.obs["predictions"].astype(str).to_dict(), LABELS)
        self.assertEqual(list(adata.uns["label_transfer"].index), list(adata.obs_names))
        self.assertNotIn("Stroma", adata.obs["predictions"].cat.categories)

    def test_missing_spots(self):
        """Spots without a row are left unlabelled, with a warning."""
        adata = spots()
        with self.assertWarnsRegex(UserWarning, "1 of 3 spots have no row"):
            st.add.labels(adata, label_file(self.path, ["GGGT-1", "AAAC-1"]))
        self.assertTrue(pd.isna(adata.obs.loc["CCCA-1", "predictions"]))
        self.assertEqual(adata.obs.loc["GGGT-1", "predictions"], "Tumour")
        self.assertTrue(adata.uns["label_transfer"].loc["CCCA-1"].isna().all())

    def test_no_barcodes_match(self):
        """R's AAAC.1 for AAAC-1 matches nothing, which is an error."""
        rows = [barcode.replace("-", ".") for barcode in LABELS]
        with self.assertRaisesRegex(ValueError, "None of adata's barcodes"):
            st.add.labels(spots(), label_file(self.path, rows))


if __name__ == "__main__":
    unittest.main()
