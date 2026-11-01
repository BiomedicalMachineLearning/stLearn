"""Test that CCI results are written to a zarr store and read again.

SpatialData allows [A-Za-z0-9_.-] in tables for obs and var columns, obsm, layers, uns

This checks `st.tl.cci.run` works using roundtrip of small synthetic data: the
expression table and the LR table st.tl.cci.run returns, as two tables of a store.
"""

import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd
import scipy.sparse
import spatialdata
from anndata import AnnData

import stlearn as st
from stlearn.tl.cci import LR_RESULTS, lr_result
from tests.utils import CCI_LRS, cci_table


def run_cci_pipeline(adata: AnnData) -> AnnData:
    """st.tl.cci.run then st.tl.cci.run_cci on `adata`; the LR table."""
    lr_results = st.tl.cci.run(
        adata,
        CCI_LRS,
        min_spots=2,
        distance=1.5,
        n_pairs=100,
        n_cpus=1,
        use_label="cell_type",
        verbose=False,
    )
    st.tl.cci.run_cci(
        adata,
        "cell_type",
        lr_results=lr_results,
        min_spots=1,
        n_perms=20,
        n_cpus=1,
        verbose=False,
    )
    return lr_results


def write_and_read(tables: dict[str, AnnData], folder: str) -> dict[str, AnnData]:
    """`tables` as the tables of a SpatialData written to a new store, read back."""
    path = pathlib.Path(folder) / "cci.zarr"
    spatialdata.SpatialData(tables=tables).write(path)
    return dict(spatialdata.read_zarr(path).tables)


def assert_same(testcase: unittest.TestCase, expected, actual, where: str) -> None:
    """`actual` equals `expected`: DataFrames, arrays, sparse matrices, scalars and
    dicts of them, compared all the way down."""
    if isinstance(expected, dict):
        testcase.assertIsInstance(actual, dict, msg=where)
        testcase.assertEqual(sorted(actual), sorted(expected), msg=where)
        for key, value in expected.items():
            assert_same(testcase, value, actual[key], f"{where}[{key!r}]")
    elif isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, obj=where)
    elif scipy.sparse.issparse(expected):
        testcase.assertTrue(scipy.sparse.issparse(actual), msg=where)
        np.testing.assert_array_equal(
            actual.toarray(), expected.toarray(), err_msg=where
        )
    else:
        np.testing.assert_array_equal(
            np.asarray(actual), np.asarray(expected), err_msg=where
        )


class TestCCIRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adata = cci_table()
        cls.lr = run_cci_pipeline(cls.adata)
        cls.tables = {"table": cls.adata, "cci_lr": cls.lr}

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name

    def test_every_lr_result(self):
        """Each of st.tl.cci.run's per-spot results is stored under its own key,
        not recomputed on reading, and reads back unchanged."""
        read = write_and_read(self.tables, self.folder)
        lr = read["cci_lr"]
        for key in LR_RESULTS:
            with self.subTest(key=key):
                self.assertIn(key, lr.layers)
                np.testing.assert_array_equal(
                    lr_result(lr, key), lr_result(self.lr, key)
                )
        pd.testing.assert_frame_equal(lr.var, self.lr.var)

    def test_every_key(self):
        """Both tables read back unchanged: neighbours and cell type counts in the
        expression table; the summary, LR results and run_cci's per-LR tables in
        the LR table."""
        read = write_and_read(self.tables, self.folder)
        for name, table in self.tables.items():
            for slot in ("obs", "var"):
                with self.subTest(table=name, slot=slot):
                    pd.testing.assert_frame_equal(
                        getattr(read[name], slot), getattr(table, slot)
                    )
            for slot in ("obsm", "layers", "uns"):
                written, back = getattr(table, slot), getattr(read[name], slot)
                self.assertEqual(sorted(back), sorted(written), msg=f"{name}.{slot}")
                for key in written:
                    with self.subTest(table=name, slot=slot, key=key):
                        assert_same(self, written[key], back[key], f"{slot}[{key}]")

    def test_run_cci_after_reading(self):
        """run_cci on the tables read back gives what it gives on the originals:
        the neighbours and LR results it reads from the store are usable as
        stored."""
        before = {"table": self.adata.copy(), "cci_lr": self.lr.copy()}
        read = write_and_read(self.tables, self.folder)
        for tables in (before, read):
            st.tl.cci.run_cci(
                tables["table"],
                "cell_type",
                lr_results=tables["cci_lr"],
                min_spots=1,
                n_perms=20,
                n_cpus=1,
                verbose=False,
            )
        pd.testing.assert_frame_equal(read["cci_lr"].var, before["cci_lr"].var)
        for key in ("lr_cci_cell_type", "lr_cci_raw_cell_type"):
            with self.subTest(uns=key):
                pd.testing.assert_frame_equal(
                    read["cci_lr"].uns[key], before["cci_lr"].uns[key]
                )

    def test_checkpoint_saves_every_lr_result(self):
        """st.io.checkpoint saves the LR table whole: nothing is dropped to make
        it a legal zarr table."""
        path = pathlib.Path(self.folder) / "store.zarr"
        spatialdata.SpatialData(tables={"table": cci_table()}).write(path)
        sdata = spatialdata.read_zarr(path)

        st.io.checkpoint(sdata, "cci_lr", lambda: self.lr.copy())

        read = spatialdata.read_zarr(path).tables["cci_lr"]
        for key in LR_RESULTS:
            with self.subTest(key=key):
                self.assertIn(key, read.layers)
                assert_same(self, self.lr.layers[key], read.layers[key], key)


if __name__ == "__main__":
    unittest.main()
