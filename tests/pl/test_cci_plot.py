#!/usr/bin/env python

"""Tests for cci_plot."""

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData

import stlearn as st

CELL_TYPES = ["CT1", "CT2"]


def one_interaction() -> tuple[AnnData, AnnData]:
    """Two spots, CT1 and CT2, and an LR table whose interaction counts have
    rows as senders: a single CT1 -> CT2 interaction."""
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(CELL_TYPES, categories=CELL_TYPES)},
        index=["spot0", "spot1"],
    )
    adata = AnnData(obs=obs)
    lr = AnnData(obs=obs.copy(), var=pd.DataFrame(index=["LigA_RecA"]))
    lr.layers["lr_scores"] = scipy.sparse.csc_matrix((2, 1))
    lr.uns["lr_cci_cell_type"] = pd.DataFrame(
        [[0, 1], [0, 0]], index=CELL_TYPES, columns=CELL_TYPES
    )
    return adata, lr


def tick_text(axis, position) -> str:
    """The label of the tick at `position` on `axis` (ax.xaxis or ax.yaxis)."""
    ticks = dict(zip(axis.get_ticklocs(), axis.get_ticklabels(), strict=True))
    return ticks[position].get_text()


class TestCciPlot(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    """Issue #309, #363: the CT1 -> CT2 square is on the axis labelled Sender at
    CT1 and on the axis labelled Receiver at CT2."""

    def test_axes_are_labelled_sender_and_receiver(self):
        adata, lr = one_interaction()
        ax = st.pl.cci_map(adata, "cell_type", lr_results=lr)

        squares = ax.collections[0]
        x, y = squares.get_offsets()[np.argmax(squares.get_sizes())]
        shown = {
            ax.get_xlabel(): tick_text(ax.xaxis, x),
            ax.get_ylabel(): tick_text(ax.yaxis, y),
        }

        self.assertEqual(shown, {"Sender": "CT1", "Receiver": "CT2"})


if __name__ == "__main__":
    unittest.main()
