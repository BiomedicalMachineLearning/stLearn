#!/usr/bin/env python

"""Tests for het.count, the cell type counts per neighbourhood, and how
st.tl.cci.run weights LR scores by them when given use_label."""

import unittest

import numpy as np
import pandas as pd
from anndata import AnnData

import stlearn as st
from stlearn.tl.cci import lr_result
from stlearn.tl.cci.het import count, get_data_for_counting
from tests.utils import CCI_LRS, cci_table

LRS = st.tl.cci.LRPairs.from_names(
    ["Lig0_Rec0", "Lig1_Rec1"], species="human", source="test"
)
CELL_TYPES = ["A", "B", "C", "D", "E", "F"]


def _make_adata(n_side=6, n_extra_genes=40, seed=0):
    """Spots on a unit grid, with cell type proportions in uns["cell_type"]."""
    rng = np.random.default_rng(seed)
    n_spots = n_side * n_side
    var_names = list(LRS.genes()) + [f"gene{i}" for i in range(n_extra_genes)]
    X = rng.poisson(3.0, size=(n_spots, len(var_names))).astype(np.float64)

    rows, cols = np.divmod(np.arange(n_spots), n_side)
    obs = pd.DataFrame(
        {"imagerow": rows.astype(float), "imagecol": cols.astype(float)},
        index=[f"spot{i}" for i in range(n_spots)],
    )
    adata = AnnData(X=X, obs=obs, var=pd.DataFrame(index=var_names))
    adata.uns["spatial"] = {
        "lib": {
            "scalefactors": {"spot_diameter_fullres": 5.0, "tissue_hires_scalef": 0.1},
            "use_quality": "hires",
        }
    }

    # Mostly one cell type per spot, so neighbourhoods differ in their counts.
    proportions = pd.DataFrame(
        rng.dirichlet(np.ones(len(CELL_TYPES)) * 0.1, n_spots),
        index=adata.obs_names,
        columns=CELL_TYPES,
    )
    adata.uns["cell_type"] = proportions
    adata.obs["cell_type"] = proportions.idxmax(axis=1).astype("category")
    return adata


def _run(adata, **kwargs):
    """st.tl.cci.run keeping every LR, with the 8 surrounding spots as neighbours."""
    settings = dict(min_spots=0, distance=1.5, n_pairs=100, verbose=False)
    return st.tl.cci.run(adata, LRS, **(settings | kwargs))


class TestHet(unittest.TestCase):
    def test_counts_cell_types_in_neighbourhood(self):
        """Each spot counts the cell types strictly above cell_prop_cutoff in
        any spot within the distance, itself included."""
        adata = _make_adata()
        proportions = adata.uns["cell_type"].to_numpy()
        coor = adata.obs[["imagerow", "imagecol"]].to_numpy()
        rng = np.random.default_rng(1)
        # Random cutoffs, and cutoffs equal to a proportion, where > and >= differ.
        cutoffs = [*rng.uniform(0, 1, 5), *rng.choice(proportions.ravel(), 5)]
        cutoffs = [float(cutoff) for cutoff in cutoffs]
        expecteds = set()
        for cutoff in cutoffs:
            with self.subTest(cutoff=cutoff):
                count(
                    adata,
                    use_label="cell_type",
                    verbose=False,
                    distance=1,
                    cell_prop_cutoff=cutoff,
                )
                present = proportions > cutoff
                expected = [
                    present[np.linalg.norm(coor - spot, axis=1) <= 1].any(axis=0).sum()
                    for spot in coor
                ]
                np.testing.assert_array_equal(adata.obsm["cci_het"].ravel(), expected)
                expecteds.add(tuple(expected))
        self.assertGreater(len(expecteds), 2)

    def test_default_distance_is_two_spot_diameters(self):
        """With no distance, the neighbourhood is two spot diameters in image
        pixels (1.0 here), the same default as st.tl.cci.run."""
        default, explicit = _make_adata(), _make_adata()
        count(default, use_label="cell_type", verbose=False)
        count(explicit, use_label="cell_type", verbose=False, distance=1.0)
        np.testing.assert_array_equal(default.obsm["cci_het"], explicit.obsm["cci_het"])

    def test_counts_cell_types_within_spot(self):
        """With distance=0, each spot counts its own cell types above 0.2,
        matched to the spots by barcode."""
        adata = _make_adata()
        expected = (adata.uns["cell_type"].to_numpy() > 0.2).sum(axis=1)
        adata.uns["cell_type"] = adata.uns["cell_type"].iloc[::-1]
        count(adata, use_label="cell_type", verbose=False, distance=0)
        np.testing.assert_array_equal(adata.obsm["cci_het"].ravel(), expected)

    def test_obs_labels(self):
        """Single cells, with a label in obs and no proportions in uns, count
        as one cell type each, the same as one-hot proportions."""
        one_hot = _make_adata()
        one_hot.uns["cell_type"] = pd.get_dummies(one_hot.obs["cell_type"], dtype=float)
        count(one_hot, use_label="cell_type", verbose=False, distance=1)

        obs_only = _make_adata()
        del obs_only.uns["cell_type"]
        count(obs_only, use_label="cell_type", verbose=False, distance=1)

        np.testing.assert_array_equal(obs_only.obsm["cci_het"], one_hot.obsm["cci_het"])
        self.assertGreater(obs_only.obsm["cci_het"].max(), 1)

    def test_missing_labels(self):
        """count and run raise a KeyError naming a label in neither uns nor obs."""
        adata = _make_adata()
        with self.assertRaisesRegex(KeyError, "'no_such_label'"):
            count(adata, use_label="no_such_label", verbose=False)
        with self.assertRaisesRegex(KeyError, "'no_such_label'"):
            _run(adata, use_label="no_such_label")

    def test_scores_weighted_by_cell_type_counts(self):
        """With use_label, each spot's LR scores are its scores without
        use_label times its cell type count, whether the cell types are
        proportions in uns or labels in obs."""
        results = _run(_make_adata())
        unweighted = lr_result(results[:, LRS.names()], "lr_scores")
        obs_only = _make_adata()
        del obs_only.uns["cell_type"]
        one_hot = _make_adata()
        one_hot.uns["cell_type"] = pd.get_dummies(one_hot.obs["cell_type"], dtype=float)
        layouts = [
            ("proportions", _make_adata(), _make_adata()),
            ("labels", obs_only, one_hot),
        ]
        for layout, adata, expected in layouts:
            with self.subTest(layout):
                count(expected, use_label="cell_type", verbose=False, distance=1.5)
                counts = expected.obsm["cci_het"].ravel()
                self.assertGreater(counts.max(), 1)
                lr_results = _run(adata, use_label="cell_type")
                weighted = lr_result(lr_results[:, LRS.names()], "lr_scores")
                np.testing.assert_array_equal(weighted, unweighted * counts[:, None])

    def test_run_passes_on_cell_prop_cutoff(self):
        adata = _make_adata()
        _run(adata, use_label="cell_type", distance=0, cell_prop_cutoff=0.5)
        expected = (adata.uns["cell_type"].to_numpy() > 0.5).sum(axis=1)
        np.testing.assert_array_equal(adata.obsm["cci_het"].ravel(), expected)

    def test_mixture_columns_match_labels_exactly(self):
        """Ensure that with leiden style cluster number "2", "12" that we don't
        use substrings or anything but perfect matching of the labels.
        """
        adata = AnnData(obs=pd.DataFrame(index=["a", "b"]))
        adata.uns["leiden"] = pd.DataFrame(
            {"12": [1.0, 0.0], "2": [0.0, 1.0]}, index=adata.obs_names
        )
        result = get_data_for_counting(adata, "leiden", True, ["2"])
        np.testing.assert_array_equal(result[:, 0], [0.0, 1.0])

    def test_spot_mixtures_counts_cell_types_never_dominant(self):
        """With spot_mixtures, run_cci counts a cell type that is in every spot at
        0.3, above cell_prop_cutoff (0.2), but is never a spot's label; without
        spot_mixtures it only has the labels, so it can't."""
        adata = cci_table()
        proportions = adata.uns["cell_type"] * 0.7
        proportions["CT4"] = 0.3
        adata.uns["cell_type"] = proportions
        lr_results = st.tl.cci.run(
            adata, CCI_LRS, min_spots=2, distance=1.5, n_pairs=100, verbose=False
        )

        counts = {}
        for spot_mixtures in (False, True):
            lr = lr_results.copy()
            st.tl.cci.run_cci(
                adata,
                "cell_type",
                lr,
                spot_mixtures=spot_mixtures,
                min_spots=1,
                n_perms=20,
                verbose=False,
            )
            counts[spot_mixtures] = lr.uns["lr_cci_raw_cell_type"]

        self.assertNotIn("CT4", counts[False].index)
        self.assertIn("CT4", counts[True].index)
        self.assertGreater(counts[True].loc["CT4"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
