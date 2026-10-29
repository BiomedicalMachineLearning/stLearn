import os

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
from PIL import Image
from spatialdata.models import Image2DModel
from spatialdata.transformations import Affine
from stlearn.tl.cci import LRPairs


def path_for_tests():
    return os.path.dirname(os.path.realpath(__file__))


def path_for_test_data():
    return f"{path_for_tests()}/test_data"


def read_test_data():
    """Reads in test data to run unit tests."""
    adata = sc.read_h5ad(f"{path_for_test_data()}/test_data.h5")
    im = Image.open(f"{path_for_test_data()}/test_image.jpg")
    adata.uns["spatial"]["V1_Breast_Cancer_Block_A_Section_1"]["images"]["hires"] = (
        np.array(im)
    )
    return adata


RESULTS_LRS = np.array(["A_B", "C_D", "E_F", "G_H"])


def scores_and_pvals(n_spots=30, seed=0):
    """Spots by LRs; about a third of the scores are 0 (untested, p-value 1),
    and the LRs differ in how many small p-values they have."""
    rng = np.random.default_rng(seed)
    scores = rng.gamma(2.0, 1.0, size=(n_spots, len(RESULTS_LRS)))
    scores[rng.random(scores.shape) < 0.35] = 0
    pvals = rng.uniform(0, 1, size=scores.shape) ** np.array([1, 4, 8, 2])
    pvals[scores == 0] = 1
    return scores, pvals


CCI_LRS = LRPairs.from_names(["LigA_RecA", "LigB_RecB", "LigC_RecC"], species="human")
CELL_TYPES = np.array(["CT1", "CT2", "CT3"])


def cci_table(
    n_side: int = 10, n_other_genes: int = 60, seed: int = 0
) -> anndata.AnnData:
    """Counts on a grid of spots, one unit apart, that the CCI pipeline runs on.

    LigA and RecA are high together in one corner, so some spots are significant
    for LigA_RecA. Rows of spots are labelled with three cell types, in bands, both
    as `obs["cell_type"]` and as proportions in `uns["cell_type"]`, so the
    heterogeneity path of `st.tl.cci.run` and `st.tl.cci.run_cci` both run. A blank
    image in `uns["spatial"]` lets the spatial plots draw it.
    """
    rng = np.random.default_rng(seed)
    n_spots = n_side * n_side
    lr_genes = [gene for lr in CCI_LRS.names() for gene in lr.split("_")]
    var_names = lr_genes + [f"gene{i}" for i in range(n_other_genes)]

    X = rng.poisson(0.3, size=(n_spots, len(var_names))).astype(np.float32)
    rows, cols = np.divmod(np.arange(n_spots), n_side)
    X[(rows < 4) & (cols < 4), :2] = 6

    obs_names = [f"spot{i}" for i in range(n_spots)]
    band = np.minimum(rows * len(CELL_TYPES) // n_side, len(CELL_TYPES) - 1)
    adata = anndata.AnnData(
        X=scipy.sparse.csr_matrix(X),
        obs=pd.DataFrame(
            {
                "imagerow": rows.astype(float),
                "imagecol": cols.astype(float),
                "cell_type": pd.Categorical(CELL_TYPES[band], categories=CELL_TYPES),
            },
            index=obs_names,
        ),
        var=pd.DataFrame(index=var_names),
    )
    adata.obsm["spatial"] = np.column_stack([cols, rows]).astype(float)
    adata.uns["cell_type"] = pd.DataFrame(
        np.eye(len(CELL_TYPES))[band], index=obs_names, columns=CELL_TYPES
    )
    adata.uns["spatial"] = {
        "synthetic": {
            "images": {"hires": np.full((n_side, n_side, 3), 255, dtype=np.uint8)},
            "scalefactors": {"tissue_hires_scalef": 1.0, "spot_diameter_fullres": 1.0},
            "use_quality": "hires",
        }
    }
    return adata


PIXEL_SIZE = 0.2125  # microns per morphology pixel, the units of "global" in Xenium

# H&E pixels (x, y) to morphology pixels, like Xenium's he_imagealignment.csv:
# an H&E pixel is 2 morphology pixels across, and the image is flipped vertically.
ALIGNMENT = np.array([[2.0, 0.0, 100.0], [0.0, -2.0, 200.0], [0.0, 0.0, 1.0]])

# An 80 x 60 H&E, grey, with a coloured square centred on each of these pixels (x, y).
MARKERS = {(16, 12): (255, 0, 0), (64, 20): (0, 255, 0), (40, 48): (0, 0, 255)}


def he_image(multiscale: bool):
    pixels = np.full((3, 60, 80), 128, dtype=np.uint8)
    for (x, y), colour in MARKERS.items():
        pixels[:, y - 6 : y + 6, x - 6 : x + 6] = np.array(colour)[:, None, None]
    return Image2DModel.parse(
        pixels,
        dims=("c", "y", "x"),
        transformations={
            "global": Affine(ALIGNMENT, input_axes=("x", "y"), output_axes=("x", "y"))
        },
        scale_factors=[2] if multiscale else None,
    )


def table(xy) -> anndata.AnnData:
    adata = anndata.AnnData(np.ones((len(xy), 1), dtype=np.float32))
    adata.obs_names = [f"cell{i}" for i in range(len(xy))]
    adata.obsm["spatial"] = np.asarray(xy, dtype=float)
    return adata
