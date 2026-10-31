"""A synthetic Visium HD sample for the trajectory acceptance test.

8 µm bins on a square grid. A latent time runs across the tissue, from one
corner to the opposite one, and each bin's expression (a 20 dimensional
stand-in for PCA) follows a curve in that time, plus noise. Clusters are equal
slices of a noisy reading of the time, as clustering noisy bins gives: each
cluster is a band across the tissue, with stray bins scattered among its
neighbours. The noise sets how many clusters PAGA connects: more noise, more
connections, and many more paths between clusters.

The kNN graph is built here, so the run times only stLearn. The same arguments
always give the same sample, on any machine.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

# 8 µm bins imaged at 0.2738 µm per pixel, hires image at 0.08 of full size.
SCALEFACTORS = {
    "bin_size_um": 8.0,
    "microns_per_pixel": 0.2738,
    "spot_diameter_fullres": 8.0 / 0.2738,
    "tissue_hires_scalef": 0.08,
    "tissue_lowres_scalef": 0.024,
}
# Distance between neighbouring bins in imagecol/imagerow (hires pixels).
BIN_SPACING = (
    SCALEFACTORS["spot_diameter_fullres"] * SCALEFACTORS["tissue_hires_scalef"]
)
N_DIMS = 20


def make_sample(
    n_bins: int = 20000,
    n_clusters: int = 12,
    label_noise: float = 0.04,
    expression_noise: float = 0.15,
    seed: int = 0,
) -> AnnData:
    """The sample, with clusters "0" (earliest) to str(n_clusters - 1) in
    obs["cluster"], the root bin in uns["iroot"] and the kNN graph."""
    rng = np.random.default_rng(seed)
    n_side = round(np.sqrt(n_bins))
    n_bins = n_side * n_side
    row, col = np.divmod(np.arange(n_bins), n_side)

    # Latent time: 0 at one corner, 1 at the opposite one.
    time = (row + col) / (2 * (n_side - 1))

    # Expression: a smooth curve in time that never doubles back, so distance
    # along it from the root grows with time: the first dimension rises with
    # time, the others bend it with slow sines of their own frequency and
    # phase, fading with the dimension like PCA's variance.
    frequencies = rng.uniform(0.2, 1.0, N_DIMS)
    phases = rng.uniform(0, 2 * np.pi, N_DIMS)
    scales = 1.0 / np.sqrt(1 + np.arange(N_DIMS))
    curve = 0.5 * np.sin(np.pi * np.outer(time, frequencies) + phases) * scales
    curve[:, 0] = 2.0 * time
    embedding = curve + rng.normal(0, expression_noise, curve.shape) * scales
    # Rounded, so every machine starts from the same values.
    embedding = np.round(embedding, 6).astype(np.float32)

    # Clusters: equal slices of the time as read through noise.
    read_time = time + rng.normal(0, label_noise, n_bins)
    edges = np.quantile(read_time, np.linspace(0, 1, n_clusters + 1)[1:-1])
    cluster = np.searchsorted(edges, read_time)
    labels = [str(i) for i in range(n_clusters)]

    barcodes = [f"s_008um_{r:05d}_{c:05d}-1" for r, c in zip(row, col, strict=True)]
    adata = AnnData(
        X=embedding.copy(),
        obs=pd.DataFrame(
            {"cluster": pd.Categorical.from_codes(cluster, categories=labels)},
            index=barcodes,
        ),
        var=pd.DataFrame(index=[f"PC{i + 1}" for i in range(N_DIMS)]),
    )
    adata.obsm["X_pca"] = embedding
    spot = SCALEFACTORS["spot_diameter_fullres"]
    hires = SCALEFACTORS["tissue_hires_scalef"]
    adata.obsm["spatial"] = np.column_stack([col, row]) * spot + 1000
    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * hires
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * hires
    adata.uns["spatial"] = {
        "synthetic_hd": {
            "images": {},
            "scalefactors": SCALEFACTORS,
            "use_quality": "hires",
        }
    }
    adata.uns["iroot"] = int(np.argmin(time))
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca", random_state=0)
    return adata
