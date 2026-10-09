"""A synthetic Visium HD sample for the CCI acceptance test.

8 µm bins on a square grid, with sparse log-normalised expression with ~1.2%
non-zero values (about what 8 µm bins have), every ligand and receptor gene of
connectomeDB2020_lit, and cell type proportions. The same arguments always give
the same sample, on any machine.
"""

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

# 8 µm bins imaged at 0.2738 µm per pixel, hires image at 0.08 of full size.
SCALEFACTORS = {
    "bin_size_um": 8.0,
    "microns_per_pixel": 0.2738,
    "spot_diameter_fullres": 8.0 / 0.2738,
    "tissue_hires_scalef": 0.08,
    "tissue_lowres_scalef": 0.024,
}


def make_sample(
    lr_genes: list[str], n_bins: int = 20000, n_genes: int = 18000, seed: int = 0
) -> AnnData:
    """The sample, with `lr_genes` first among its genes."""
    rng = np.random.default_rng(seed)
    n_side = round(np.sqrt(n_bins))
    n_bins = n_side * n_side
    genes = lr_genes + list(map(lambda i: f"GENE{i}", range(n_genes - len(lr_genes))))

    # Each gene's share of bins expressing it: mostly rare, a few common.
    rates = np.clip(rng.lognormal(np.log(0.006), 1.2, n_genes), 0, 0.6)
    n_expressing = rng.binomial(n_bins, rates)
    columns = np.repeat(np.arange(n_genes), n_expressing)
    rows = np.concatenate(
        list(map(lambda k: rng.choice(n_bins, k, replace=False), n_expressing))
    )
    counts = rng.geometric(0.6, len(rows))
    # Rounded, so every machine's log1p gives the same values.
    values = np.round(np.log1p(counts * 2.5), 3).astype(np.float32)
    X = sparse.csr_matrix(
        (values, (rows, columns)), shape=(n_bins, n_genes), dtype=np.float32
    )

    row, col = np.divmod(np.arange(n_bins), n_side)
    barcodes = list(
        map(
            lambda rc: f"s_008um_{rc[0]:05d}_{rc[1]:05d}-1",
            zip(row, col, strict=True),
        )
    )
    adata = AnnData(X, obs=pd.DataFrame(index=barcodes), var=pd.DataFrame(index=genes))
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

    # Mostly one cell type per bin.
    proportions = pd.DataFrame(
        rng.dirichlet(np.full(6, 0.1), n_bins),
        index=adata.obs_names,
        columns=list(map(lambda i: f"type{i}", range(6))),
    )
    adata.uns["cell_type"] = proportions
    adata.obs["cell_type"] = proportions.idxmax(axis=1).astype("category")
    return adata
