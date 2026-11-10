"""A synthetic Visium HD sample for the CCI acceptance test.

8 µm bins on a square grid, with sparse log-normalised expression with ~1.2%
non-zero values (about what 8 µm bins have), every ligand and receptor gene of
connectomeDB2025, and cell type proportions. The same arguments always give
the same sample, on any machine.
"""

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from tests.acceptance.visium_hd import barcodes, bin_grid, place_bins


def make_sample(
    lr_genes: list[str], n_bins: int = 20000, n_genes: int = 18000, seed: int = 0
) -> AnnData:
    """The sample, with `lr_genes` first among its genes."""
    rng = np.random.default_rng(seed)
    _, row, col = bin_grid(n_bins)
    n_bins = len(row)
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

    adata = AnnData(
        X, obs=pd.DataFrame(index=barcodes(row, col)), var=pd.DataFrame(index=genes)
    )
    place_bins(adata, row, col)

    # Mostly one cell type per bin.
    proportions = pd.DataFrame(
        rng.dirichlet(np.full(6, 0.1), n_bins),
        index=adata.obs_names,
        columns=list(map(lambda i: f"type{i}", range(6))),
    )
    adata.uns["cell_type"] = proportions
    adata.obs["cell_type"] = proportions.idxmax(axis=1).astype("category")
    return adata
