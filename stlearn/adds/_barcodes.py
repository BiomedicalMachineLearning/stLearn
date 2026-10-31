import warnings

import pandas as pd
from anndata import AnnData


def by_barcode(table: pd.DataFrame, adata: AnnData, source) -> pd.DataFrame:
    """`table`'s rows for adata's spots, in adata's order, matched on its index of
    barcodes. Raises if no barcode matches; spots without a row get NaN, with a
    warning."""
    table = table.set_axis(table.index.astype(str))
    missing = adata.obs_names.difference(table.index)
    if len(missing) == adata.n_obs:
        raise ValueError(
            f"None of adata's barcodes are in {source}. Check index_col, and "
            "the barcodes' format: R writes AAAC-1 as AAAC.1."
        )
    if len(missing) > 0:
        warnings.warn(
            f"{len(missing)} of {adata.n_obs} spots have no row in {source} "
            f"and are left empty (NaN), e.g. {list(missing[:3])}.",
            stacklevel=3,
        )
    return table.reindex(adata.obs_names)
