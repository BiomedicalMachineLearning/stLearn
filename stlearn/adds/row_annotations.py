from pathlib import Path

import pandas as pd
from anndata import AnnData


def row_annotations(
    adata: AnnData,
    annotations: pd.DataFrame | str | Path,
    join_column: str | None = None,
    columns: list[str] | None = None,
    copy: bool = False,
) -> AnnData | None:
    """\
    Add annotations to adata.obs by joining on cell/spot identifiers.

    Merges a DataFrame (or CSV file) into adata.obs based on a
    shared index or column. Useful for adding metadata such as
    manual labels, clinical annotations, or external classifications.

    Parameters
    ----------
    adata
        Annotated data matrix.
    annotations
        DataFrame or path to a CSV/TSV file containing annotations.
    join_column
        Column in annotations to join on. If None, uses the
        DataFrame index. The join is always against adata.obs_names.
    columns
        Subset of columns to add. If None, adds all columns
        (excluding join_column).
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    Depending on `copy`, returns or updates `adata` with new
    columns added to `adata.obs`.
    """
    adata = adata.copy() if copy else adata

    if isinstance(annotations, (str, Path)):
        path = Path(annotations)
        sep = "\t" if path.suffix in (".tsv", ".txt") else ","
        annotations = pd.read_csv(path, sep=sep)

    if join_column is not None:
        if join_column not in annotations.columns:
            raise ValueError(
                f"Column '{join_column}' not found. "
                f"Available: {list(annotations.columns)}"
            )
        annotations = annotations.set_index(join_column)

    if columns is not None:
        missing = [c for c in columns if c not in annotations.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        annotations = annotations[columns]

    merged = annotations.reindex(adata.obs_names)
    added_cols = list(merged.columns)

    for col in added_cols:
        adata.obs[col] = merged[col].values

    return adata if copy else None
