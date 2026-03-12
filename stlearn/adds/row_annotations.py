from pathlib import Path

import pandas as pd
from anndata import AnnData
from pandas import DataFrame


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

    annotations = _read_annotations(annotations, columns, join_column)
    merged = annotations.reindex(adata.obs_names)

    added_cols = list(merged.columns)

    for col in added_cols:
        adata.obs[col] = merged[col].values

    return adata if copy else None


def row_annotations_proportions(
    adata: AnnData,
    annotations: pd.DataFrame | str | Path,
    proportion_column_name: str = "cell_type",
    join_column: str | None = None,
    columns: list[str] | None = None,
    copy: bool = False,
) -> AnnData | None:
    """\
    Add annotations to adata.obs by joining on cell/spot identifiers and then
    assuming that the values are cell proportions picking the highest proportion.

    Merges a DataFrame (or CSV file) into adata.obs based on a
    shared index or column. Useful for adding metadata such as
    manual labels, clinical annotations, or external classifications.

    Parameters
    ----------
    adata
        Annotated data matrix.
    annotations
        DataFrame or path to a CSV/TSV file containing annotations.
    proportion_column_name, default =
        The column name to use to add the high proprotion value to
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

    annotations = _read_annotations(annotations, columns, join_column)
    annotations = annotations.astype('float64')
    labels = annotations.idxmax(axis=1)
    adata.obs[proportion_column_name] = labels

    return adata if copy else None


def _read_annotations(annotations: DataFrame | str | Path,
                      columns: list[str] | None,
                      join_column: str | None) -> DataFrame:
    if isinstance(annotations, (str, Path)):
        path = Path(annotations)
        sep = "\t" if path.suffix in (".tsv", ".txt") else ","
        annotations = pd.read_csv(path, sep=sep)

    if join_column is None:
        annotations = annotations.set_index(annotations.columns[0])
    else:
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
    return annotations
