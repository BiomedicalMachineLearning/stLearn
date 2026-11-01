import pandas as pd
from anndata import AnnData
from natsort import natsorted

from stlearn.adds._barcodes import by_barcode


def labels(
    adata: AnnData,
    label_filepath: str,
    index_col: int = 0,
    use_label: str | None = None,
    sep: str = "\t",
    copy: bool = False,
) -> AnnData | None:
    """\
    Add label transfer results into AnnData object.

    Parameters
    ----------
    adata: AnnData
        The data object to add L-R info into
    label_filepath: str
        The path to the label transfer results file
    use_label: str
        Where to store the label_transfer results, defaults to 'predictions'
        in adata.obs & 'label_transfer' in adata.uns.
    sep: str
        Separator of the csv file
    copy: bool
        Copy flag indicating copy or direct edit

    Returns
    -------
    adata: AnnData
        The data object that L-R added into

    """
    adata = adata.copy() if copy else adata

    labels = pd.read_csv(label_filepath, index_col=index_col, sep=sep)
    labels = by_barcode(labels, adata, label_filepath)

    uns_key = "label_transfer" if use_label is None else use_label
    adata.uns[uns_key] = labels.drop(["predicted.id", "prediction.score.max"], axis=1)

    key_add = "predictions" if use_label is None else use_label
    key_source = "predicted.id"
    predicted = labels[key_source].map(str, na_action="ignore")
    adata.obs[key_add] = pd.Categorical(
        values=predicted.to_numpy(),
        categories=natsorted(predicted.dropna().unique()),
    )

    print(f"label transfer results added to adata.uns['{uns_key}']")
    print(f"predicted label added to adata.obs['{key_add}'].")

    return adata if copy else None
