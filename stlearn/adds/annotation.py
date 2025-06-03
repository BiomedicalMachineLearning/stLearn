
from anndata import AnnData


def annotation(
    adata: AnnData,
    label_list: list[str],
    use_label: str = "louvain",
    copy: bool = False,
) -> AnnData | None:
    """\
    Adding annotation for cluster

    Parameters
    ----------
    adata
        Annotated data matrix.
    label_list
        List of the labels which assigned to current cluster result.
    use_label
        Choosing cluster type.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **[cluster method name]_anno** : `adata.obs` field
        The annotation of cluster results.
    """

    if label_list is None:
        raise ValueError("Please give the label list!")

    if len(label_list) != len(adata.obs[use_label].unique()):
        raise ValueError("Please give the correct number of label list!")

    adata.obs[use_label + "_anno"] = adata.obs[use_label].cat.rename_categories(
        label_list
    )

    print("The annotation is added to adata.obs['" + use_label + "_anno" + "']")

    return adata if copy else None
