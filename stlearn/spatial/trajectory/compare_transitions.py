import numpy as np


def compare_transitions(adata, trajectories):
    """\
    Compare transition markers between two clades

    Parameters
    ----------
    adata
        Annotated data matrix.
    trajectories
        List of clades names user want to compare.
    Returns
    -------
    Anndata
    """

    pos_1 = list(
        adata.uns[trajectories[0]][adata.uns[trajectories[0]]["score"] >= 0]["gene"]
    )
    pos_2 = list(
        adata.uns[trajectories[1]][adata.uns[trajectories[1]]["score"] >= 0]["gene"]
    )
    compare_pos_1 = np.setdiff1d(pos_1, pos_2, assume_unique=True)
    compare_pos_2 = np.setdiff1d(pos_2, pos_1, assume_unique=True)

    neg_1 = list(
        adata.uns[trajectories[0]][adata.uns[trajectories[0]]["score"] < 0]["gene"]
    )
    neg_2 = list(
        adata.uns[trajectories[1]][adata.uns[trajectories[1]]["score"] < 0]["gene"]
    )
    compare_neg_1 = np.setdiff1d(neg_1, neg_2, assume_unique=True)
    compare_neg_2 = np.setdiff1d(neg_2, neg_1, assume_unique=True)

    compare_result = {}
    compare_result["pos_1"] = compare_pos_1
    compare_result["pos_2"] = compare_pos_2
    compare_result["neg_1"] = compare_neg_1
    compare_result["neg_2"] = compare_neg_2

    compare_result["trajectories"] = trajectories

    adata.uns["compare_result"] = compare_result
    print(
        "The result of comparison between "
        + trajectories[0]
        + " and "
        + trajectories[1]
        + " stored in 'adata.uns['compare_result']'"
    )
