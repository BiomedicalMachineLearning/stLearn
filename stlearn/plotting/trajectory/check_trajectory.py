from anndata import AnnData
from typing import Optional, Union
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np


def check_trajectory(
    adata: AnnData,
    library_id: str = None,
    use_label: str = "louvain",
    basis: str = "umap",
    pseudotime_key: str = "dpt_pseudotime",
    trajectory: list = None,
    figsize=(10, 4),
    size_umap: int = 50,
    size_spatial: int = 1.5,
    img_key: str = "hires",
) -> Optional[AnnData]:
    trajectory = np.array(trajectory).astype(int)
    assert (
        trajectory in adata.uns["available_paths"].values()
    ), "Please choose the right path!"
    trajectory = trajectory.astype(str)
    assert (
        pseudotime_key in adata.obs.columns
    ), "Please run the pseudotime or choose the right one!"
    assert (
        use_label in adata.obs.columns
    ), "Please run the clustering or choose the right label!"
    assert basis in adata.obsm, (
        "Please run the " + basis + "before you check the trajectory!"
    )
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    adata.obsm["X_spatial"] = adata.obs[["imagecol", "imagerow"]].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1 = sc.pl.umap(adata, size=size_umap, show=False, ax=ax1)
    sc.pl.umap(
        adata[adata.obs[use_label].isin(trajectory)],
        size=size_umap,
        color=pseudotime_key,
        ax=ax1,
        show=False,
        frameon=False,
    )

    ax2 = sc.pl.scatter(
        adata,
        size=25,
        show=False,
        basis="spatial",
        ax=ax2,
    )
    sc.pl.spatial(
        adata[adata.obs[use_label].isin(trajectory)],
        size=size_spatial,
        ax=ax2,
        color=pseudotime_key,
        legend_loc="none",
        basis="spatial",
        frameon=False,
        show=False,
    )

    im = ax2.imshow(
        adata.uns["spatial"][library_id]["images"][img_key], alpha=0, zorder=-1
    )

    plt.show()

    del adata.obsm["X_spatial"]
