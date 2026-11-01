import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from anndata import AnnData

from stlearn.utils import _cluster_labels, _quality_scalef


def check_trajectory(
    adata: AnnData,
    trajectory: list[str],
    library_id: str | None = None,
    use_label: str = "leiden",
    basis: str = "umap",
    pseudotime_key: str = "dpt_pseudotime",
    figsize=(10, 4),
    size_umap: int = 50,
    size_spatial: float = 7,
    img_key: str | None = None,
) -> None:
    """\
    Plot the spots along a trajectory path, coloured by pseudotime, on an
    embedding and on the tissue.

    Parameters
    ----------
    adata
        Annotated data matrix.
    trajectory
        Labels of the clusters along a path in ``adata.uns["available_paths"]``,
        e.g. ``["6", "9"]``.
    library_id
        Library id stored in AnnData.
    use_label
        Use label result of cluster method.
    basis
        The embedding to plot, from ``adata.obsm[f"X_{basis}"]``.
    pseudotime_key
        Key of the pseudotime in ``adata.obs``.
    figsize
        Size of the figure.
    size_umap
        Size of the spots on the embedding.
    size_spatial
        Area of the spots on the tissue, in points^2, as for matplotlib's scatter.
    img_key
        Image to show the tissue on: "hires", "lowres" or "fulres". Defaults to
        the image the spot coordinates were read at,
        ``adata.uns["spatial"][library_id]["use_quality"]``.
    """
    if use_label not in adata.obs.columns:
        raise ValueError(
            f"adata.obs[{use_label!r}] not found: run the clustering or choose "
            "another use_label."
        )
    trajectory = _cluster_labels(adata, use_label, trajectory)
    paths = [list(path) for path in adata.uns["available_paths"].values()]
    if trajectory not in paths:
        raise ValueError(
            f"{trajectory} is not an available trajectory path. Choose from: {paths}"
        )
    if pseudotime_key not in adata.obs.columns:
        raise ValueError(
            f"adata.obs[{pseudotime_key!r}] not found: run "
            "stlearn.spatial.trajectory.pseudotime or choose another pseudotime_key."
        )
    if f"X_{basis}" not in adata.obsm:
        raise ValueError(
            f"adata.obsm['X_{basis}'] not found: compute the {basis} embedding "
            "before checking the trajectory."
        )
    if library_id is None:
        library_id = next(iter(adata.uns["spatial"].keys()))
    spatial = adata.uns["spatial"][library_id]

    # imagecol and imagerow are in the pixels of the image they were read at,
    # as in stlearn's other spatial plots; scale them if another image is shown.
    use_quality = spatial.get("use_quality", "hires")
    if img_key is None:
        img_key = use_quality
    if img_key not in spatial["images"]:
        raise ValueError(
            f"Image {img_key!r} not found in adata.uns['spatial'][{library_id!r}]"
            f"['images']. Choose from: {list(spatial['images'])}"
        )

    scale = _quality_scalef(spatial, img_key) / _quality_scalef(spatial, use_quality)

    imagecol = adata.obs["imagecol"].to_numpy() * scale
    imagerow = adata.obs["imagerow"].to_numpy() * scale

    on_path = adata.obs[use_label].isin(trajectory).to_numpy()
    pseudotime = adata.obs[pseudotime_key].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    sc.pl.embedding(adata, basis=basis, size=size_umap, show=False, ax=ax1)
    sc.pl.embedding(
        adata[on_path],
        basis=basis,
        size=size_umap,
        color=pseudotime_key,
        ax=ax1,
        show=False,
        frameon=False,
    )

    # Both layers come from the same coordinates as the image, so they line up
    # whichever image quality the data was read at.
    ax2.imshow(spatial["images"][img_key], zorder=-1)
    ax2.scatter(imagecol, imagerow, s=size_spatial, c="lightgrey", edgecolor="none")
    plot = ax2.scatter(
        imagecol[on_path],
        imagerow[on_path],
        c=pseudotime[on_path],
        s=size_spatial,
        edgecolor="none",
    )
    cb = fig.colorbar(plot, ax=ax2, label=pseudotime_key)
    cb.outline.set_visible(False)

    # Crop to the spots, keeping the image's orientation (y pointing down).
    span = max(np.ptp(imagecol), np.ptp(imagerow))
    margin = 0.05 * span
    ax2.set_xlim(imagecol.min() - margin, imagecol.max() + margin)
    ax2.set_ylim(imagerow.max() + margin, imagerow.min() - margin)
    ax2.axis("off")

    plt.show()
