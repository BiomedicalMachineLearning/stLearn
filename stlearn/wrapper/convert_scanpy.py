from anndata import AnnData

from ..utils import _quality_scalef


def convert_scanpy(
    adata: AnnData,
    use_quality: str = "hires",
) -> AnnData:
    adata.var_names_make_unique()

    library_id = next(iter(adata.uns["spatial"].keys()))

    if use_quality == "fulres":
        image_coor = adata.obsm["spatial"]
    else:
        scale = _quality_scalef(adata.uns["spatial"][library_id], use_quality)
        image_coor = adata.obsm["spatial"] * scale

    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = use_quality

    return adata
