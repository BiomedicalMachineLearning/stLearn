from typing import Optional, Union
from anndata import AnnData
from matplotlib import pyplot as plt
from pathlib import Path
import os


def convert_scanpy(
    adata: AnnData,
    use_quality: str = "hires",
) -> Optional[AnnData]:

    adata.var_names_make_unique()

    library_id = list(adata.uns["spatial"].keys())[0]

    if use_quality == "fulres":
        image_coor = adata.obsm["spatial"]
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + use_quality + "_scalef"
        ]
        image_coor = adata.obsm["spatial"] * scale

    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = use_quality

    return adata
