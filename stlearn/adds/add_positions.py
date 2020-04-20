from typing import Optional, Union
from anndata import AnnData
import pandas as pd
from pathlib import Path
import os


def positions(
    adata: AnnData,
    position_filepath: Union[Path, str] = None,
    scale_filepath: Union[Path, str] = None,
    quality: str = "low",
    copy: bool = False,
) -> Optional[AnnData]:

    """\
    Adding spatial information into the Anndata object

    Parameters
    ----------
    adata
        Annotated data matrix.
    position_filepath
        Path to tissue_positions_list file.
    scale_filepath
        Path to scalefactors_json file.
    quality
        Choosing low or high resolution image.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **imagecol** and **imagerow** : `adata.obs` field
        Spatial information of the tissue image.
    """

    tissue_positions = pd.read_csv(position_filepath, header=None)
    tissue_positions.columns = ["barcode", "tissue",
                                "row", "col", "imagerow", "imagecol"]
    import json
    with open(scale_filepath) as json_file:
        scale_factors = json.load(json_file)

    if quality == "low":
        tissue_positions["imagerow"] = tissue_positions["imagerow"] * \
            scale_factors['tissue_lowres_scalef']
        tissue_positions["imagecol"] = tissue_positions["imagecol"] * \
            scale_factors['tissue_lowres_scalef']
    elif quality == "high":
        tissue_positions["imagerow"] = tissue_positions["imagerow"] * \
            scale_factors['tissue_hires_scalef']
        tissue_positions["imagecol"] = tissue_positions["imagecol"] * \
            scale_factors['tissue_hires_scalef']

    tmp = adata.obs.merge(tissue_positions.reset_index()
                          .set_index(['barcode']),
                          left_index=True,
                          right_index=True,
                          how='left').reset_index()[["imagerow", "imagecol"]]

    adata.obs["imagerow"] = tmp["imagerow"].values
    adata.obs["imagecol"] = tmp["imagecol"].values

    return adata if copy else None
