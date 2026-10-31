import json
from pathlib import Path

import pandas as pd
from anndata import AnnData

from stlearn.adds._barcodes import by_barcode
from stlearn.utils import _quality_scalef

_COLUMNS = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
]
_OLD_QUALITY = {"low": "lowres", "high": "hires"}


def positions(
    adata: AnnData,
    position_filepath: Path | str,
    scale_filepath: Path | str,
    quality: str = "low",
    copy: bool = False,
) -> AnnData | None:
    """\
    Adding spatial information into the Anndata object

    Parameters
    ----------
    adata
        Annotated data matrix.
    position_filepath
        Path to tissue positions file.
    scale_filepath
        Path to scalefactors_json file.
    quality
        Choosing hires, lowres or fulres image.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **imagecol** and **imagerow** : `adata.obs` field
        Spatial information of the tissue image.
    """

    adata = adata.copy() if copy else adata

    quality = _OLD_QUALITY.get(quality, quality)
    if quality not in ("hires", "lowres", "fulres"):
        raise ValueError(
            f"quality must be 'hires', 'lowres' or 'fulres', not {quality!r}."
        )

    # Space Ranger 2.0+ writes a header row - 1.x doesn't
    with open(position_filepath) as file:
        has_header = file.readline().startswith("barcode")
    tissue_positions = pd.read_csv(
        position_filepath,
        header=0 if has_header else None,
        names=_COLUMNS,
        index_col="barcode",
    )
    tissue_positions = by_barcode(tissue_positions, adata, position_filepath)

    with open(scale_filepath) as json_file:
        scale = _quality_scalef({"scalefactors": json.load(json_file)}, quality)

    fullres = tissue_positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]]
    adata.obsm["spatial"] = fullres.to_numpy(dtype=float)
    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale

    return adata if copy else None
