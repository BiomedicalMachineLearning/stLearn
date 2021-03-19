from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from typing import Optional, Union
from anndata import AnnData
import os
from stlearn._compat import Literal


def add_mask(
    adata: AnnData,
    imgpath: Union[Path, str],
    key: str = "mask",
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Adding binary mask image to the Anndata object

    Parameters
    ----------
    adata
        Anndata object.
    imgpath
        Image mask path.
    key
        Label for mask.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **mask_image** : `adata.uns` field
        Array format of image, saving by Pillow package.
    """
    try:
        library_id = list(adata.uns["spatial"].keys())[0]
        quality = adata.uns["spatial"][library_id]["use_quality"]
    except:
        raise KeyError(
            f"""\
        Please read ST data first and try again
        """
        )

    if imgpath is not None and os.path.isfile(imgpath):
        try:
            img = plt.imread(imgpath, 0)
            assert (
                img.shape == adata.uns["spatial"][library_id]["images"][quality].shape
            ), "\
            size of mask image does not match size of H&E images"
            if "mask_image" not in adata.uns:
                adata.uns["mask_image"] = {}
            if library_id not in adata.uns["mask_image"]:
                adata.uns["mask_image"][library_id] = {}
            if key not in adata.uns:
                adata.uns["mask_image"][library_id][key] = {}

            adata.uns["mask_image"][library_id][key][quality] = img
            print("Added tissue mask to the object!")

            return adata if copy else None
        except:
            raise ValueError(
                f"""\
            {imgpath!r} does not end on a valid extension.
            """
            )
    else:
        raise ValueError(
            f"""\
        {imgpath!r} does not end on a valid extension.
        """
        )
    return adata if copy else None


def apply_mask(
    adata: AnnData,
    masks: Optional[list] = "all",
    select: str = "black",
    cmap: str = "default",
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Parsing the old spaital transcriptomics data

    Parameters
    ----------
    adata
        Anndata object.
    masks
        List of masks for parsing
        if `all`, apply on all masks
    select
        color for ROI, choose from `black` and `white`.
    cmap
        Color map to use.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **mask_image** : `adata.uns` field
        Array format of image, saving by Pillow package.
    """
    from scanpy.plotting import palettes
    from stlearn.plotting import palettes_st

    if cmap == "vega_10_scanpy":
        cmap = palettes.vega_10_scanpy
    elif cmap == "vega_20_scanpy":
        cmap = palettes.vega_20_scanpy
    elif cmap == "default_102":
        cmap = palettes.default_102
    elif cmap == "default_28":
        cmap = palettes.default_28
    elif cmap == "jana_40":
        cmap = palettes_st.jana_40
    elif cmap == "default":
        cmap = palettes_st.default
    else:
        raise ValueError(
            "We only support vega_10_scanpy, vega_20_scanpy, default_28, default_102"
        )

    cmaps = matplotlib.colors.LinearSegmentedColormap.from_list("", cmap)

    cmap_ = plt.cm.get_cmap(cmaps)

    try:
        library_id = list(adata.uns["spatial"].keys())[0]
        quality = adata.uns["spatial"][library_id]["use_quality"]
    except:
        raise KeyError(
            f"""\
        Please read ST data first and try again
        """
        )

    if masks == "all":
        masks = list(adata.uns["mask_image"][library_id].keys())
    else:
        assert isinstance(masks, list)
    key = "mask_annotation"
    adata.obs[key] = "unassigned"
    adata.obs[key + "_code"] = -1
    mask_final = np.zeros(adata.uns["spatial"][library_id]["images"][quality].shape)
    for i, mask in enumerate(masks):
        try:
            mask_image = adata.uns["mask_image"][library_id][mask][quality]
        except:
            raise KeyError(
                f"""\
            Please load mask {mask} images first and try again
            """
            )

        if select == "black":
            mask_image = np.where(mask_image > 155, 0, 1)
        elif select == "white":
            mask_image = np.where(mask_image > 155, 0, 1)
        else:
            raise ValueError(
                f"""\
            Only support black and white mask yet.
            """
            )
        mask_image_2d = mask_image.mean(axis=2)
        apply_spot_mask = (
            lambda x: [i, mask]
            if mask_image_2d[int(x["imagerow"]), int(x["imagecol"])] == 1
            else [x[key + "_code"], x[key]]
        )
        spot_mask_df = adata.obs.apply(apply_spot_mask, axis=1, result_type="expand")
        adata.obs[key + "_code"] = spot_mask_df[0]
        adata.obs[key] = spot_mask_df[1]

        c = cmap_(int(i) / (len(cmap) - 1))
        mask_final = mask_final + mask_image * np.array(c[0:3])
    print(f"""Mask annotation for spots added to `adata.obs["{key}"]`""")
    mask_final[mask_final == 0] = 1
    adata.uns[key] = mask_final
    print(f"""Mask annotation for H&E image added to `adata.uns["{key}"]`""")
    return adata if copy else None
