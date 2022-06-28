import numpy as np
from skimage.transform import resize


def transform_spatial(coordinates, original, resized):
    # obs transform
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    x1p = x / original[1]
    y1p = y / original[0]
    x1 = x1p * resized[1]
    y1 = y1p * resized[0]

    return np.vstack([x1, y1]).transpose()


def correct_size(adata, fixed_size):

    image = adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]]["images"][
        "hires"
    ]
    image_size = image.shape[:2]
    if image_size != fixed_size:
        adata.obs[["imagerow", "imagecol"]] = transform_spatial(
            adata.obs[["imagerow", "imagecol"]].values, image_size, fixed_size
        )
        adata.obsm["spatial"] = transform_spatial(
            adata.obsm["spatial"], image_size, fixed_size
        )
        image_resized = resize(image, fixed_size)
        adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]]["images"][
            "hires"
        ] = image_resized

    return adata


def concatenate_spatial_adata(adata_list, ncols=2, fixed_size=(2000, 2000)):
    """\
    Concatnate multiple anndata for visualization of spatial transcriptomics

    Parameters
    ----------
    adata_list
        A list of anndata objet.
    ncols
        Number of columns
    fixed_size
        The size that fixed for every spatial transcriptomics data
    Returns
    -------
    Returns adata.
    """

    use_adata_list = []
    for adata in adata_list:
        use_adata_list.append(adata.copy())

    import math

    # check valid
    n_adata = len(use_adata_list)
    nrows = math.ceil(n_adata / ncols)
    if ncols > n_adata:
        raise ValueError("Number of column is out of bound")

    # Correct size
    for adata in use_adata_list:
        correct_size(adata, fixed_size=fixed_size)

    # Transform
    n = 0
    break_out_flag = False
    scale = use_adata_list[0].uns["spatial"][
        list(use_adata_list[0].uns["spatial"].keys())[0]
    ]["scalefactors"]["tissue_hires_scalef"]
    for i in range(0, nrows):
        for j in range(0, ncols):
            obs_spatial = use_adata_list[n].obs[["imagerow", "imagecol"]].values
            obsm_spatial = use_adata_list[n].obsm["spatial"]
            obs_spatial = np.vstack(
                (
                    obs_spatial[:, 0] + fixed_size[0] * i,
                    obs_spatial[:, 1] + fixed_size[1] * j,
                )
            ).transpose()
            obsm_spatial = np.vstack(
                (
                    obsm_spatial[:, 0] + fixed_size[0] / scale * i,
                    obsm_spatial[:, 1] + fixed_size[1] / scale * j,
                )
            ).transpose()
            use_adata_list[n].obs[["imagerow", "imagecol"]] = obs_spatial
            use_adata_list[n].obsm["spatial"] = obsm_spatial
            if n == len(use_adata_list) - 1:
                break_out_flag = True
                break
            n += 1
        if break_out_flag:
            break

    # Combine images
    imgs = []
    for i, adata in enumerate(use_adata_list):
        imgs.append(
            adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]]["images"][
                "hires"
            ]
        )

    from PIL import Image

    if (nrows * ncols - len(use_adata_list)) > 0:
        for i in range(0, (nrows * ncols - len(use_adata_list))):
            image = Image.new("RGB", fixed_size, (255, 255, 255, 255))
            imgs.append(np.array(image))

    print(len(imgs))

    img_rows = []
    for min_id in range(0, len(use_adata_list), ncols):
        img_row = np.hstack(imgs[min_id : min_id + ncols])
        img_rows.append(img_row)
    imgs_comb = np.vstack((i for i in img_rows))

    adata_concat = use_adata_list[0].concatenate(use_adata_list[1:])
    adata_concat.uns["spatial"] = use_adata_list[0].uns["spatial"]

    adata_concat.uns["spatial"][list(adata_concat.uns["spatial"].keys())[0]]["images"][
        "hires"
    ] = imgs_comb

    return adata_concat
