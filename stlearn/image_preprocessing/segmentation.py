from typing import Optional
import histomicstk as htk
import numpy as np
import scipy as sp
import skimage.color
import skimage.io
import skimage.measure
from anndata import AnnData
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tqdm import tqdm


def morph_watershed(
    adata: AnnData,
    library_id: str = None,
    verbose: bool = False,
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Watershed method to segment nuclei and calculate morphological statistics

    Parameters
    ----------
    adata
        Annotated data matrix.
    library_id
        Library id stored in AnnData.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **n_nuclei** : `adata.obs` field
        saved number of nuclei of each spot image tiles
    **nuclei_total_area** : `adata.obs` field
        saved of total area of nuclei of each spot image tiles
    **nuclei_mean_area** : `adata.obs` field
        saved mean area of nuclei of each spot image tiles
    **nuclei_std_area** : `adata.obs` field
        saved stand deviation of nuclei area of each spot image tiles
    **eccentricity** : `adata.obs` field
        saved eccentricity of each spot image tiles
    **mean_pix_r** : `adata.obs` field
        saved mean pixel value of red channel of of each spot image tiles
    **std_pix_r** : `adata.obs` field
        saved stand deviation of red channel of each spot image tiles
    **mean_pix_g** : `adata.obs` field
        saved mean pixel value of green channel of each spot image tiles
    **std_pix_g** : `adata.obs` field
        saved stand deviation of green channel of each spot image tiles
    **mean_pix_b** : `adata.obs` field
        saved mean pixel value of blue channel of each spot image tiles
    **std_pix_b** : `adata.obs` field
        saved stand deviation of blue channel of each spot image tiles
    **nuclei_total_area_per_tile** : `adata.obs` field
        saved total nuclei area per tile of each spot image tiles
    """

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    n_nuclei_list = []
    nuclei_total_area_list = []
    nuclei_mean_area_list = []
    nuclei_std_area_list = []
    eccentricity_list = []
    mean_pix_list_r = []
    std_pix_list_r = []
    mean_pix_list_g = []
    std_pix_list_g = []
    mean_pix_list_b = []
    std_pix_list_b = []
    with tqdm(
        total=len(adata),
        desc="calculate morphological stats",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for tile in adata.obs["tile_path"]:
            (
                n_nuclei,
                nuclei_total_area,
                nuclei_mean_area,
                nuclei_std_area,
                eccentricity,
                solidity,
                mean_pix_r,
                std_pix_r,
                mean_pix_g,
                std_pix_g,
                mean_pix_b,
                std_pix_b,
            ) = _calculate_morph_stats(tile)
            n_nuclei_list.append(n_nuclei)
            nuclei_total_area_list.append(nuclei_total_area)
            nuclei_mean_area_list.append(nuclei_mean_area)
            nuclei_std_area_list.append(nuclei_std_area)
            eccentricity_list.append(eccentricity)
            mean_pix_list_r.append(mean_pix_r)
            std_pix_list_r.append(std_pix_r)
            mean_pix_list_g.append(mean_pix_g)
            std_pix_list_g.append(std_pix_g)
            mean_pix_list_b.append(mean_pix_b)
            std_pix_list_b.append(std_pix_b)
            pbar.update(1)

    adata.obs["n_nuclei"] = n_nuclei_list
    adata.obs["nuclei_total_area"] = nuclei_total_area_list
    adata.obs["nuclei_mean_area"] = nuclei_mean_area_list
    adata.obs["nuclei_std_area"] = nuclei_std_area_list
    adata.obs["eccentricity"] = eccentricity_list
    adata.obs["mean_pix_r"] = mean_pix_list_r
    adata.obs["std_pix_r"] = std_pix_list_r
    adata.obs["mean_pix_g"] = mean_pix_list_g
    adata.obs["std_pix_g"] = std_pix_list_g
    adata.obs["mean_pix_b"] = mean_pix_list_b
    adata.obs["std_pix_b"] = std_pix_list_b
    adata.obs["nuclei_total_area_per_tile"] = adata.obs["nuclei_total_area"] / 299 / 299
    return adata if copy else None


def _calculate_morph_stats(tile_path):
    imInput = skimage.io.imread(tile_path)
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    stains = [
        "hematoxylin",  # nuclei stain
        "eosin",  # cytoplasm stain
        "null",
    ]  # set to null if input contains only two stains
    w_est = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(
        imInput, 255
    )

    # Perform color deconvolution
    deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(
        imInput, w_est, 255
    )

    channel = htk.preprocessing.color_deconvolution.find_stain_index(
        stain_color_map[stains[0]], w_est
    )
    im_nuclei_stain = deconv_result.Stains[:, :, channel]

    thresh = skimage.filters.threshold_otsu(im_nuclei_stain)
    # im_fgnd_mask = im_nuclei_stain < thresh
    im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
        im_nuclei_stain < 0.8 * thresh
    )

    distance = ndi.distance_transform_edt(im_fgnd_mask)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=im_fgnd_mask)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)

    labels = watershed(im_nuclei_stain, markers, mask=im_fgnd_mask)
    min_nucleus_area = 60
    im_nuclei_seg_mask = htk.segmentation.label.area_open(
        labels, min_nucleus_area
    ).astype(np.int)

    # compute nuclei properties
    objProps = skimage.measure.regionprops(im_nuclei_seg_mask)

    #     # Display results
    #     plt.figure(figsize=(20, 10))
    #     plt.imshow(skimage.color.label2rgb(im_nuclei_seg_mask, im_nuclei_stain, bg_label=0),
    #            origin='upper')
    #     plt.title('Nuclei segmentation mask overlay')
    #     plt.savefig("./Nuclei_segmentation_tiles_bc_wh/{}.png".format(tile_path.split("/")[-1].split(".")[0]), dpi=300)

    n_nuclei = len(objProps)

    nuclei_total_area = sum(map(lambda x: x.area, objProps))
    nuclei_mean_area = np.mean(list(map(lambda x: x.area, objProps)))
    nuclei_std_area = np.std(list(map(lambda x: x.area, objProps)))

    mean_pix = imInput.reshape(3, -1).mean(1)
    std_pix = imInput.reshape(3, -1).std(1)

    eccentricity = np.mean(list(map(lambda x: x.eccentricity, objProps)))

    solidity = np.mean(list(map(lambda x: x.solidity, objProps)))

    return (
        n_nuclei,
        nuclei_total_area,
        nuclei_mean_area,
        nuclei_std_area,
        eccentricity,
        solidity,
        mean_pix[0],
        std_pix[0],
        mean_pix[1],
        std_pix[1],
        mean_pix[2],
        std_pix[2],
    )
