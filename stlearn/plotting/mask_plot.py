import matplotlib
from matplotlib import pyplot as plt

from typing import Optional, Union
from anndata import AnnData


def plot_mask(
    adata: AnnData,
    library_id: str = None,
    show_spot: bool = True,
    spot_alpha: float = 1.0,
    cmap: str = "vega_20_scanpy",
    tissue_alpha: float = 1.0,
    mask_alpha: float = 0.5,
    spot_size: Union[float, int] = 6.5,
    show_legend: bool = True,
    name: str = "mask_plot",
    dpi: int = 150,
    output: str = None,
    show_axis: bool = False,
    show_plot: bool = True,
) -> Optional[AnnData]:
    """\
    mask plot for sptial transcriptomics data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    library_id
        Library id stored in AnnData.
    show_spot
        Show spot or not
    spot_alpha
        Opacity of the spot.
    cmap
        Color map to use.
    tissue_alpha
        Opacity of the tissue.
    mask_alpha
        Opacity of the mask.
    spot_size
        Size of the spot.
    show_axis
        Show axis or not.
    show_legend
        Show legend or not.
    name
        Name of the output figure file.
    dpi
        DPI of the output figure.
    output
        Save the figure as file or not.
    show_plot
        Show plot or not
    Returns
    -------
    Nothing
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

    plt.rcParams["figure.dpi"] = dpi

    # Option for turning off showing figure
    plt.ioff()

    # Initialize matplotlib
    fig, a = plt.subplots()

    x_min, x_max = adata.obs["imagecol"].min(), adata.obs["imagecol"].max()
    y_min, y_max = adata.obs["imagerow"].min(), adata.obs["imagerow"].max()
    add_x = (x_max - x_min) // 20
    add_y = (y_max - y_min) // 20

    plt.xlim([x_min - add_x, x_max + add_x])
    plt.ylim([y_max + add_y, y_min - add_y])
    key = "mask_annotation"
    if show_spot:
        for i, cluster in enumerate(adata.obs.groupby(key)):
            if cluster[1][key + "_code"][0] == -1:
                # Plot scatter plot based on pixel of spots
                _ = a.scatter(
                    cluster[1]["imagecol"],
                    cluster[1]["imagerow"],
                    edgecolor="none",
                    alpha=spot_alpha,
                    s=spot_size,
                    marker="o",
                    c=[(1.0, 1.0, 1.0, 1.0)],
                )
            else:
                # Plot scatter plot based on pixel of spots
                _ = a.scatter(
                    cluster[1]["imagecol"],
                    cluster[1]["imagerow"],
                    edgecolor="none",
                    alpha=spot_alpha,
                    s=spot_size,
                    marker="o",
                    c=[cmap_(int(cluster[1][key + "_code"][0]) / (len(cmap) - 1))],
                )

    if show_legend:
        from matplotlib.patches import Patch

        legend_elements = []
        for index, row in adata.obs.groupby(key).first().reset_index().iterrows():
            if row[key + "_code"] == -1:
                legend_elements.append(
                    Patch(color=(1.0, 1.0, 1.0, 1.0), label=row[key])
                )
            else:
                legend_elements.append(
                    Patch(
                        color=cmap_(int(row[key + "_code"]) / (len(cmap) - 1)),
                        label=row[key],
                    )
                )

        a.legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            prop=dict(size=8),
        )

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    image = adata.uns["spatial"][library_id]["images"][
        adata.uns["spatial"][library_id]["use_quality"]
    ]

    if not show_axis:
        a.axis("off")
    # Overlay the tissue image
    a.imshow(
        image,
        alpha=tissue_alpha,
        zorder=-1,
    )
    a.imshow(
        adata.uns[key],
        alpha=mask_alpha,
        zorder=-1,
    )

    if output is not None:
        fig.savefig(output + "/" + name, dpi=dpi, bbox_inches="tight", pad_inches=0)

    if show_plot == True:
        plt.show()
