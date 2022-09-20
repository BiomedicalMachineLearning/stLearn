from typing import Optional, Union
from anndata import AnnData
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import numpy as np
import stlearn.plotting.utils as utils


def deconvolution_plot(
    adata: AnnData,
    library_id: str = None,
    use_label: str = "louvain",
    cluster: [int, str] = None,
    celltype: str = None,
    celltype_threshold: float = 0,
    data_alpha: float = 1.0,
    threshold: float = 0.0,
    cmap: str = "tab20",
    colors: list = None,  # The colors to use for each label...
    tissue_alpha: float = 1.0,
    title: str = None,
    spot_size: Union[float, int] = 10,
    show_axis: bool = False,
    show_legend: bool = True,
    show_donut: bool = True,
    cropped: bool = True,
    margin: int = 100,
    name: str = None,
    dpi: int = 150,
    output: str = None,
    copy: bool = False,
    figsize: tuple = (6.4, 4.8),
    show=True,
) -> Optional[AnnData]:

    """\
    Clustering plot for sptial transcriptomics data. Also it has a function to display trajectory inference.

    Parameters
    ----------
    adata
        Annotated data matrix.
    library_id
        Library id stored in AnnData.
    use_label
        Use label result of cluster method.
    list_cluster
        Choose set of clusters that will display in the plot.
    data_alpha
        Opacity of the spot.
    tissue_alpha
        Opacity of the tissue.
    cmap
        Color map to use.
    spot_size
        Size of the spot.
    show_axis
        Show axis or not.
    show_legend
        Show legend or not.
    show_donut
        Whether to show the donut plot or not.
    show_trajectory
        Show the spatial trajectory or not. It requires stlearn.spatial.trajectory.pseudotimespace.
    show_subcluster
        Show subcluster or not. It requires stlearn.spatial.trajectory.global_level.
    name
        Name of the output figure file.
    dpi
        DPI of the output figure.
    output
        Save the figure as file or not.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Nothing
    """

    # plt.rcParams['figure.dpi'] = dpi

    imagecol = adata.obs["imagecol"]
    imagerow = adata.obs["imagerow"]

    fig, ax = plt.subplots(figsize=figsize)

    label = adata.obsm["deconvolution"].T

    tmp = label.sum(axis=1)

    label_filter = label.loc[tmp[tmp >= np.quantile(tmp, threshold)].index]

    if cluster is not None:
        base = adata.obs[adata.obs[use_label] == str(cluster)][["imagecol", "imagerow"]]
    else:
        base = adata.obs[["imagecol", "imagerow"]]

    if celltype is not None:
        base = base.loc[
            adata.obs_names[adata.obsm["deconvolution"][celltype] > celltype_threshold]
        ]

    label_filter_ = label_filter[base.index]

    if type(colors) == type(None):
        color_vals = list(range(0, len(label_filter_), 1))
        my_norm = mpl.colors.Normalize(0, len(label_filter_))
        my_cmap = mpl.cm.get_cmap(cmap, len(color_vals))
        colors = my_cmap.colors

    for i, xy in enumerate(base.values):
        _ = ax.pie(
            label_filter_.T.iloc[i].values,
            colors=colors,
            center=(xy[0], xy[1]),
            radius=spot_size,
            frame=True,
        )
    ax.autoscale()

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    image = adata.uns["spatial"][library_id]["images"][
        adata.uns["spatial"][library_id]["use_quality"]
    ]

    if show_donut:
        ax_pie = fig.add_axes([0.5, -0.4, 0.03, 0.5])

        def my_autopct(pct):
            return ("%1.0f%%" % pct) if pct >= 4 else ""

        ax_pie.pie(
            label_filter_.sum(axis=1),
            colors=my_cmap.colors,
            radius=10,
            # frame=True,
            autopct=my_autopct,
            pctdistance=1.1,
            startangle=90,
            wedgeprops=dict(width=(3), edgecolor="w", antialiased=True),
            textprops={"fontsize": 5},
        )

    if show_legend == True:
        ax_cb = fig.add_axes([0.9, 0.25, 0.03, 0.5], axisbelow=False)
        cb = mpl.colorbar.ColorbarBase(
            ax_cb, cmap=my_cmap, norm=my_norm, ticks=color_vals
        )

        cb.ax.tick_params(size=0)
        loc = np.array(color_vals) + 0.5
        cb.set_ticks(loc)
        cb.set_ticklabels(label_filter_.index)
        cb.outline.set_visible(False)

    # Overlay the tissue image
    ax.imshow(
        image,
        alpha=1,
        zorder=-1,
    )

    ax.axis("off")

    if cropped:
        ax.set_xlim(imagecol.min() - margin, imagecol.max() + margin)

        ax.set_ylim(imagerow.min() - margin, imagerow.max() + margin)

        ax.set_ylim(ax.get_ylim()[::-1])

        # plt.gca().invert_yaxis()

    if name is None:
        name = use_label

    if output is not None:
        fig.savefig(output + "/" + name, dpi=dpi, bbox_inches="tight", pad_inches=0)

    if show:
        plt.show()
