from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import sys
from anndata import AnnData
from typing import Optional, Union

from typing import Optional, Union, Mapping  # Special
from typing import Sequence, Iterable  # ABCs
from typing import Tuple  # Classes

import warnings

from .classes import CciPlot
from .classes_bokeh import BokehCciPlot
from ._docs import doc_spatial_base_plot, doc_het_plot, doc_lr_plot
from ..utils import Empty, _empty, _AxesSubplot, _docs_params
from .cluster_plot import cluster_plot
from .deconvolution_plot import deconvolution_plot
from .gene_plot import gene_plot

from bokeh.io import push_notebook, output_notebook
from bokeh.plotting import show

@_docs_params(spatial_base_plot=doc_spatial_base_plot, het_plot=doc_lr_plot)
def lr_plot(
    adata: AnnData, lr: str,
    min_expr: float = 0, sig_spots=True,
    use_label: str = None, use_mix: str = None, outer_mode: str = 'continuous',
    l_cmap=None, r_cmap=None, lr_cmap=None, inner_cmap=None,
    inner_size_prop: float=0.25, middle_size_prop: float=0.5,
    outer_size_prop: float=1, pt_scale: int=100, title='',
    show_image: bool=True,
    # plotting params
    **kwargs,
) -> Optional[AnnData]:

    # Input checking #
    l, r = lr.split('_')
    ran_lr = 'lr_summary' in adata.uns
    ran_sig = False if not ran_lr else 'n_spots_sig' in adata.uns['lr_summary'].columns
    if ran_lr and lr in adata.uns['lr_summary'].index:
        if ran_sig:
            lr_sig = adata.uns['lr_summary'].loc[lr, :].values[1] > 0
        else:
            lr_sig = True
    else:
        lr_sig = False

    if sig_spots and not ran_lr:
        raise Exception("No LR results testing results found, "
                      "please run st.tl.cci.run first, or set sig_spots=False.")

    elif sig_spots and not lr_sig:
        raise Exception("LR has no significant spots, to visualise anyhow set"
                        "sig_spots=False")

    # Getting which are the allowed stats for the lr to plot #
    if not ran_sig:
        lr_use_labels = ['lr_scores']
    else:
        lr_use_labels = ['lr_scores', 'p_val', 'p_adj', '-log10(p_adj)', 'lr_sig_scores']

    if type(use_mix)!=type(None) and use_mix not in adata.uns:
        raise Exception(f"Specified use_mix, but no deconvolution results added "
                       "to adata.uns matching the use_mix ({use_mix}) key.")
    elif type(use_label)!=type(None) and use_label in lr_use_labels and not lr_sig:
        raise Exception(f"Since use_label refers to lr stats, "
                        f"need to first run st.tl.cci.run.")
    elif type(use_label)!=type(None) and use_label not in adata.obs.keys() \
                                             and use_label not in lr_use_labels:
        raise Exception(f"use_label must be in adata.obs or "
                        f"one of lr stats: {lr_use_labels}.")

    out_options = ['binary', 'continuous', None]
    if outer_mode not in out_options:
        raise Exception(f"{outer_mode} should be one of {out_options}")

    if l not in adata.var_names or r not in adata.var_names:
        raise Exception("L or R not found in adata.var_names.")

    # Whether to show just the significant spots or all spots
    if sig_spots:
        lr_results = adata.uns['per_lr_results'][lr]
        sig_spots = lr_results.loc[:, 'lr_sig_scores'].values != 0
        adata = adata[sig_spots, :]

    # Dealing with the axis #
    fig, ax = plt.subplots()

    l_expr = adata[:, l].X.toarray()[:, 0]
    r_expr = adata[:, r].X.toarray()[:, 0]
    if outer_mode == 'binary':
        l_bool, r_bool = l_expr > min_expr, r_expr > min_expr
        lr_binary_labels = []
        for i in range(len(l_bool)):
            if l_bool[i] and not r_bool[i]:
                lr_binary_labels.append( l )
            elif not l_bool[i] and r_bool[i]:
                lr_binary_labels.append( r )
            elif l_bool[i] and r_bool[i]:
                lr_binary_labels.append( lr )
            else:
                lr_binary_labels.append( '' )
        lr_binary_labels = pd.Series(np.array(lr_binary_labels),
                                       index=adata.obs_names).astype('category')
        adata.obs[f'{lr}_binary_labels'] = lr_binary_labels

        if type(lr_cmap) == type(None):
            lr_cmap = "default" #This gets ignored due to setting colours below
            adata.uns[f'{lr}_binary_labels_set'] = [l, r, lr, '']
            adata.uns[f'{lr}_binary_labels_colors'] = \
                [matplotlib.colors.to_hex('r'), matplotlib.colors.to_hex('limegreen'),
                 matplotlib.colors.to_hex('b'), matplotlib.colors.to_hex('k')]
        else:
            lr_cmap = get_cmap(lr_cmap)

        cluster_plot(adata, use_label=f'{lr}_binary_labels', cmap=lr_cmap,
                           size=outer_size_prop * pt_scale,
                           ax=ax, fig=fig, show_image=show_image, **kwargs)

    elif outer_mode == 'continuous':
        if type(l_cmap)==type(None):
            l_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('lcmap',
                                                                [(0, 0, 0),
                                                                 (.5, 0, 0),
                                                                 (.75, 0, 0),
                                                                 (1, 0, 0)])
        else:
            l_cmap = get_cmap(l_cmap)
        if type(r_cmap)==type(None):
            r_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('rcmap',
                                                                [(0, 0, 0),
                                                                 (0, .5, 0),
                                                                 (0, .75, 0),
                                                                 (0, 1, 0)])
        else:
            r_cmap = get_cmap(r_cmap)

        gene_plot(adata, gene_symbols=l, size=outer_size_prop * pt_scale,
               cmap=l_cmap, color_bar_label=l, ax=ax, fig=fig,
                                                show_image=show_image, **kwargs)
        gene_plot(adata, gene_symbols=r, size=middle_size_prop * pt_scale,
               cmap=r_cmap, color_bar_label=r, ax=ax, fig=fig,
                                                show_image=show_image, **kwargs)

    if type(use_label) != type(None):
        if use_label in lr_use_labels:
            inner_cmap = inner_cmap if type(inner_cmap) != type(None) else "copper"
            adata.obsm[f'{lr}_{use_label}'] = adata.uns['per_lr_results'][
                                     lr].loc[adata.obs_names,use_label].values
            het_plot(adata, use_het=f'{lr}_{use_label}', show_image=show_image,
                     cmap=inner_cmap,
                     ax=ax, fig=fig, size=inner_size_prop * pt_scale, **kwargs)
        else:
            inner_cmap = inner_cmap if type(inner_cmap)!=type(None) else "default"
            cluster_plot(adata, use_label=use_label, cmap=inner_cmap,
                         size=inner_size_prop * pt_scale,
                         ax=ax, fig=fig, show_image=show_image, **kwargs)

    plt.title(title)

def get_cmap(cmap):
    """ Initialize cmap
    """
    scanpy_cmap = ["vega_10_scanpy", "vega_20_scanpy", "default_102",
                   "default_28"]
    stlearn_cmap = ["jana_40", "default"]
    cmap_available = plt.colormaps() + scanpy_cmap + stlearn_cmap
    error_msg = "cmap must be a matplotlib.colors.LinearSegmentedColormap OR" \
                "one of these: " + str(cmap_available)
    if type(cmap) == str:
        assert cmap in cmap_available, error_msg
    elif type(cmap) != matplotlib.colors.LinearSegmentedColormap:
        raise Exception(error_msg)

    return cmap

@_docs_params(spatial_base_plot=doc_spatial_base_plot, het_plot=doc_het_plot)
def het_plot(
    adata: AnnData,
    # plotting param
    title: Optional["str"] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = "Spectral_r",
    use_label: Optional[str] = None,
    list_clusters: Optional[list] = None,
    ax: Optional[matplotlib.axes._subplots.Axes] = None,
    fig: Optional[matplotlib.figure.Figure] = None,
    show_plot: Optional[bool] = True,
    show_axis: Optional[bool] = False,
    show_image: Optional[bool] = True,
    show_color_bar: Optional[bool] = True,
    crop: Optional[bool] = True,
    margin: Optional[bool] = 100,
    size: Optional[float] = 7,
    image_alpha: Optional[float] = 1.0,
    cell_alpha: Optional[float] = 1.0,
    use_raw: Optional[bool] = False,
    fname: Optional[str] = None,
    dpi: Optional[int] = 120,
    # cci param
    use_het: Optional[str] = "het",
    contour: bool = False,
    step_size: Optional[int] = None,
) -> Optional[AnnData]:

    """\
    Allows the visualization of significant cell-cell interaction
    as the values of dot points or contour in the Spatial
    transcriptomics array.


    Parameters
    -------------------------------------
    {spatial_base_plot}
    {het_plot}

    Examples
    -------------------------------------
    >>> import stlearn as st
    >>> adata = st.datasets.example_bcba()
    >>> pvalues = "lr_pvalues"
    >>> st.pl.gene_plot(adata, use_het = pvalues)

    """

    CciPlot(
        adata,
        title=title,
        figsize=figsize,
        cmap=cmap,
        use_label=use_label,
        list_clusters=list_clusters,
        ax=ax,
        fig=fig,
        show_plot=show_plot,
        show_axis=show_axis,
        show_image=show_image,
        show_color_bar=show_color_bar,
        crop=crop,
        margin=margin,
        size=size,
        image_alpha=image_alpha,
        cell_alpha=cell_alpha,
        use_raw=use_raw,
        fname=fname,
        dpi=dpi,
        use_het=use_het,
        contour=contour,
        step_size=step_size,
    )


def het_plot_interactive(adata: AnnData):
    bokeh_object = BokehCciPlot(adata)
    output_notebook()
    show(bokeh_object.app, notebook_handle=True)


def grid_plot(
    adata: AnnData,
    use_het: str = None,
    num_row: int = 10,
    num_col: int = 10,
    vmin: float = None,
    vmax: float = None,
    cropped: bool = True,
    margin: int = 100,
    dpi: int = 100,
    name: str = None,
    output: str = None,
    copy: bool = False,
) -> Optional[AnnData]:

    """
    Cell diversity plot for sptial transcriptomics data.

    Parameters
    ----------
    adata:                  Annotated data matrix.
    use_het:                Cluster heterogeneity count results from tl.cci.het
    num_row: int            Number of grids on height
    num_col: int            Number of grids on width
    cropped                 crop image or not.
    margin                  margin used in cropping.
    dpi:                    Set dpi as the resolution for the plot.
    name:                   Name of the output figure file.
    output:                 Save the figure as file or not.
    copy:                   Return a copy instead of writing to adata.

    Returns
    -------
    Nothing
    """

    try:
        import seaborn as sns
    except:
        raise ImportError("Please run `pip install seaborn`")
    plt.subplots()

    sns.heatmap(
        pd.DataFrame(np.array(adata.obsm[use_het]).reshape(num_col, num_row)).T,
        vmin=vmin,
        vmax=vmax,
    )
    plt.axis("equal")

    if output is not None:
        plt.savefig(
            output + "/" + name + "_heatmap.pdf",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
        )

    plt.show()
