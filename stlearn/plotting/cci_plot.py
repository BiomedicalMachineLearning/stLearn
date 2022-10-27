from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib
import pandas as pd
import numpy as np
import networkx as nx
import math
import matplotlib.patches as patches
from numba.typed import List
import seaborn as sns
import sys
from anndata import AnnData
from typing import Optional, Union

from typing import Optional, Union, Mapping  # Special
from typing import Sequence, Iterable  # ABCs
from typing import Tuple  # Classes

import warnings

from .classes import CciPlot, LrResultPlot
from .classes_bokeh import BokehSpatialCciPlot, BokehLRPlot
from ._docs import doc_spatial_base_plot, doc_het_plot, doc_lr_plot
from ..utils import Empty, _empty, _AxesSubplot, _docs_params
from .utils import get_cmap, check_cmap, get_colors
from .cluster_plot import cluster_plot
from .deconvolution_plot import deconvolution_plot
from .gene_plot import gene_plot
from stlearn.plotting.utils import get_colors
import stlearn.plotting.cci_plot_helpers as cci_hs
from .cci_plot_helpers import (
    get_int_df,
    add_arrows,
    create_flat_df,
    _box_map,
    chordDiagram,
)
from scipy.stats import gaussian_kde

import importlib

importlib.reload(cci_hs)

from bokeh.io import push_notebook, output_notebook
from bokeh.plotting import show

#### Functions for visualising the overall LR results and diagnostics.


def lr_diagnostics(
    adata,
    highlight_lrs: list = None,
    n_top: int = None,
    color0: str = "turquoise",
    color1: str = "plum",
    figsize: tuple = (10, 4),
    lr_text_fp: dict = None,
    show: bool = True,
):
    """Diagnostic plot looking at relationship between technical features of lrs and lr rank.
        Two plots generated: left is the average of the median for nonzero
        expressing spots for both the ligand and the receptor on the y-axis, &
        LR-rank by no. of significant spots on the x-axis. Right is the average
        of the proportion of zeros for the ligand and receptor gene on teh y-axis.

    Parameters
    ----------
    adata: AnnData
        The data object on which st.tl.cci.run has been applied.
    highlight_lrs: list
        List of LRs to highlight, will add text and change point color for these LR pairs.
    n_top: int
        The number of LRs to display. If None shows all.
    color0: str
        The color of the nonzero-median scatter plot.
    lr_text_fp: dict
        Font dict for the LR text if highlight_lrs not None.
    axis_text_fp: dict
        Font dict for the axis text labels.
    Returns
    -------
    Figure, Axes
        Figure and axes of the plot, if show=False.
    """
    if type(n_top) == type(None):
        n_top = adata.uns["lr_summary"].shape[0]
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    cci_hs.lr_scatter(
        adata,
        "nonzero-median",
        highlight_lrs=highlight_lrs,
        n_top=n_top,
        color=color0,
        ax=axes[0],
        lr_text_fp=lr_text_fp,
        show=False,
    )
    cci_hs.lr_scatter(
        adata,
        "zero-prop",
        highlight_lrs=highlight_lrs,
        n_top=n_top,
        color=color1,
        ax=axes[1],
        lr_text_fp=lr_text_fp,
        show=False,
    )
    if show:
        plt.show()
    else:
        return fig, axes


def lr_summary(
    adata,
    n_top: int = 50,
    highlight_lrs: list = None,
    y: str = "n_spots_sig",
    color: str = "gold",
    figsize: tuple = None,
    highlight_color: str = "red",
    max_text: int = 50,
    lr_text_fp: dict = None,
    ax: Axes = None,
    show: bool = True,
):
    """Plotting the top LRs ranked by number of significant spots.

    Parameters
    ----------
    adata: AnnData
        The data object on which st.tl.cci.run has been applied.
    n_top: int
        The no. of LRs to plot.
    highlight_lrs: list
        A list of LRs to highlight on the plot, will added text and change color
        of points for these LRs. Useful for highlighting LRs of interest.
    y: str
        The way to rank the LRs, default is by the no. of signifcant spots,
        but can be any column in adata.uns['lr_summary'].
    color: str
        The color of the points.
    figsize: tuple
        Size of the figure; (width, height).
    highlight_color: str
        Only relevant if highlight_lrs specified; controls colour of LRs to
        highlight.
    max_text: int
        If the no. of n_top is above this limit, stop showing text to indicate
        the LR names. Allows to see global shape without crowding with LR name
        text.
    lr_text_fp: dict
        Matplotlib font dictionary specifying text details, eg fontsize.
    ax: Axes
        Axes on which to draw the scatter plot; if not inputted constructs own.
    show: bool
        Whether to show the plot, if False will return the ax.
    Returns
    _______
    Axes
        If show=False, returns the ax for additional modification.
    """
    allowed = ["n_spots", "n_spots_sig", "n_spots_sig_pval"]
    if y not in allowed:
        raise Exception(f"Got {y} for y; must be one of {allowed}")

    return cci_hs.lr_scatter(
        adata,
        y,
        n_top=n_top,
        color=color,
        show_all=n_top <= max_text,
        figsize=figsize,
        highlight_lrs=highlight_lrs,
        ax=ax,
        lr_text_fp=lr_text_fp,
        highlight_color=highlight_color,
        show=show,
    )


def lr_n_spots(
    adata,
    n_top: int = 100,
    font_dict: dict = None,
    xtick_dict: dict = None,
    bar_width: float = 1,
    max_text: int = 50,
    non_sig_color: str = "dodgerblue",
    sig_color: str = "springgreen",
    figsize: tuple = (6, 4),
    show_title: bool = True,
    show: bool = True,
):
    """Bar plot showing for each LR no. of sig versus non-sig spots.

    Parameters
    ----------
    adata: AnnData
        Data on which st.tl.cci.run has been applied.
    n_top: int
        The no. of LRs to display, ranked by adata.uns['lr_summary']
    font_dict: dict
        dictionary specifying matplotlib font parameters e.g. weight.
    xtick_dict: dict
        dictionary specifying matplotlib font parameters for x labels e.g. weight.
    bar_width: int
        Width of each bar in the bar plot.
    max_text: int
        If n_top exceeds this number, stop showing the LR text, since can cause
        crowding.
    non_sig_color: str
        Specifies the color for bar plot proportion indicating no. of non-sig
        spots.
    sig_color: str
        Specifies color for bars indicating the sig. spots counts.
    figsize: tuple
        Specifies figure dimensions.
    show_title: bool
        Whether to show title on outputted plot.
    show: bool
        Whether to show the plot; if false returns the figure & axes for further
        modification.
    Returns
    -------
    Fig, Axes
        Figure & axes with the plot draw on; only if show=False. Else None.
    """
    if type(font_dict) == type(None):
        font_dict = {"weight": "bold", "size": 12}
    if type(xtick_dict) == type(None):
        xtick_dict = {"fontweight": "bold", "rotation": 90, "size": 6}

    lrs = adata.uns["lr_summary"].index.values[0:n_top]
    n_sig = adata.uns["lr_summary"].loc[:, "n_spots_sig"].values
    n_non_sig = adata.uns["lr_summary"].loc[:, "n_spots"].values - n_sig
    rank = list(range(len(n_sig)))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(rank[0:n_top], n_non_sig[0:n_top], bar_width, color=non_sig_color)
    ax.bar(
        rank[0:n_top],
        n_sig[0:n_top],
        bar_width,
        bottom=n_non_sig[0:n_top],
        color=sig_color,
    )
    ax.set_ylabel("n_spots", font_dict)
    ax.set_xlabel("LRs Ranked (n_spots_sig)", font_dict)
    if show_title:
        ax.set_title("Signficant and non-signficant spots per LR", font_dict)
    ax.legend(labels=["non-sig", "sig"], loc="upper right")
    if n_top <= max_text:
        ax.set_xticks(rank[0:n_top])
        ax.set_xticklabels(lrs, fontdict=xtick_dict)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show:
        plt.show()
    else:
        return fig, ax


def lr_go(
    adata,
    n_top: int = 20,
    highlight_go: list = None,
    figsize=(6, 4),
    rot: float = 50,
    lr_text_fp: dict = None,
    highlight_color: str = "yellow",
    max_text: int = 50,
    show: bool = True,
):
    """Plots the results from the LR GO analysis.

    Parameters
    ----------
    adata: AnnData
        Data object on which st.tl.cci.lr & st.tl.cci.run_lr_go has been called.
    n_top: int
        Specifies the no. of GO terms to show.
    highlight_go: list
        Names of GO terms to highlight in different color & text.
    figsize: tuple
        Size of the figure.
    rot: int
        Rotation of the text labels for the GO terms.
    lr_text_fp: dict
        Dictionary specifying matplotlib text params that control rendering of
        label on the points.
    highlight_color: str
        The color to plot the GO terms specified in highlight_go.
    max_text: int
        If n_top exceeds this number, stop showing the GO text, since can cause
        crowding.
    show: bool
        Whether to show the plot.
    """
    # Making sure LR GO has been run #
    if "lr_go" not in adata.uns:
        raise Exception("Need to run st.tl.cci.run_lr_go() first!")

    go_results = adata.uns["lr_go"]
    gos = go_results.loc[:, "Description"].values.astype(str)
    y = -np.log10(go_results.loc[:, "p.adjust"].values)
    sizes = go_results.loc[:, "Count"].values
    cci_hs.rank_scatter(
        gos[0:n_top],
        y[0:n_top],
        point_sizes=sizes[0:n_top],
        highlight_items=highlight_go,
        lr_text_fp=lr_text_fp,
        highlight_color=highlight_color,
        figsize=figsize,
        y_label="-log10(padjs)",
        x_label="GO Rank",
        height=6,
        color="deepskyblue",
        rot=rot,
        width_ratio=0.4,
        show=show,
        point_size_name="n-genes",
        show_all=n_top <= max_text,
    )


def cci_check(
    adata: AnnData,
    use_label: str,
    figsize=(16, 10),
    cell_label_size=20,
    axis_text_size=18,
    tick_size=14,
    show=True,
):
    """Checks relationship between no. of significant CCI-LR interactions and cell type frequency.

    Parameters
    ----------
    adata: AnnData
        Data on which st.tl.cci.run & st.tl.cci.run_cci has been performed.
    use_label: str
        The cell type label information used when running st.tl.cci.run_cci
    figsize: tuple
        Size of outputted figure.
    cell_label_size: int
        Size of the cell labels put on top of the bar chart.
    axis_text_size: int
        Size of the axis text.
    tick_size: int
        Size of the ticks displayed at bottom of chart.
    show: bool
        Whether to show the plot or not; if false returns figure & axes.
    Returns
    -------
    Figure, Ax1, Ax2
        The figure, axes for the barchart, and twin axes for the lineplot.
    """
    labels = adata.obs[use_label].values.astype(str)
    label_set = np.array(list(adata.obs[use_label].cat.categories))
    colors = get_colors(adata, use_label)
    xs = np.array(list(range(len(label_set))))
    int_dfs = adata.uns[f"per_lr_cci_{use_label}"]

    # Counting!!! #
    cell_counts = []  # Cell type frequencies
    cell_sigs = []  # Cell type significant interactions
    for j, label in enumerate(label_set):
        counts = sum(labels == label)
        cell_counts.append(counts)

        int_count = 0
        for lr in int_dfs:
            int_df = int_dfs[lr]
            label_index = np.where(int_df.index.values == label)[0][0]
            int_bool = int_df.values > 0
            int_count += sum(int_bool[label_index, :])
            int_count += sum(int_bool[:, label_index])
            # prevent double counts
            int_count -= int_bool[label_index, label_index]

        cell_sigs.append(int_count)

    cell_counts = np.array(cell_counts)
    cell_sigs = np.array(cell_sigs)
    order = np.argsort(cell_counts)
    cell_counts = cell_counts[order]
    cell_sigs = cell_sigs[order]
    colors = np.array(colors)[order]
    label_set = label_set[order]

    # Plotting bar plot #
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(xs, cell_counts, color=colors)
    text_dist = max(cell_counts) * 0.015
    fontdict = {"fontweight": "bold", "fontsize": cell_label_size}
    for j in range(len(xs)):
        ax.text(
            xs[j],
            cell_counts[j] + text_dist,
            label_set[j],
            rotation=90,
            fontdict=fontdict,
        )
    axis_text_fp = {"fontweight": "bold", "fontsize": axis_text_size}
    ax.set_ylabel("Cell type frequency", color="black", **axis_text_fp)
    ax.spines["top"].set_visible(False)
    ax.tick_params(labelsize=tick_size)
    ax.set_xlabel("Cell type rank", **axis_text_fp)

    # Line-plot of the interaction counts #
    ax2 = ax.twinx()
    ax2.set_ylabel("CCI-LR interactions", color="blue", **axis_text_fp)
    ax2.plot(xs, cell_sigs, color="blue", linewidth=2)
    ax2.tick_params(axis="y", labelcolor="blue", labelsize=tick_size)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=tick_size)
    fig.tight_layout()

    if show:
        plt.show()
    else:
        return fig, ax, ax2


# Functions for visualisation the LR results per spot.
def lr_result_plot(
    adata: AnnData,
    use_lr: Optional["str"] = None,
    use_result: Optional["str"] = "lr_sig_scores",
    # plotting param
    title: Optional["str"] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = "Spectral_r",
    ax: Optional[matplotlib.axes._subplots.Axes] = None,
    fig: Optional[matplotlib.figure.Figure] = None,
    show_plot: Optional[bool] = True,
    show_axis: Optional[bool] = False,
    show_image: Optional[bool] = True,
    show_color_bar: Optional[bool] = True,
    zoom_coord: Optional[float] = None,
    crop: Optional[bool] = True,
    margin: Optional[float] = 100,
    size: Optional[float] = 7,
    image_alpha: Optional[float] = 1.0,
    cell_alpha: Optional[float] = 1.0,
    use_raw: Optional[bool] = False,
    fname: Optional[str] = None,
    dpi: Optional[int] = 120,
    contour: bool = False,
    step_size: Optional[int] = None,
    vmin: float = None,
    vmax: float = None,
):
    """Plots the per spot statistics for given LR.

    Parameters
    ----------
    adata: AnnData
        Data on which st.tl.cci.run has been performed
    use_lr: str
        LR to show results for.
    use_result: str
        LR matrix in data.obsm; 'lr_scores', 'lr_sig_scores', 'p_vals', 'p_adjs'
        '-log10(p_adjs)'.
    title: str
        Plot title.
    figsize: tuple
        Figure size.
    cmap: str
        Color of points.
    ax: Axes
        Axes on which to plot.
    fig: Figure
        Figure associated with axes.
    show_plot: bool
        Whether to show plot or not.
    show_axis: bool
        Whether to show axis or not.
    show_image: bool
        Whether to plot the image.
    show_color_bar: bool
        Whether to show the color bar.
    crop: bool
        Whether to crop the image down to match spatial coordinates of gene
        spot present.
    margin: float
        Margin around the points for the image.
    size: float
        Size of the points.
    image_alpha: float
        Transparency of image.
    cell_alpha: float
        Transparency of points.
    use_raw: bool
        Whether to use adata.raw or not.
    fname: str
        Name of file to save plot to.
    dpi: int
        Plot saving quality.
    contour: bool
        Whether to plot as contour.
    step_size: int
        Step size of contour==True
    vmin: float
        Minimum value of scale bar.
    vmax: float
        Maximum value of scale bar.
    """
    LrResultPlot(
        adata,
        use_lr,
        use_result,
        # plotting param
        title,
        figsize,
        cmap,
        None,
        ax,
        fig,
        show_plot,
        show_axis,
        show_image,
        show_color_bar,
        crop,
        zoom_coord,
        margin,
        size,
        image_alpha,
        cell_alpha,
        use_raw,
        fname,
        dpi,
        # cci_rank param
        contour,
        step_size,
        vmin,
        vmax,
    )


# @_docs_params(het_plot=doc_lr_plot)
def lr_plot(
    adata: AnnData,
    lr: str,
    min_expr: float = 0,
    sig_spots=True,
    use_label: str = None,
    outer_mode: str = "continuous",
    l_cmap=None,
    r_cmap=None,
    lr_cmap=None,
    inner_cmap=None,
    inner_size_prop: float = 0.25,
    middle_size_prop: float = 0.5,
    outer_size_prop: float = 1,
    pt_scale: int = 100,
    title="",
    show_image: bool = True,
    show_arrows: bool = False,
    fig: Figure = None,
    ax: Axes = None,
    arrow_head_width: float = 4,
    arrow_width: float = 0.001,
    arrow_cmap: str = None,
    arrow_vmax: float = None,
    sig_cci: bool = False,
    lr_colors: dict = None,
    figsize: tuple = (6.4, 4.8),
    use_mix: bool = None,
    # plotting params
    **kwargs,
) -> Optional[AnnData]:
    """Creates different kinds of spatial visualisations for the LR analysis results.
        To see combinations of parameters refer to stLearn CCI tutorial.

    Parameters
    ----------
    adata: Anndata
        Data on which st.tl.cci.run has been performed; extra options unlocked
        below when have performed st.tl.cci.run_cci as well.
    lr: str
        The LR to display results for.
    min_expr: float
        The minimum expr above which LR considered expressed when plotting
        binary LR expression.
    sig_spots: bool
        Whether to subset to significant spots or not.
    use_label: str
        The cell type labels to use if plotting cell types.
    outer_mode: str
        The mode for the larger points when displaying LR expression;
        can either be 'binary' or 'continuous' or None.
        'Binary' discretizes each spot as expressing L, R, both, or neither.
        'Continuous' shows color gradient for levels of LR expression by plotting
        two points for each spot, the 'inner' point is the receptor expression
        levels, and the 'outer' point is the ligand expression level. None
        plots no ligand/receptor expression.
    l_cmap: str
        Cmap for coloring the ligand expression, only if
        outer_mode=='continuous'.
    r_cmap: str
        Cmap for coloring the receptor expression, only if
        outer_mode=='continuous'.
    lr_cmap: str
        Cmap for coloring coexpression.
    inner_cmap: str
        Cmap for the inner point if outer_mode is 'binary'.
    inner_size_prop: float
        Proportion of the inner point size when plotting to points for one spot.
        Scale of 0 to 1.
    middle_size_prop: float
        Controls size of middle point if specifying parameters that plot
        3 points per spot to display multiple information. Scale 0 to 1.
    outer_size_prop: float
        Point size of the outer point.
    pt_scale: float
        Overall size of point.
    title: str
        Title of figure.
    show_image: bool
        Whether to show the background image.
    show_arrows: bool
        Whether to plot arrows indicating interactions between spots.
    fig: Figure
        Figure to draw on.
    ax: Axes
        Axes to draw on.
    arrow_head_width: float
        Width of arrow head; only if show_arrows is true.
    arrow_width: float
        Width the the arrow body; only if show_arrows is true.
    arrow_cmap: float
        Cmap to color arrows; default is black arrows, but if specified will
        color the arrow by the average expression of the ligand and receptor
        of the spots connected by the arrow.
    arrow_vmax: float
        Maximum value of the arrow colour bar.
    sig_cci: bool
        Whether to only show results which involve signficant celltype-celltype
        interactions; particularly relevant when plotting the arrows.
    lr_colors: dict
        Specifies the colors of the LRs when plotting with outer_mode='binary';
        structures is {'l': color, 'r': color, 'lr': color, '': color};
        the last key-value indicates colour for spots not expressing the ligand
        or receptor.
    figsize: tuple
        (width, height) of figure if not inputted.
    kwargs:
        Extra arguments parsed to plotting functions used internally.
    """

    # Input checking #
    l, r = lr.split("_")
    ran_lr = "lr_summary" in adata.uns
    ran_sig = False if not ran_lr else "n_spots_sig" in adata.uns["lr_summary"].columns
    if ran_lr and lr in adata.uns["lr_summary"].index:
        if ran_sig:
            lr_sig = adata.uns["lr_summary"].loc[lr, :].values[1] > 0
        else:
            lr_sig = True
    else:
        lr_sig = False

    if sig_spots and not ran_lr:
        raise Exception(
            "No LR results testing results found, "
            "please run st.tl.cci_rank.run first, or set sig_spots=False."
        )

    elif sig_spots and not lr_sig:
        raise Exception(
            "LR has no significant spots, to visualise anyhow set" "sig_spots=False"
        )

    # Making sure have run_cci first with respective labelling #
    if (
        show_arrows
        and sig_cci
        and use_label
        and f"per_lr_cci_{use_label}" not in adata.uns
    ):
        raise Exception(
            "Cannot subset arrow interactions to significant ccis "
            "without performing st.tl.run_cci with "
            f"use_label={use_label} first."
        )

    # Getting which are the allowed stats for the lr to plot #
    if not ran_sig:
        lr_use_labels = ["lr_scores"]
    else:
        lr_use_labels = [
            "lr_scores",
            "p_val",
            "p_adj",
            "-log10(p_adj)",
            "lr_sig_scores",
        ]

    if type(use_mix) != type(None) and use_mix not in adata.uns:
        raise Exception(
            f"Specified use_mix, but no deconvolution results added "
            "to adata.uns matching the use_mix ({use_mix}) key."
        )
    elif (
        type(use_label) != type(None)
        and use_label in lr_use_labels
        and ran_sig
        and not lr_sig
    ):
        raise Exception(
            f"Since use_label refers to lr stats & ran permutation testing, "
            f"LR needs to be significant to view stats."
        )
    elif (
        type(use_label) != type(None)
        and use_label not in adata.obs.keys()
        and use_label not in lr_use_labels
    ):
        raise Exception(
            f"use_label must be in adata.obs or " f"one of lr stats: {lr_use_labels}."
        )

    out_options = ["binary", "continuous", None]
    if outer_mode not in out_options:
        raise Exception(f"{outer_mode} should be one of {out_options}")

    if l not in adata.var_names or r not in adata.var_names:
        raise Exception("L or R not found in adata.var_names.")

    # Whether to show just the significant spots or all spots
    lr_index = np.where(adata.uns["lr_summary"].index.values == lr)[0][0]
    sig_bool = adata.obsm["lr_sig_scores"][:, lr_index] > 0
    if sig_spots:
        adata_full = adata
        adata = adata[sig_bool, :]
    else:
        adata_full = adata

    # Dealing with the axis #
    if type(fig) == type(None) or type(ax) == type(None):
        fig, ax = plt.subplots(figsize=figsize)

    expr = adata.to_df()
    l_expr = expr.loc[:, l].values
    r_expr = expr.loc[:, r].values
    # Adding binary points of the ligand/receptor pair #
    if outer_mode == "binary":
        l_bool, r_bool = l_expr > min_expr, r_expr > min_expr
        lr_binary_labels = []
        for i in range(len(l_bool)):
            if l_bool[i] and not r_bool[i]:
                lr_binary_labels.append(l)
            elif not l_bool[i] and r_bool[i]:
                lr_binary_labels.append(r)
            elif l_bool[i] and r_bool[i]:
                lr_binary_labels.append(lr)
            elif not l_bool[i] and not r_bool[i]:
                lr_binary_labels.append("")
        lr_binary_labels = pd.Series(
            np.array(lr_binary_labels), index=adata.obs_names
        ).astype("category")
        adata.obs[f"{lr}_binary_labels"] = lr_binary_labels

        if type(lr_cmap) == type(None):
            lr_cmap = "default"  # This gets ignored due to setting colours below
            if type(lr_colors) == type(None):
                lr_colors = {
                    l: matplotlib.colors.to_hex("r"),
                    r: matplotlib.colors.to_hex("limegreen"),
                    lr: matplotlib.colors.to_hex("b"),
                    "": "#836BC6",  # Neutral color in H&E images.
                }

            label_set = adata.obs[f"{lr}_binary_labels"].cat.categories
            adata.uns[f"{lr}_binary_labels_colors"] = [
                lr_colors[label] for label in label_set
            ]
        else:
            lr_cmap = check_cmap(lr_cmap)

        cluster_plot(
            adata,
            use_label=f"{lr}_binary_labels",
            cmap=lr_cmap,
            size=outer_size_prop * pt_scale,
            crop=False,
            ax=ax,
            fig=fig,
            show_image=show_image,
            show_plot=False,
            **kwargs,
        )

    # Showing continuous gene expression of the LR pair #
    elif outer_mode == "continuous":
        if type(l_cmap) == type(None):
            l_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "lcmap", [(0, 0, 0), (0.5, 0, 0), (0.75, 0, 0), (1, 0, 0)]
            )
        else:
            l_cmap = check_cmap(l_cmap)
        if type(r_cmap) == type(None):
            r_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "rcmap", [(0, 0, 0), (0, 0.5, 0), (0, 0.75, 0), (0, 1, 0)]
            )
        else:
            r_cmap = check_cmap(r_cmap)

        gene_plot(
            adata,
            gene_symbols=l,
            size=outer_size_prop * pt_scale,
            cmap=l_cmap,
            color_bar_label=l,
            ax=ax,
            fig=fig,
            crop=False,
            show_image=show_image,
            **kwargs,
        )
        gene_plot(
            adata,
            gene_symbols=r,
            size=middle_size_prop * pt_scale,
            cmap=r_cmap,
            color_bar_label=r,
            ax=ax,
            fig=fig,
            crop=False,
            show_image=show_image,
            **kwargs,
        )

    # Adding the cell type labels #
    if type(use_label) != type(None):
        if use_label in lr_use_labels:
            inner_cmap = inner_cmap if type(inner_cmap) != type(None) else "copper"
            # adata.obsm[f'{lr}_{use_label}'] = adata.uns['per_lr_results'][
            #                          lr].loc[adata.obs_names,use_label].values
            lr_result_plot(
                adata,
                use_lr=lr,
                show_image=show_image,
                cmap=inner_cmap,
                crop=False,
                ax=ax,
                fig=fig,
                size=inner_size_prop * pt_scale,
                **kwargs,
            )
        else:
            inner_cmap = inner_cmap if type(inner_cmap) != type(None) else "default"
            cluster_plot(
                adata,
                use_label=use_label,
                cmap=inner_cmap,
                size=inner_size_prop * pt_scale,
                crop=False,
                ax=ax,
                fig=fig,
                show_image=show_image,
                show_plot=False,
                **kwargs,
            )

    # Adding in labels which show the interactions between signicant spots &
    # neighbours
    if show_arrows:
        l_expr = adata_full[:, l].X.toarray()[:, 0]
        r_expr = adata_full[:, r].X.toarray()[:, 0]

        if sig_cci:
            int_df = adata.uns[f"per_lr_cci_{use_label}"][lr]
        else:
            int_df = None

        cci_hs.add_arrows(
            adata_full,
            l_expr,
            r_expr,
            min_expr,
            sig_bool,
            fig,
            ax,
            use_label,
            int_df,
            arrow_head_width,
            arrow_width,
            arrow_cmap,
            arrow_vmax,
        )

    # Cropping #
    # if crop:
    #     x0, x1 = ax.get_xlim()
    #     y0, y1 = ax.get_ylim()
    #     x_margin, y_margin = (x1-x0)*margin_ratio, (y1-y0)*margin_ratio
    #     print(x_margin, y_margin)
    #     print(x0, x1, y0, y1)
    #     ax.set_xlim(x0 - x_margin, x1 + x_margin)
    #     ax.set_ylim(y0 - y_margin, y1 + y_margin)
    #     #ax.set_ylim(ax.get_ylim()[::-1])

    fig.suptitle(title)


#### het_plot currently out of date;
#### from old data structure when only test individual LRs.
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
    zoom_coord: Optional[float] = None,
    crop: Optional[bool] = True,
    margin: Optional[bool] = 100,
    size: Optional[float] = 7,
    image_alpha: Optional[float] = 1.0,
    cell_alpha: Optional[float] = 1.0,
    use_raw: Optional[bool] = False,
    fname: Optional[str] = None,
    dpi: Optional[int] = 120,
    # cci_rank param
    use_het: Optional[str] = "het",
    contour: bool = False,
    step_size: Optional[int] = None,
    vmin: float = None,
    vmax: float = None,
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
        zoom_coord=zoom_coord,
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
        vmin=vmin,
        vmax=vmax,
    )


# Functions relating to visualising celltype-celltype interactions after
# calling: st.tl.cci.run_cci


def ccinet_plot(
    adata: AnnData,
    use_label: str,
    lr: str = None,
    pos: dict = None,
    return_pos: bool = False,
    cmap: str = "default",
    font_size: int = 12,
    node_size_exp: int = 1,
    node_size_scaler: int = 1,
    min_counts: int = 0,
    sig_interactions: bool = True,
    fig: matplotlib.figure.Figure = None,
    ax: matplotlib.axes.Axes = None,
    pad=0.25,
    title: str = None,
    figsize: tuple = (10, 10),
):
    """Circular celltype-celltype interaction network based on LR-CCI analysis.
    The size of the nodes drawn for each cell type indicates the total no. of
    spot interactions that cell type is involved in; while the color of the
    arrows between nodes is coloured by the total no. of interactions between
    those particular cell types.

    Parameters
    ----------
    adata: AnnData
        Data on which st.tl.cci.run & st.tl.cci.run_cci has been applied.
    use_label: str
        Indicates the cell type labels or deconvolution results used for
        cell-cell interaction counting by LR pairs.
    lr: str
        The LR pair to visualise the cci network for. If None, will use spot
        cci counts across all LR pairs from adata.uns[f'lr_cci_{use_label}'].
    pos: dict
        Positions to draw each cell type, format as outputted from running
        networkx.circular_layout(graph). If not inputted will be generated.
    return_pos: bool
        Whether to return the positions of the cell types drawn or not;
        useful for input back into this function via the 'pos' parameter to get
        consistent positioning of the cell types when plotting for different LR
        pairs.
    cmap: str
        Cmap to use when generating the cell colors, if not already specified by
        adata.uns[f'{use_label}_colors'].
    font_size: int
        Size of the cell type labels.
    node_size_scaler: float
        Scaler to multiply by node sizes to increase/decrease size.
    node_size_exp: int
        Increases difference between node sizes by this exponent.
    min_counts: int
        Minimum no. of LR interactions for connection to be drawn.

    Returns
    -------
    pos: dict
        Dictionary of positions where the nodes are draw if return_pos is True, useful for consistent layouts.
    """
    cmap, cmap_n = get_cmap(cmap)
    # Making sure adata in correct state that this function should run #
    if f"lr_cci_{use_label}" not in adata.uns:
        raise Exception(
            "Need to first call st.tl.run_cci with the equivalnt "
            "use_label to visualise cell-cell interactions."
        )
    elif type(lr) != type(None) and lr not in adata.uns[f"per_lr_cci_{use_label}"]:
        raise Exception(
            f"{lr} not found in {f'per_lr_cci_{use_label}'}, "
            "suggesting no significant interactions."
        )

    # Either plotting overall interactions, or just for a particular LR #
    int_df, title = get_int_df(adata, lr, use_label, sig_interactions, title)

    # Creating the interaction graph #
    all_set = int_df.index.values
    int_matrix = int_df.values
    graph = nx.MultiDiGraph()
    int_bool = int_matrix > min_counts
    int_matrix = int_matrix * int_bool
    for i, cell_A in enumerate(all_set):
        if cell_A not in graph:
            graph.add_node(cell_A)
        for j, cell_B in enumerate(all_set):
            if int_bool[i, j]:
                count = int_matrix[i, j]
                graph.add_edge(cell_A, cell_B, weight=count)

    # Determining graph layout, node sizes, & edge colours #
    if type(pos) == type(None):
        pos = nx.circular_layout(graph)  # position the nodes using the layout
    total = sum(sum(int_matrix))
    node_names = list(graph.nodes.keys())
    node_indices = [np.where(all_set == node_name)[0][0] for node_name in node_names]
    node_sizes = np.array(
        [
            (
                ((sum(int_matrix[i, :] + int_matrix[:, i]) - int_matrix[i, i]) / total)
                * 10000
                * node_size_scaler
            )
            ** (node_size_exp)
            for i in node_indices
        ]
    )
    node_sizes[node_sizes == 0] = 0.1  # pseudocount

    edges = list(graph.edges.items())
    e_totals = []
    for i, edge in enumerate(edges):
        trans_i = np.where(all_set == edge[0][0])[0][0]
        receive_i = np.where(all_set == edge[0][1])[0][0]
        e_total = (
            sum(list(int_matrix[trans_i, :]) + list(int_matrix[:, receive_i]))
            - int_matrix[trans_i, receive_i]
        )  # so don't double count
        e_totals.append(e_total)
    edge_weights = [edge[1]["weight"] / e_totals[i] for i, edge in enumerate(edges)]

    # Determining node colors #
    nodes = np.unique(list(graph.nodes.keys()))
    node_colors = get_colors(adata, use_label, cmap, label_set=nodes)
    if not np.all(np.array(node_names) == nodes):
        nodes_indices = [np.where(nodes == node)[0][0] for node in node_names]
        node_colors = np.array(node_colors)[nodes_indices]

    #### Drawing the graph #####
    if type(fig) == type(None) or type(ax) == type(None):
        fig, ax = plt.subplots(figsize=figsize, facecolor=[0.7, 0.7, 0.7, 0.4])

    # Adding in the self-loops #
    z = 55
    for i, edge in enumerate(edges):
        cell_type = edge[0][0]
        if cell_type != edge[0][1]:
            continue
        x, y = pos[cell_type]
        angle = math.degrees(math.atan(y / x))
        if x > 0:
            angle = angle + 180
        arc = patches.Arc(
            xy=(x, y),
            width=0.3,
            height=0.025,
            lw=5,
            ec=plt.cm.get_cmap("Blues")(edge_weights[i]),
            angle=angle,
            theta1=z,
            theta2=360 - z,
        )
        ax.add_patch(arc)

    # Drawing the main components of the graph #
    edges = nx.draw_networkx(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        arrowstyle="->",
        arrowsize=50,
        width=5,
        font_size=font_size,
        font_weight="bold",
        edge_color=edge_weights,
        edge_cmap=plt.cm.Blues,
        ax=ax,
    )
    fig.suptitle(title, fontsize=30)
    plt.tight_layout()

    # Adding padding #
    xlims = ax.get_xlim()
    ax.set_xlim(xlims[0] - pad, xlims[1] + pad)
    ylims = ax.get_ylim()
    ax.set_ylim(ylims[0] - pad, ylims[1] + pad)

    if return_pos:
        return pos


def cci_map(
    adata: AnnData,
    use_label: str,
    lr: str = None,
    ax: matplotlib.figure.Axes = None,
    show: bool = False,
    figsize: tuple = None,
    cmap: str = "Spectral_r",
    sig_interactions: bool = True,
    title=None,
):
    """Heatmap visualising sender->receivers of cell type interactions.

    Parameters
    ----------
    adata: AnnData
        Data on which st.tl.cci.run & st.tl.cci.run_cci has been applied.
    use_label: str
        Indicates the cell type labels or deconvolution results used for
        cell-cell interaction counting by LR pairs.
    lr: str
        The LR pair to visualise the sender->receiver interactions for.
        If None, will use all pairs via adata.uns[f'lr_cci_{use_label}'].
    ax: Axes
        Axes on which to plot the heatmap, if None then generates own.
    show: bool
        Whether to show the plot or not; if not, then returns ax.
    figsize: tuple
        (width, height), specifies the dimensions of the figure. Only relevant
        if ax=None.
    cmap: str
        Cmap used to color the number of LR interactions.
    sig_interactions: bool
        Whether to only show significant CCIs, or all observed interactions.
    title: None
        Title to display over the heatmap. If not provided, will be determined
        based on the run parameters.

    Returns
    -------
    ax: matplotlib.figure.Axes
        Axes where the heatmap was drawn if show=False.
    """

    # Either plotting overall interactions, or just for a particular LR #
    int_df, title = get_int_df(adata, lr, use_label, sig_interactions, title)

    if type(figsize) == type(None):  # Adjust size depending on no. cell types
        add = np.array([int_df.shape[0] * 0.1, int_df.shape[0] * 0.05])
        figsize = tuple(np.array([6.4, 4.8]) + add)

    # Rank by total interactions #
    int_vals = int_df.values
    total_ints = int_vals.sum(axis=1) + int_vals.sum(axis=0) - int_vals.diagonal()
    order = np.argsort(-total_ints)
    int_df = int_df.iloc[order, order[::-1]]

    # Reformat the interaction df #
    flat_df = create_flat_df(int_df)

    ax = _box_map(
        flat_df["x"],
        flat_df["y"],
        flat_df["value"].astype(int),
        ax=ax,
        figsize=figsize,
        cmap=cmap,
    )

    ax.set_ylabel("Sender")
    ax.set_xlabel("Receiver")
    plt.suptitle(title)

    if show:
        plt.show()
    else:
        return ax


def lr_cci_map(
    adata: AnnData,
    use_label: str,
    lrs: list or np.array = None,
    n_top_lrs: int = 5,
    n_top_ccis: int = 15,
    min_total: int = 0,
    ax: matplotlib.figure.Axes = None,
    figsize: tuple = (6.48, 4.8),
    show: bool = False,
    cmap: str = "Spectral_r",
    square_scaler: int = 700,
    sig_interactions: bool = True,
):
    """Heatmap of interaction counts.
        Rows are lrs and columns are celltype->celltype interactions.

    Parameters
    ----------
    adata: AnnData
        Data on which st.tl.cci.run & st.tl.cci.run_cci has been applied.
    use_label: str
        Indicates the cell type labels or deconvolution results used for
        the cell-cell interaction counting by LR pairs.
    lrs: list-like
        LR pairs to show in the heatmap, if None then top 5 lrs with highest no.
        of interactions used from adata.uns['lr_summary'].
    n_top_lrs: int
        Indicates how many top lrs to show; is ignored if lrs is not None.
    n_top_ccis: int
        Indicates maximum no. of CCIs to show.
    min_total: int
        Minimum no. of totals interaction celltypes must have to be shown.
    ax: Axes
        Axes on which to draw the heatmap, is generated internally if None.
    figsize: tuple
        (width, height), only relevant if ax=None.
    show: bool
        Whether to show the plot or not, if not returns ax.
    cmap: str
        Cmap used to color the number of LR interactions.
    square_scaler: int
        Scaler to size the squares displayed.
    sig_interactions: bool
        Whether to only show significant CCIs, or all observed interactions.

    Returns
    -------
    ax: matplotlib.figure.Axes
        Axes where the heatmap was drawn on if show=False.
    """
    if sig_interactions:
        lr_int_dfs = adata.uns[f"per_lr_cci_{use_label}"]
    else:
        lr_int_dfs = adata.uns[f"per_lr_cci_raw_{use_label}"]

    if type(lrs) == type(None):
        lrs = np.array(list(lr_int_dfs.keys()))
    else:
        lrs = np.array(lrs)
        n_top_lrs = len(lrs)

    # Creating a new int_df with lrs as rows & cell-cell as column #
    cell_types = list(lr_int_dfs.values())[0].index.values.astype(str)
    n_ints = len(cell_types) ** 2
    new_ints = np.zeros((len(lrs), n_ints))
    for lr_i, lr in enumerate(lrs):
        col_i = 0
        int_df = lr_int_dfs[lr]
        ccis = []
        for c_i, cell_i in enumerate(cell_types):
            for c_j, cell_j in enumerate(cell_types):
                new_ints[lr_i, col_i] = int_df.values[c_i, c_j]
                ccis.append("->".join([cell_i, cell_j]))
                col_i += 1
    new_int_df = pd.DataFrame(new_ints, index=lrs, columns=ccis)

    # Filtering out ccis which have few LR interactions #
    total_ints = new_int_df.values.sum(axis=0)
    order = np.argsort(-total_ints)
    new_int_df = new_int_df.iloc[:, order[0:n_top_ccis]]

    # Getting the top_lrs to display by top loadings in PCA #
    if n_top_lrs < len(lrs):
        top_lrs = adata.uns["lr_summary"].index.values[0:n_top_lrs]
        new_int_df = new_int_df.loc[top_lrs, :]

    # Ordering by the no. of interactions #
    cci_ints = new_int_df.values.sum(axis=0)
    cci_order = np.argsort(-cci_ints)
    lr_ints = new_int_df.values.sum(axis=1)
    lr_order = np.argsort(-lr_ints)
    new_int_df = new_int_df.iloc[lr_order, cci_order]

    # Getting a flat version of the array for plotting #
    flat_df = create_flat_df(new_int_df.transpose())
    if flat_df.shape[0] == 0 or flat_df.shape[1] == 0:
        raise Exception(f"No interactions greater than min: {min_total}")

    ax = _box_map(
        flat_df["x"],
        flat_df["y"],
        flat_df["value"].astype(int),
        ax=ax,
        cmap=cmap,
        figsize=figsize,
        square_scaler=square_scaler,
    )

    ax.set_ylabel("LR-pair")
    ax.set_xlabel("Cell-cell interaction")

    if show:
        plt.show()
    else:
        return ax


def lr_chord_plot(
    adata: AnnData,
    use_label: str,
    lr: str = None,
    min_ints: int = 2,
    n_top_ccis: int = 10,
    cmap: str = "default",
    sig_interactions: bool = True,
    label_size: int = 10,
    label_rotation: float = 0,
    title: str = None,
    figsize: tuple = (8, 8),
    show: bool = True,
):
    """Chord diagram of interactions between cell types.
        Note that interaction is measured as the total no. of edges connecting
        two cell types expressing the ligand and/or receptor in significant
        neighbourhoods for given LR pair.

        The chord diagram is read as follows:

        Each cell type has a labelled edge taking up a proportion of the outter circle.
        Chords connecting cell type edges are coloured by the dominant sending cell.
        Each chord linking cell types has an assymetric shape.
        For two cell types, A and B, the side of the chord attached to edge A is sized by
        the total interactions from B->A, where B is expressing the ligand & A
        is expressing the receptor.
        Hence, the proportion of a cell type's edge in the chordplot circle
        represents the total input signals to that cell type; while the
        area of the chordplot circle taken up by the outputted chords from a given
        cell type represents the total output signals from that cell type.

    Parameters
    ----------
    adata: AnnData
        Data on which st.tl.cci.run & st.tl.cci.run_cci has been applied.
    use_label: str
        Indicates the cell type labels or deconvolution results used for
        cell-cell interaction counting by LR pairs.
    lr: str
        The LR pair to visualise the CCIs for.
        If None, will use all pairs via adata.uns[f'lr_cci_{use_label}'].
    min_ints: int
        Minimum no. of interactions celltypes must have to be shown.
    n_top_ccis: int
        Maximum no. of CCIs to show, will take the top number of these to display.
    cmap: str
        Cmap to use to get colors if colors not already in adata.uns[f'{use_label}_colors']
    sig_interactions: bool
        Whether to show only significant CCIs or all interaction counts.
    label_size: str
        The size of the cell type labels to render.
    label_rotation: float
        Rotation of the cell type label text.
    title: str
        The title above the plot; informative default is determined based on input.
    figsize: tuple
        Figure dimensions.
    show: bool
        Show or not; if not return figure & axes.
    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure on which the heatmap was drawn if show=False.
    ax: matplotlib.figure.Axes
        Axes where the heatmap was drawn on if show=False.
    """
    # Either plotting overall interactions, or just for a particular LR #
    int_df, title = get_int_df(adata, lr, use_label, sig_interactions, title)

    int_df = int_df.transpose()
    fig = plt.figure(figsize=figsize)

    flux = int_df.values
    total_ints = flux.sum(axis=1) + flux.sum(axis=0) - flux.diagonal()
    keep = np.where(total_ints > min_ints)[0]
    # Limit of 10 for good display #
    if len(keep) > n_top_ccis:
        keep = np.argsort(-total_ints)[0:n_top_ccis]
    # Filter any with all zeros after filtering #
    all_zero = np.array(
        [np.all(np.logical_and(flux[i, keep] == 0, flux[keep, i] == 0)) for i in keep]
    )
    keep = keep[all_zero == False]
    if len(keep) == 0:  # If we don't keep anything, warn the user
        print(
            f"Warning: for {lr} at the current min_ints ({min_ints}), there "
            f"are no interaction to display. Adjust min_ints to a lower value"
            f" to visualise chordplot for this LR."
        )
        return

    flux = flux[:, keep]
    flux = flux[keep, :].astype(float)

    # Add pseudocount to row/column which has all zeros for the incoming
    # so can make the connection between the two
    for i in range(flux.shape[0]):
        if np.all(flux[i, :] == 0):
            flux[i, flux[:, i] > 0] += sys.float_info.min
        elif np.all(flux[:, i] == 0):
            flux[flux[i, :] > 0, i] += sys.float_info.min

    cell_names = int_df.index.values.astype(str)[keep]
    nodes = cell_names

    # Retrieving colors of cell types #
    colors = get_colors(adata, use_label, cmap=cmap, label_set=cell_names)

    ax = plt.axes([0, 0, 1, 1])
    nodePos = chordDiagram(flux, ax, lim=1.25, colors=colors)
    ax.axis("off")
    prop = dict(fontsize=label_size, ha="center", va="center")
    label_rotation_ = label_rotation
    for i in range(len(cell_names)):
        x, y = nodePos[i][0:2]
        rotation = nodePos[i][2]
        # Prevent text going upside down at certain rotations
        if (rotation < 90 and rotation > 18 and label_rotation != 0) or (
            rotation < 120 and rotation > 90
        ):
            label_rotation_ = -label_rotation
        else:
            label_rotation_ = label_rotation
        ax.text(
            x, y, nodes[i], rotation=nodePos[i][2] + label_rotation_, **prop
        )  # size=10,
    fig.suptitle(title, fontsize=12, fontweight="bold")
    if show:
        plt.show()
    else:
        return fig, ax


def grid_plot(
    adata,
    use_label: str = None,
    n_row: int = 10,
    n_col: int = 10,
    size: int = 1,
    figsize=(4.5, 4.5),
    show: bool = False,
):
    """Plots grid over the top of spatial data to show how cells will be grouped if gridded.

    Parameters
    ----------
    adata: AnnData
        Data on which st.tl.cci.run & st.tl.cci.run_cci has been applied.
    use_label: str
        Indicates the cell type labels or deconvolution results used for
        cell-cell interaction counting by LR pairs.
    n_row: str
        The number of rows in the grid.
    n_col: int
        The number of columns in the grid.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure on which the heatmap is draw if show=False.
    ax: matplotlib.figure.Axes
        Axes where the heatmap was drawn on if show=False.
    """
    xs, ys = adata.obs["imagecol"].values, adata.obs["imagerow"].values
    grid_counts, xedges, yedges = np.histogram2d(xs, ys, bins=[n_col, n_row])
    xmin, xmax = min(xedges), max(xedges)
    ymin, ymax = min(yedges), max(yedges)

    fig, ax = plt.subplots(figsize=figsize)

    # Plotting the points #
    if type(use_label) != type(None):
        if f"{use_label}_colors" in adata.uns:
            color_map = {}
            for i, ct in enumerate(adata.obs[use_label].cat.categories):
                color_map[ct] = adata.uns[f"{use_label}_colors"][i]
            cell_colors = [color_map[ct] for ct in adata.obs[use_label]]
    else:  # Otherwise plot by cell density #
        stack = np.vstack([xs, ys])
        cell_colors = gaussian_kde(stack)(stack)

    ax.scatter(xs, -ys, s=size, c=cell_colors)
    ax.vlines(xedges, -ymin, -ymax, color="#36454F")
    ax.hlines(-yedges, xmin, xmax, color="#36454F")

    if show:
        plt.show()
    else:
        return fig, ax


####################### Bokeh Interactive Plots ################################
def lr_plot_interactive(adata: AnnData):
    """Plots the LR scores for significant spots interatively using Bokeh.

    Parameters
    ----------
    adata: AnnData
        Data on which st.tl.cci.run has been applied.
    """
    bokeh_object = BokehLRPlot(adata)
    output_notebook()
    show(bokeh_object.app, notebook_handle=True)


def spatialcci_plot_interactive(adata: AnnData):
    """Plots the significant CCI in the spatial context interactively using Bokeh.

    Parameters
    ----------
    adata: AnnData
        Data on which st.tl.cci.run & st.tl.cci.run_cci has been applied.
    """
    bokeh_object = BokehSpatialCciPlot(adata)
    output_notebook()
    show(bokeh_object.app, notebook_handle=True)


# def het_plot_interactive(adata: AnnData):
#     bokeh_object = BokehCciPlot(adata)
#     output_notebook()
#     show(bokeh_object.app, notebook_handle=True)


# Bokeh & old grid plots;
# has not been tested since multi-LR testing implimentation.

# def het_plot_interactive(adata: AnnData):
#     bokeh_object = BokehCciPlot(adata)
#     output_notebook()
#     show(bokeh_object.app, notebook_handle=True)


# def grid_plot(
# 	adata: AnnData,
# 	use_het: str = None,
# 	num_row: int = 10,
# 	num_col: int = 10,
# 	vmin: float = None,
# 	vmax: float = None,
# 	cropped: bool = True,
# 	margin: int = 100,
# 	dpi: int = 100,
# 	name: str = None,
# 	output: str = None,
# 	copy: bool = False,
# ) -> Optional[AnnData]:
#
# 	"""
# 	Cell diversity plot for sptial transcriptomics data.
#
# 	Parameters
# 	----------
# 	adata:                  Annotated data matrix.
# 	use_het:                Cluster heterogeneity count results from tl.cci_rank.het
# 	num_row: int            Number of grids on height
# 	num_col: int            Number of grids on width
# 	cropped                 crop image or not.
# 	margin                  margin used in cropping.
# 	dpi:                    Set dpi as the resolution for the plot.
# 	name:                   Name of the output figure file.
# 	output:                 Save the figure as file or not.
# 	copy:                   Return a copy instead of writing to adata.
#
# 	Returns
# 	-------
# 	Nothing
# 	"""
#
# 	try:
# 		import seaborn as sns
# 	except:
# 		raise ImportError("Please run `pip install seaborn`")
# 	plt.subplots()
#
# 	sns.heatmap(
# 		pd.DataFrame(np.array(adata.obsm[use_het]).reshape(num_col, num_row)).T,
# 		vmin=vmin,
# 		vmax=vmax,
# 	)
# 	plt.axis("equal")
#
# 	if output is not None:
# 		plt.savefig(
# 			output + "/" + name + "_heatmap.pdf",
# 			dpi=dpi,
# 			bbox_inches="tight",
# 			pad_inches=0,
# 		)
#
# 	plt.show()
