""" Helper functions for cci_plot.py.
"""

import sys
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Arc, Wedge
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.colors as plt_colors
import matplotlib.cm as cm

from ..tools.microenv.cci.het import get_edges

from anndata import AnnData

# Helper functions for overview plots of the LRs.


def lr_scatter(
    data,
    feature,
    highlight_lrs=None,
    show_text=True,
    n_top=50,
    color="gold",
    alpha=0.5,
    lr_text_fp=None,
    axis_text_fp=None,
    ax=None,
    show=True,
    max_text=100,
    highlight_color="red",
    figsize: tuple = None,
    show_all: bool = False,
):
    """General plotting of the LR features."""
    highlight = type(highlight_lrs) != type(None)
    if not highlight:
        show_text = show_text if n_top <= max_text else False
    else:
        highlight_lrs = highlight_lrs[0:max_text]

    lr_df = data.uns["lr_summary"]
    lrs = lr_df.index.values.astype(str)[0:n_top]
    lr_features = data.uns["lrfeatures"]
    lr_df = pd.concat([lr_df, lr_features], axis=1).loc[lrs, :]
    if feature not in lr_df.columns:
        raise Exception(f"Inputted {feature}; must be one of " f"{list(lr_df.columns)}")

    rot = 90 if feature != "n_spots_sig" else 70

    n_spots = lr_df.loc[:, feature].values[0:n_top]

    return rank_scatter(
        lrs,
        n_spots,
        y_label=feature,
        x_label="LR Rank (n_sig_spots)",
        figsize=figsize,
        highlight_items=highlight_lrs,
        show_text=show_text,
        color=color,
        alpha=alpha,
        lr_text_fp=lr_text_fp,
        axis_text_fp=axis_text_fp,
        ax=ax,
        show=show,
        highlight_color=highlight_color,
        rot=rot,
        pad=0,
        show_all=show_all,
    )
    # ranks = np.array(list(range(len(n_spots))))
    #
    # if type(lr_text_fp)==type(None):
    #     lr_text_fp = {'weight': 'bold', 'size': 8}
    # if type(axis_text_fp)==type(None):
    #     axis_text_fp = {'weight': 'bold', 'size': 12}
    #
    # if type(ax)==type(None):
    #     width = (7.5 / 50) * n_top if show_text and not highlight else 7.5
    #     if width > 20:
    #         width = 20
    #     fig, ax = plt.subplots(figsize=(width, 4))
    #
    # # Plotting the points #
    # ax.scatter(ranks, n_spots, alpha=alpha, c=color)
    #
    # if show_text:
    #     if highlight:
    #         ranks = ranks[[np.where(lrs==lr)[0][0] for lr in highlight_lrs]]
    #         ax.scatter(ranks, n_spots[ranks], alpha=alpha, c=highlight_color)
    #
    #     for i in ranks:
    #         ax.text(i-.2, n_spots[i], lrs[i], rotation=rot, fontdict=lr_text_fp)
    #
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.set_xlabel('LR Rank', axis_text_fp)
    # ax.set_ylabel(feature, axis_text_fp)
    #
    # if show:
    #     plt.show()
    # else:
    #     return ax


def rank_scatter(
    items,
    y,
    y_label: str = "",
    x_label: str = "",
    highlight_items=None,
    show_text=True,
    color="gold",
    alpha=0.5,
    lr_text_fp=None,
    axis_text_fp=None,
    ax=None,
    show=True,
    highlight_color="red",
    rot: float = 90,
    point_sizes: np.array = None,
    pad=0.2,
    figsize=None,
    width_ratio=7.5 / 50,
    height=4,
    point_size_name="Sizes",
    point_size_exp=2,
    show_all: bool = False,
):
    """General plotting function for showing ranked list of items."""
    ranks = np.array(list(range(len(items))))

    highlight = type(highlight_items) != type(None)
    if type(lr_text_fp) == type(None):
        lr_text_fp = {"weight": "bold", "size": 8}
    if type(axis_text_fp) == type(None):
        axis_text_fp = {"weight": "bold", "size": 12}

    if type(ax) == type(None):
        if type(figsize) == type(None):
            width = width_ratio * len(ranks) if show_text and not highlight else 7.5
            if width > 20:
                width = 20
            figsize = (width, height)
        fig, ax = plt.subplots(figsize=figsize)

    # Plotting the points #
    scatter = ax.scatter(
        ranks,
        y,
        alpha=alpha,
        c=color,
        s=None if type(point_sizes) == type(None) else point_sizes ** point_size_exp,
        edgecolors="none",
    )
    y_min, y_max = ax.get_ylim()
    y_max = y_max + y_max * pad
    ax.set_ylim(y_min, y_max)
    if type(point_sizes) != type(None):
        # produce a legend with a cross section of sizes from the scatter
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4)
        [handle.set_markeredgecolor("none") for handle in handles]
        starts = [label.find("{") for label in labels]
        ends = [label.find("}") + 1 for label in labels]
        sizes = [
            float(label[(starts[i] + 1) : (ends[i] - 1)])
            for i, label in enumerate(labels)
        ]
        counts = [int(size ** (1 / point_size_exp)) for size in sizes]
        labels2 = [
            label.replace(label[(starts[i]) : (ends[i])], "{" + str(counts[i]) + "}")
            for i, label in enumerate(labels)
        ]
        legend2 = ax.legend(
            handles,
            labels2,
            frameon=False,
            # bbox_to_anchor=(0.1, 0.05, 1., 1.),
            handletextpad=1.6,
            loc="upper right",
            title=point_size_name,
        )

    if show_text:
        if highlight:
            ranks_ = ranks[[np.where(items == item)[0][0] for item in highlight_items]]
            ax.scatter(
                ranks_,
                y[ranks_],
                alpha=alpha,
                c=highlight_color,
                s=None
                if type(point_sizes) == type(None)
                else (point_sizes[ranks_] ** point_size_exp),
                edgecolors=color,
            )
            ranks = ranks_ if not show_all else ranks

        for i in ranks:
            ax.text(i - 0.2, y[i], items[i], rotation=rot, fontdict=lr_text_fp)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(x_label, axis_text_fp)
    ax.set_ylabel(y_label, axis_text_fp)

    if show:
        plt.show()
    else:
        return ax


# Helper functions for lr_plot


def add_arrows(
    adata: AnnData,
    l_expr: np.array,
    r_expr: np.array,
    min_expr: float,
    sig_bool: np.array,
    fig,
    ax: Axes,
    use_label: str,
    int_df: pd.DataFrame,
    head_width=4,
    width=0.001,
    arrow_cmap=None,
    arrow_vmax=None,
):
    """ Adds arrows to the current plot for significant spots to neighbours \
        which is interacting with.
        Parameters
        ----------
        adata: AnnData          The anndata object.
        L_bool: np.array
        Returns
        -------
        counts: int   Total number of interactions satisfying the conditions, \
                      or np.array<set> if return_edges=True, where each set is \
                      an edge, only returns unique edges.
    """

    library_id = list(adata.uns["spatial"].keys())[0]
    res = adata.uns["spatial"][library_id]["use_quality"]
    scale_factor = adata.uns["spatial"][library_id]["scalefactors"][
        f"tissue_{res}_scalef"
    ]
    # scale_factor = 1
    # scale_factor = 1 # scale factor pretty much isn't used since always parsed as 1
    # in the base plotting function class.
    # Reason why is because scale_factor refers to scaling the
    # image to match the spot spatial coordinates, not the
    # the spots to match the image coordinates!!!

    L_bool = l_expr > min_expr
    R_bool = r_expr > min_expr

    # Getting the edges, from sig-L->R and sig-R<-L #
    forward_edges, reverse_edges = get_edges(adata, L_bool, R_bool, sig_bool)

    # If int_df specified, means need to subset to CCIs which are significant #
    if type(int_df) != type(None):
        spot_bcs = adata.obs_names.values.astype(str)
        spot_labels = adata.obs[use_label].values.astype(str)
        label_set = int_df.index.values.astype(str)
        interact_bool = int_df.values > 0

        # Subsetting to only significant CCI #
        edges_sub = [[], []]  # forward, reverse
        # ints_2 = np.zeros(int_df.shape) # Just for debugging make sure edge
        # list re-capitulates edge-counts.
        for i, edges in enumerate([forward_edges, reverse_edges]):
            for j, edge in enumerate(edges):
                k_ = [0, 1] if i == 0 else [1, 0]
                celltype0 = np.where(label_set == spot_labels[spot_bcs == edge[k_[0]]])[
                    0
                ][0]
                celltype1 = np.where(label_set == spot_labels[spot_bcs == edge[k_[1]]])[
                    0
                ][0]
                celltypes = np.array([celltype0, celltype1])
                # For debugging purposes, used to find indexing bug #
                # print(label_set[celltypes[k_[0]]],
                #       label_set[celltypes[k_[1]]])
                # print( celltypes[k_[0]], celltypes[k_[1]] )
                # print( interact_bool[celltypes[k_[0]], celltypes[k_[1]]] )
                if interact_bool[celltypes[k_[0]], celltypes[k_[1]]]:
                    edges_sub[i].append(edge)
                    # ints_2[celltypes[k_[0]], celltypes[k_[1]]] += 1

        forward_edges, reverse_edges = edges_sub

    # If cmap specified, colour arrows by average LR expression on edge #
    if type(arrow_cmap) != type(None):
        edges_means = [[], []]
        all_means = []
        for i, edges in enumerate([forward_edges, reverse_edges]):
            for j, edge in enumerate(edges):
                edge0_bool = spot_bcs == edge[0]
                edge1_bool = spot_bcs == edge[1]
                l_expr0 = l_expr[edge0_bool]
                l_expr1 = l_expr[edge1_bool]
                r_expr0 = r_expr[edge0_bool]
                r_expr1 = r_expr[edge1_bool]
                mean_expr = np.mean([l_expr0, l_expr1, r_expr0, r_expr1])
                edges_means[i].append(mean_expr)
                all_means.append(mean_expr)

        # Determining the color maps #
        arrow_vmax = np.max(all_means) if type(arrow_vmax) == type(None) else arrow_vmax
        cmap = plt.get_cmap(arrow_cmap)
        c_norm = plt_colors.Normalize(vmin=0, vmax=arrow_vmax)
        scalar_map = cm.ScalarMappable(norm=c_norm, cmap=cmap)

        # Determining the edge colors #
        edges_colors = [[], []]
        for i, edges in enumerate([forward_edges, reverse_edges]):
            for j, edge in enumerate(edges):
                color_val = scalar_map.to_rgba(edges_means[i][j])
                edges_colors[i].append(color_val)

        # Need to add new axes #
        # xlims = ax.get_xlim()
        # ylims = ax.get_ylim()
        # left, bottom = xlims[0]*0.025, ylims[0]*0.17
        # axc = fig.add_axes([0, 0, 0.28, 0.015])
        divider = make_axes_locatable(ax)
        cax = divider.new_vertical(size="2%", pad=0.1, pack_start=True)
        axc = fig.add_axes(cax)

    else:
        edges_colors = [None, None]

    # Now performing the plotting #
    # The arrows #
    # Now converting the edges to coordinates #
    add_arrows_by_edges(
        ax,
        adata,
        forward_edges,
        scale_factor,
        head_width,
        width,
        edge_colors=edges_colors[0],
    )
    add_arrows_by_edges(
        ax,
        adata,
        reverse_edges,
        scale_factor,
        head_width,
        width,
        forward=False,
        edge_colors=edges_colors[1],
    )
    # Adding the color map #
    if type(arrow_cmap) != type(None):
        cb1 = matplotlib.colorbar.ColorbarBase(
            axc, cmap=cmap, norm=c_norm, orientation="horizontal"
        )


def add_arrows_by_edges(
    ax,
    adata,
    edges,
    scale_factor,
    head_width,
    width,
    forward=True,
    edge_colors=None,
    axc=None,
):
    """Adds the arrows using an edge list."""
    for i, edge in enumerate(edges):
        # cols = ["imagecol", "imagerow"]
        if forward:
            edge0, edge1 = edge
        else:
            edge0, edge1 = edge[::-1]

        # Arrow details #
        # x1, y1 = adata.obs.loc[edge0, cols].values.astype(float) * scale_factor
        # x2, y2 = adata.obs.loc[edge1, cols].values.astype(float) * scale_factor
        edge0_index = np.where(adata.obs_names.values == edge0)[0][0]
        edge1_index = np.where(adata.obs_names.values == edge1)[0][0]
        x1, y1 = adata.obsm["spatial"][edge0_index, :] * scale_factor
        x2, y2 = adata.obsm["spatial"][edge1_index, :] * scale_factor
        dx, dy = (x2 - x1) * 0.75, (y2 - y1) * 0.75
        arrow_color = "k" if type(edge_colors) == type(None) else edge_colors[i]

        ax.arrow(
            x1,
            y1,
            dx,
            dy,
            head_width=head_width,
            width=width,
            linewidth=0.01,
            facecolor=arrow_color,
        )


# Helper functions for cci_map


def get_int_df(adata, lr, use_label, sig_interactions, title):
    """Retrieves the relevant interaction count matrix."""
    no_title = type(title) == type(None)
    if type(lr) == type(None):  # No LR inputted, so just use all
        int_df = (
            adata.uns[f"lr_cci_{use_label}"]
            if sig_interactions
            else adata.uns[f"lr_cci_raw_{use_label}"]
        )
        title = "Cell-Cell LR Interactions" if no_title else title
    else:
        int_df = (
            adata.uns[f"per_lr_cci_{use_label}"][lr]
            if sig_interactions
            else adata.uns[f"per_lr_cci_raw_{use_label}"][lr]
        )
        title = f"Cell-Cell {lr} interactions" if no_title else title

    return int_df, title


def create_flat_df(int_df):
    """Reformats a dataframe representing interactions to a flat format."""
    n_rows = int_df.shape[0] * int_df.shape[1]
    flat_df = pd.DataFrame(index=list(range(n_rows)), columns=["x", "y", "value"])
    row_i = 0
    for i, cell_typei in enumerate(int_df.index.values):
        for j, cell_typej in enumerate(int_df.columns.values):
            flat_df.iloc[row_i, :] = [cell_typei, cell_typej, int_df.values[i, j]]
            row_i += 1

    return flat_df


def _box_map(x, y, size, ax=None, figsize=(6.48, 4.8), cmap=None, square_scaler=700):
    """Main underlying helper function for generating the heatmaps."""
    if type(cmap) == type(None):
        cmap = "Spectral_r"

    if type(ax) == type(None):
        fig, ax = plt.subplots(figsize=figsize)

    # Mapping from column names to integer coordinates
    x_labels = list(x.values)  # [v for v in sorted(x.unique())]
    y_labels = list(y.values)  # [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    out = ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s=size / sum(size) * square_scaler,
        c=size,
        cmap=cmap,
        # Vector of square sizes, proportional to size parameter
        marker="s",  # Use square as scatterplot marker
    )
    out.set_array(size.values.astype(int))
    out.set_clim(min(size), max(size))
    cbar = plt.colorbar(out)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("# of interactions", rotation=270)

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment="right")
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

    return ax


# Matplotlib chord diagram, adapted & refactored from:
# https://github.com/fengwangPhysics/matplotlib-chord-diagram/blob/master/matplotlib-chord.py

###################
# chord diagram
LW = 0.3


def polar2xy(r, theta):
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def hex2rgb(c):
    return tuple(int(c[i : i + 2], 16) / 256.0 for i in (1, 3, 5))


def IdeogramArc(
    start=0, end=60, radius=1.0, width=0.2, ax=None, color=(1, 0, 0), curve_steps=1
):
    # start, end should be in [0, 360)
    if start > end:
        start, end = end, start
    start *= np.pi / 180.0
    end *= np.pi / 180.0
    # optimal distance to the control points
    # https://stackoverflow.com/questions/1734745/how-to-create-circle-with-b%C3%A9zier-curves
    opt = 4.0 / 3.0 * np.tan((end - start) / 4.0) * radius
    inner = radius * (1 - width)

    # Defines descrete points along the arc #
    # The outter part of the arc #

    val = 0.5
    verts_upper_start = [polar2xy(radius, start)]
    verts_upper_curve = [
        polar2xy(radius, start)
        + polar2xy(opt, start + (np.pi * ((val * i) / curve_steps)))
        for i in range(1, curve_steps + 1)
    ]
    verts_upper_curve += [
        polar2xy(radius, end) + polar2xy(opt, end - (np.pi * ((val * i) / curve_steps)))
        for i in range(1, curve_steps + 1)
    ]
    verts_upper = verts_upper_start + verts_upper_curve + [polar2xy(radius, end)]

    verts_inner_start = [polar2xy(inner, end)]
    verts_inner_curve = [
        polar2xy(inner, end)
        + polar2xy(opt * (1 - width), end - (np.pi * ((val * i) / curve_steps)))
        for i in range(1, curve_steps + 1)
    ]
    verts_inner_curve += [
        polar2xy(inner, start)
        + polar2xy(opt * (1 - width), start + (np.pi * ((val * i) / curve_steps)))
        for i in range(1, curve_steps + 1)
    ]
    verts_inner = (
        verts_inner_start
        + verts_inner_curve
        + [polar2xy(inner, start), polar2xy(radius, start)]
    )

    verts = verts_upper + verts_inner

    codes = (
        [Path.MOVETO]
        + [Path.CURVE4] * curve_steps * 2
        + [Path.CURVE4, Path.LINETO]
        + [Path.CURVE4] * curve_steps * 2
        + [
            Path.CURVE4,
            Path.CLOSEPOLY,
        ]
    )

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, facecolor=color + (0.5,), edgecolor=color + (0.4,), lw=LW
        )
        ax.add_patch(patch)


def ChordArc(
    start1=0,
    end1=60,
    start2=180,
    end2=240,
    radius=1.0,
    chordwidth=0.7,
    ax=None,
    color=(1, 0, 0),
):
    # start, end should be in [0, 360)
    if start1 > end1:
        start1, end1 = end1, start1
    if start2 > end2:
        start2, end2 = end2, start2
    start1 *= np.pi / 180.0
    end1 *= np.pi / 180.0
    start2 *= np.pi / 180.0
    end2 *= np.pi / 180.0
    opt1 = 4.0 / 3.0 * np.tan((end1 - start1) / 4.0) * radius
    opt2 = 4.0 / 3.0 * np.tan((end2 - start2) / 4.0) * radius
    rchord = radius * (1 - chordwidth)
    verts = [
        polar2xy(radius, start1),
        polar2xy(radius, start1) + polar2xy(opt1, start1 + 0.5 * np.pi),
        polar2xy(radius, end1) + polar2xy(opt1, end1 - 0.5 * np.pi),
        polar2xy(radius, end1),
        polar2xy(rchord, end1),
        polar2xy(rchord, start2),
        polar2xy(radius, start2),
        polar2xy(radius, start2) + polar2xy(opt2, start2 + 0.5 * np.pi),
        polar2xy(radius, end2) + polar2xy(opt2, end2 - 0.5 * np.pi),
        polar2xy(radius, end2),
        polar2xy(rchord, end2),
        polar2xy(rchord, start1),
        polar2xy(radius, start1),
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, facecolor=color + (0.5,), edgecolor=color + (0.4,), lw=LW
        )
        ax.add_patch(patch)


def selfChordArc(start=0, end=60, radius=1.0, chordwidth=0.7, ax=None, color=(1, 0, 0)):
    # start, end should be in [0, 360)
    if start > end:
        start, end = end, start
    start *= np.pi / 180.0
    end *= np.pi / 180.0
    opt = 4.0 / 3.0 * np.tan((end - start) / 4.0) * radius
    rchord = radius * (1 - chordwidth)
    verts = [
        polar2xy(radius, start),
        polar2xy(radius, start) + polar2xy(opt, start + 0.5 * np.pi),
        polar2xy(radius, end) + polar2xy(opt, end - 0.5 * np.pi),
        polar2xy(radius, end),
        polar2xy(rchord, end),
        polar2xy(rchord, start),
        polar2xy(radius, start),
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, facecolor=color + (0.5,), edgecolor=color + (0.4,), lw=LW
        )
        ax.add_patch(patch)


def chordDiagram(X, ax, colors=None, width=0.1, pad=2, chordwidth=0.7, lim=1.1):
    """Plot a chord diagram
    Parameters
    ----------
    X :
        flux data, X[i, j] is the flux from i to j
    ax :
        matplotlib `axes` to show the plot
    colors : optional
        user defined colors in rgb format. Use function hex2rgb() to convert hex color to rgb color. Default: d3.js category10
    width : optional
        width/thickness of the ideogram arc
    pad : optional
        gap pad between two neighboring ideogram arcs, unit: degree, default: 2 degree
    chordwidth : optional
        position of the control points for the chords, controlling the shape of the chords
    """
    # X[i, j]:  i -> j
    x = X.sum(axis=1)  # sum over rows
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    diam = 1.8

    if colors is None:
        # use d3.js category10 https://github.com/d3/d3-3.x-api-reference/blob/master/Ordinal-Scales.md#category10
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        if len(x) > 10:
            print("x is too large! Use x smaller than 10")
    if type(colors[0]) == str:
        colors = [hex2rgb(colors[i]) for i in range(len(x))]

    # find position for each start and end
    y = x / np.sum(x).astype(float) * (360 - pad * len(x))

    pos = {}
    arc = []
    nodePos = []
    start = 0
    for i in range(len(x)):
        end = start + y[i]
        arc.append((start, end))
        angle = 0.5 * (start + end)
        # print(start, end, angle)
        if -30 <= angle <= 210:
            angle -= 90
        else:
            angle -= 270
        nodePos.append(
            tuple(
                polar2xy((diam / 2) + diam * 0.05, 0.5 * (start + end) * np.pi / 180.0)
            )
            + (angle,)
        )
        z = (X[i, :] / x[i].astype(float)) * (end - start)
        ids = np.argsort(z)
        z0 = start
        for j in ids:
            pos[(i, j)] = (z0, z0 + z[j])
            z0 += z[j]
        start = end + pad

    for i in range(len(x)):
        start, end = arc[i]
        # This draws the outter ring #
        # IdeogramArc(start=start, end=end, radius=1.0, ax=ax,
        #            color=colors[i], width=width)
        a = Arc((0, 0), diam, diam, 0, start, end, color=colors[i], lw=10)
        ax.add_patch(a)
        start, end = pos[(i, i)]
        # This draws the paths to itself #
        if end - start < 180:  # Indicates this method will work fine !
            selfChordArc(
                start,
                end,
                radius=1.0 - width,
                color=colors[i],
                chordwidth=chordwidth * 0.7,
                ax=ax,
            )
        else:  # Need to use a wedge because the arch distorts past 180-degrees
            path = Wedge(0, diam / 2, start, end, color=colors[i] + (0.5,))
            ax.add_patch(path)
        for j in range(i):
            if X[i, j] == 0 and X[j, i] == 0:  # don't draw anything for no interaction
                continue
            color = colors[i]
            if X[i, j] > X[j, i]:  # Color by the dominant signal #
                color = colors[j]
            start1, end1 = pos[(i, j)]
            start2, end2 = pos[(j, i)]
            ChordArc(
                start1,
                end1,
                start2,
                end2,
                radius=1.0 - width,
                color=color,
                chordwidth=chordwidth,
                ax=ax,
            )

    return nodePos
