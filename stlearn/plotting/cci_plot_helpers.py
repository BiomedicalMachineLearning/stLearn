""" Helper functions for cci_plot.py.
"""

import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Arc, Wedge

from matplotlib.path import Path
import matplotlib.patches as patches

from ..tools.microenv.cci.het import get_edges

from anndata import AnnData

""" Helper functions for lr_plot
"""

def add_arrows(adata: AnnData, L_bool: np.array, R_bool: np.array,
               sig_bool: np.array, ax: Axes):
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
    # TODO the below could cause issues by hardcoding tissue res. #
    scale_factor = adata.uns['spatial'][library_id]['scalefactors'] \
                                                        ['tissue_lowres_scalef']
    scale_factor = 1

    # Getting the edges #
    all_edges_unique = get_edges(adata, L_bool, R_bool, sig_bool)

    # Now performing the plotting #
    # The arrows #
    # Now converting the edges to coordinates #
    for edge in all_edges_unique:
        cols = ['imagecol', 'imagerow']
        x1, y1 = adata.obs.loc[edge[0], cols].values.astype(float) * scale_factor
        x2, y2 = adata.obs.loc[edge[1], cols].values.astype(float) * scale_factor
        dx, dy = x2-x1, y2-y1
        ax.arrow(x1, y1, dx, dy, head_width=4)

""" Helper functions for cci_map
"""

def create_flat_df(int_df):
    """Reformats a dataframe representing interactions to a flat format."""
    n_rows = int_df.shape[0] * int_df.shape[1]
    flat_df = pd.DataFrame(index=list(range(n_rows)),
                           columns=['x', 'y', 'value'])
    row_i = 0
    for i, cell_typei in enumerate(int_df.index.values):
        for j, cell_typej in enumerate(int_df.columns.values):
            flat_df.iloc[row_i, :] = [cell_typei, cell_typej,
                                      int_df.values[i, j]]
            row_i += 1

    return flat_df

def _box_map(x, y, size, ax=None, figsize=(6.48,4.8), cmap=None,
             square_scaler=700):
    """ Main underlying helper function for generating the heatmaps.
    """
    if type(cmap)==type(None):
        cmap = 'Spectral_r'

    if type(ax) == type(None):
        fig, ax = plt.subplots(figsize=figsize)

    # Mapping from column names to integer coordinates
    x_labels = list(x.values) #[v for v in sorted(x.unique())]
    y_labels = list(y.values) #[v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    out = ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s=size / sum(size) * square_scaler,
        c=size, cmap=cmap,
        # Vector of square sizes, proportional to size parameter
        marker='s'  # Use square as scatterplot marker
    )
    out.set_array(size.values.astype(int))
    out.set_clim(min(size), max(size))
    cbar = plt.colorbar(out)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('# of interactions', rotation=270)

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

    return ax

"""
Matplotlib chord diagram, adapted & refactored from: 
https://github.com/fengwangPhysics/matplotlib-chord-diagram/blob/master/matplotlib-chord.py
"""

###################
# chord diagram
LW = 0.3

def polar2xy(r, theta):
    return np.array([r*np.cos(theta), r*np.sin(theta)])

def hex2rgb(c):
    return tuple(int(c[i:i+2], 16)/256.0 for i in (1, 3 ,5))

def IdeogramArc(start=0, end=60, radius=1.0, width=0.2, ax=None, color=(1,0,0),
                curve_steps=1):
    # start, end should be in [0, 360)
    if start > end:
        start, end = end, start
    start *= np.pi/180.
    end *= np.pi/180.
    # optimal distance to the control points
    # https://stackoverflow.com/questions/1734745/how-to-create-circle-with-b%C3%A9zier-curves
    opt = 4./3. * np.tan((end-start)/ 4.) * radius
    inner = radius*(1-width)

    # Defines descrete points along the arc #
    # The outter part of the arc #

    val = .5
    verts_upper_start = [polar2xy(radius, start)]
    verts_upper_curve = [polar2xy(radius, start) +
                        polar2xy(opt, start+(np.pi*((val*i)/curve_steps)))
                        for i in range(1,curve_steps+1)]
    verts_upper_curve += [polar2xy(radius, end) +
                        polar2xy(opt, end-(np.pi*((val*i)/curve_steps)))
                        for i in range(1,curve_steps+1)]
    verts_upper = verts_upper_start+verts_upper_curve+[polar2xy(radius, end)]

    verts_inner_start = [polar2xy(inner, end)]
    verts_inner_curve = [polar2xy(inner, end) +
                        polar2xy(opt*(1-width),
                                              end-(np.pi*((val*i)/curve_steps)))
                        for i in range(1,curve_steps+1)]
    verts_inner_curve += [polar2xy(inner, start) +
                          polar2xy(opt * (1 - width),
                                      start+(np.pi * ((val * i) / curve_steps)))
                         for i in range(1, curve_steps + 1)]
    verts_inner = verts_inner_start+verts_inner_curve+\
                  [polar2xy(inner, start), polar2xy(radius, start)]

    verts = verts_upper+verts_inner

    codes = [Path.MOVETO]+ \
            [Path.CURVE4]*curve_steps*2 +\
             [Path.CURVE4,
             Path.LINETO] +\
             [Path.CURVE4]*curve_steps*2 +\
             [Path.CURVE4,
             Path.CLOSEPOLY,
             ]

    if ax == None:
        return verts, codes
    else:
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=color+(0.5,),
                                  edgecolor=color+(0.4,), lw=LW)
        ax.add_patch(patch)


def ChordArc(start1=0, end1=60, start2=180, end2=240, radius=1.0,
             chordwidth=0.7, ax=None, color=(1,0,0)):
    # start, end should be in [0, 360)
    if start1 > end1:
        start1, end1 = end1, start1
    if start2 > end2:
        start2, end2 = end2, start2
    start1 *= np.pi/180.
    end1 *= np.pi/180.
    start2 *= np.pi/180.
    end2 *= np.pi/180.
    opt1 = 4./3. * np.tan((end1-start1)/ 4.) * radius
    opt2 = 4./3. * np.tan((end2-start2)/ 4.) * radius
    rchord = radius * (1-chordwidth)
    verts = [
        polar2xy(radius, start1),
        polar2xy(radius, start1) + polar2xy(opt1, start1+0.5*np.pi),
        polar2xy(radius, end1) + polar2xy(opt1, end1-0.5*np.pi),
        polar2xy(radius, end1),
        polar2xy(rchord, end1),
        polar2xy(rchord, start2),
        polar2xy(radius, start2),
        polar2xy(radius, start2) + polar2xy(opt2, start2+0.5*np.pi),
        polar2xy(radius, end2) + polar2xy(opt2, end2-0.5*np.pi),
        polar2xy(radius, end2),
        polar2xy(rchord, end2),
        polar2xy(rchord, start1),
        polar2xy(radius, start1),
        ]

    codes = [Path.MOVETO,
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
        patch = patches.PathPatch(path, facecolor=color+(0.5,),
                                  edgecolor=color+(0.4,), lw=LW)
        ax.add_patch(patch)

def selfChordArc(start=0, end=60, radius=1.0, chordwidth=0.7,
                 ax=None, color=(1,0,0)):
    # start, end should be in [0, 360)
    if start > end:
        start, end = end, start
    start *= np.pi/180.
    end *= np.pi/180.
    opt = 4./3. * np.tan((end-start)/ 4.) * radius
    rchord = radius * (1-chordwidth)
    verts = [
        polar2xy(radius, start),
        polar2xy(radius, start) + polar2xy(opt, start+0.5*np.pi),
        polar2xy(radius, end) + polar2xy(opt, end-0.5*np.pi),
        polar2xy(radius, end),
        polar2xy(rchord, end),
        polar2xy(rchord, start),
        polar2xy(radius, start),
        ]

    codes = [Path.MOVETO,
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
        patch = patches.PathPatch(path,
                          facecolor=color+(0.5,), edgecolor=color+(0.4,), lw=LW)
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
    x = X.sum(axis = 1) # sum over rows
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    diam = 1.8

    if colors is None:
    # use d3.js category10 https://github.com/d3/d3-3.x-api-reference/blob/master/Ordinal-Scales.md#category10
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        if len(x) > 10:
            print('x is too large! Use x smaller than 10')
    if type(colors[0])==str:
        colors = [hex2rgb(colors[i]) for i in range(len(x))]

    # find position for each start and end
    y = x/np.sum(x).astype(float) * (360 - pad*len(x))

    pos = {}
    arc = []
    nodePos = []
    start = 0
    for i in range(len(x)):
        end = start + y[i]
        arc.append((start, end))
        angle = 0.5*(start+end)
        #print(start, end, angle)
        if -30 <= angle <= 210:
            angle -= 90
        else:
            angle -= 270
        nodePos.append(tuple(polar2xy((diam/2)+diam*.05,
                                      0.5*(start+end)*np.pi/180.)) + (angle,))
        z = (X[i, :]/x[i].astype(float)) * (end - start)
        ids = np.argsort(z)
        z0 = start
        for j in ids:
            pos[(i, j)] = (z0, z0+z[j])
            z0 += z[j]
        start = end + pad

    for i in range(len(x)):
        start, end = arc[i]
        # This draws the outter ring #
        #IdeogramArc(start=start, end=end, radius=1.0, ax=ax,
        #            color=colors[i], width=width)
        a = Arc((0, 0), diam, diam, 0, start, end, color=colors[i], lw=10)
        ax.add_patch(a)
        start, end = pos[(i,i)]
        # This draws the paths to itself #
        if end-start < 180: # Indicates this method will work fine !
            selfChordArc(start, end, radius=1.-width, color=colors[i],
                        chordwidth=chordwidth*0.7, ax=ax)
        else: # Need to use a wedge because the arch distorts past 180-degrees
            path = Wedge(0, diam/2, start, end, color=colors[i]+(.5,))
            ax.add_patch(path)
        for j in range(i):
            color = colors[i]
            if X[i, j] > X[j, i]: # Color by the dominant signal #
                color = colors[j]
            start1, end1 = pos[(i,j)]
            start2, end2 = pos[(j,i)]
            ChordArc(start1, end1, start2, end2,
                radius=1.-width, color=color, chordwidth=chordwidth, ax=ax)

    return nodePos





