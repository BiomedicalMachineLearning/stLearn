import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from stlearn.utils import _check_cluster


def local_plot(
    adata: AnnData,
    use_cluster: str,
    use_label: str = "leiden",
    reverse: bool = False,
    data_alpha: float = 1.0,
    arrow_alpha: float = 1.0,
    branch_alpha: float = 0.2,
    spot_size: float | int = 1,
    show_color_bar: bool = True,
    show_axis: bool = False,
    show_plot: bool = True,
    name: str | None = None,
    dpi: int = 150,
    output: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """\
    Local spatial trajectory inference plot. Run
    ``st.spatial.trajectory.pseudotimespace_local`` with ``cluster=use_cluster`` first.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_cluster
        Label of the cluster in ``adata.obs[use_label]`` to plot, e.g. ``"6"``.
    use_label
        Use label result of cluster method.
    reverse
        Reverse the direction of the arrows.
    data_alpha
        Opacity of the spot.
    arrow_alpha
        Opacity of the arrow.
    branch_alpha
        Opacity of the branch edge.
    spot_size
        Size of the spot.
    show_color_bar
        Show color bar or not.
    show_axis
        Show axis or not.
    show_plot
        Show the plot or not.
    name
        File name used when ``output`` is given.
    dpi
        Set dpi as the resolution for the plot.
    output
        Directory to save the plot to.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Nothing
    """

    use_cluster = _check_cluster(adata, use_label, use_cluster)
    tmp = adata.obs[adata.obs[use_label] == use_cluster]
    ref_cluster = adata[list(tmp.index)]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")

    # Subclusters in the same order local_level used for ST_distance_matrix, so
    # row i of the matrix is centroids_[i].
    centroid_dict = adata.uns["centroid_dict"]
    centroids_ = []
    for i in ref_cluster.obs["sub_cluster_labels"].unique():
        if (
            len(adata.obs[adata.obs["sub_cluster_labels"] == i])
            > adata.uns["threshold_spots"]
        ):
            centroids_.append(centroid_dict[i])

    stdm = adata.uns["ST_distance_matrix"]
    non_abs_dpt = adata.uns["nonabs_dpt_distance_matrix"]
    if stdm.shape[0] != len(centroids_):
        raise ValueError(
            "adata.uns['ST_distance_matrix'] was computed for a different cluster. "
            "Run st.spatial.trajectory.pseudotimespace_local with "
            f"cluster={use_cluster!r} first."
        )

    for i in range(0, len(centroids_)):
        if i == len(centroids_) - 1:
            break
        j = 0
        while j <= (len(centroids_) - 2 - i):
            j = j + 1

            m = stdm[i, i + j]
            dpt_distance = non_abs_dpt[i, i + j]
            y = calculate_y(np.abs(m))

            x = np.linspace(centroids_[i][0], centroids_[i + j][0], 1000)
            z = np.linspace(centroids_[i][1], centroids_[i + j][1], 1000)

            # The branch: an arc from one subcluster centroid to the other, rising
            # out of the tissue plane by the PTS distance m.
            ax.plot(x, y, z, zorder=10, c="#333333", linewidth=1, alpha=branch_alpha)

            if reverse:
                dpt_distance = -dpt_distance
            if dpt_distance <= 0:
                xyz = ([x[500], x[520]], [y[500], y[520]], [z[500], z[520]])
            else:
                xyz = ([x[520], x[500]], [y[520], y[500]], [z[520], z[500]])

            arrow = Arrow3D(
                xyz[0],
                xyz[1],
                xyz[2],
                zorder=10,
                mutation_scale=7,
                lw=1,
                arrowstyle="->",
                color="r",
                alpha=arrow_alpha,
            )
            ax.add_artist(arrow)

            ax.text(
                x[500],
                y[500] - 0.15,
                z[500],
                np.round(np.abs(m), 3),
                color="black",
                size=5,
                zorder=100,
            )

    sc = ax.scatter(
        ref_cluster.obs["imagecol"],
        0,
        ref_cluster.obs["imagerow"],
        c=ref_cluster.obs["dpt_pseudotime"],
        s=spot_size,
        cmap="viridis",
        zorder=0,
        alpha=data_alpha,
    )

    _ = ax.scatter(
        adata.obs[adata.obs[use_label] != use_cluster]["imagecol"],
        0,
        adata.obs[adata.obs[use_label] != use_cluster]["imagerow"],
        c="grey",
        s=spot_size,
        zorder=0,
        alpha=0.1,
    )

    if show_color_bar:
        cb = fig.colorbar(sc, cax=fig.add_axes([0.1, 0.3, 0.03, 0.5]))
        cb.outline.set_visible(False)

    ax.set_ylim([-1, 0])
    ax.set_xlim([min(adata.obs["imagecol"]) - 10, max(adata.obs["imagecol"]) + 10])
    ax.set_zlim([min(adata.obs["imagerow"]) - 10, max(adata.obs["imagerow"]) + 10])

    if not show_axis:
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.get_zaxis().set_ticks([])

    ax.invert_zaxis()
    ax.patch.set_visible(False)
    if show_plot:
        plt.show()

    # plt.rcParams['figure.figsize'] = 6, 4

    if output is not None:
        if name is None:
            print("The file name is not defined!")
            name = use_label
        fig.savefig(output + "/" + name, dpi=dpi, bbox_inches="tight", pad_inches=0)

    return adata


def calculate_y(m):
    import math

    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    y = np.linspace(mu - 0.5 * sigma, mu + 0.5 * sigma, 1000)
    y = np.cos(np.absolute(y))
    y = -(m) * (y - np.min(y)) / np.ptp(y)

    return y


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)
