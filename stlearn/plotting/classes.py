"""
Title: SpatialBasePlot for all spatial coordinates and image plot
Author: Duy Pham
Date: 20 Feb 2021
"""

import numbers
import warnings
from typing import (  # Special
    Optional,  # Classes
)

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.interpolate import griddata

from ..classes import Spatial
from ..utils import Axes, _AxesSubplot, _read_graph
from .utils import centroidpython, check_sublist, get_cluster, get_cmap, get_node


class SpatialBasePlot(Spatial):
    def __init__(
        self,
        # plotting param
        adata: AnnData,
        title: Optional["str"] = None,
        figsize: tuple[float, float] | None = None,
        cmap: str | None = "Spectral_r",
        use_label: str | None = None,
        list_clusters: list | None = None,
        ax: matplotlib.axes.Axes | None = None,
        fig: matplotlib.figure.Figure | None = None,
        show_plot: bool | None = True,
        show_axis: bool | None = False,
        show_image: bool | None = True,
        show_color_bar: bool | None = True,
        color_bar_label: str | None = "",
        zoom_coord: float | None = None,
        crop: bool | None = True,
        margin: float | None = 100,
        size: float | None = 7,
        image_alpha: float | None = 1.0,
        cell_alpha: float | None = 0.7,
        use_raw: bool | None = False,
        fname: str | None = None,
        dpi: int | None = 120,
        **kwds,
    ):
        super().__init__(
            adata,
        )
        self.title = title
        self.figsize = figsize
        self.image_alpha = image_alpha
        self.cell_alpha = cell_alpha
        self.size = size
        self.query_adata = self.adata[0].copy()
        self.list_clusters = list_clusters
        self.fname = fname
        self.dpi = dpi

        if use_raw:
            self.query_adata = self.adata[0].raw.to_adata().copy()

        if self.list_clusters is not None:
            assert use_label is not None, "Please specify `use_label` parameter!"

        if use_label is not None:

            assert (
                use_label in self.adata[0].obs.columns
            ), "Please choose the right label in `adata.obs.columns`!"
            self.use_label = use_label

            if self.list_clusters is None:

                self.list_clusters = np.array(
                    self.adata[0].obs[use_label].cat.categories
                )
            else:
                if self.list_clusters is not list:
                    self.list_clusters = [self.list_clusters]

                clusters_indexes = [
                    np.where(adata.obs[use_label].cat.categories == i)[0][0]
                    for i in self.list_clusters
                ]
                self.list_clusters = np.array(self.list_clusters)[
                    np.argsort(clusters_indexes)
                ]

            self.query_indexes = self._get_query_clusters_index()

            self._select_clusters()

        # Initialize cmap
        scanpy_cmap = ["vega_10_scanpy", "vega_20_scanpy", "default_102", "default_28"]
        stlearn_cmap = ["jana_40", "default"]
        cmap_available = plt.colormaps() + scanpy_cmap + stlearn_cmap
        error_msg = (
            "cmap must be a matplotlib.colors.LinearSegmentedColormap OR"
            "one of these: " + str(cmap_available)
        )
        if cmap is str:
            assert cmap in cmap_available, error_msg
        elif cmap is not matplotlib.colors.LinearSegmentedColormap:
            raise Exception(error_msg)
        self.cmap = cmap

        if fig is None and ax is None:
            self.fig, self.ax = self._generate_frame()
        else:
            self.fig, self.ax = fig, ax

        if not show_axis:
            self._remove_axis(self.ax)

        if show_image:
            self._add_image(self.ax)

        if zoom_coord is not None:
            self._zoom_image(self.ax, zoom_coord)
            crop = False
        if crop:
            self._crop_image(self.ax, margin)

    def _select_clusters(self):
        def create_query(list_cl, use_label):
            ini = ""
            for sub in list_cl:
                ini = ini + self.use_label + ' == "' + str(sub) + '" | '
            return ini[:-2]

        if self.list_clusters is not None:
            # IF not all clusters specified, subset, otherwise just copy.
            if len(self.list_clusters) != len(
                self.adata[0].obs[self.use_label].cat.categories
            ):
                self.query_adata = self.query_adata[
                    self.query_adata.obs.query(
                        create_query(self.list_clusters, self.use_label)
                    ).index
                ].copy()
            else:
                self.query_adata = self.query_adata.copy()
        else:
            self.query_adata = self.query_adata.copy()

    def _generate_frame(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.figsize)
        return [fig, ax]

    def _add_image(self, main_ax: Axes):
        image = self.query_adata.uns["spatial"][self.library_id]["images"][self.img_key]
        main_ax.imshow(
            image,
            alpha=self.image_alpha,
            zorder=-1,
        )

    def _plot_colorbar(self, plot_ax: Axes, color_bar_label: str = ""):

        cb = plt.colorbar(
            plot_ax, aspect=10, shrink=0.5, cmap=self.cmap, label=color_bar_label
        )
        cb.outline.set_visible(False)

    def _remove_axis(self, main_ax: Axes):
        main_ax.axis("off")

    def _crop_image(self, main_ax: _AxesSubplot, margin: float):

        main_ax.set_xlim(self.imagecol.min() - margin, self.imagecol.max() + margin)

        main_ax.set_ylim(self.imagerow.min() - margin, self.imagerow.max() + margin)

        main_ax.set_ylim(main_ax.get_ylim()[::-1])

    def _zoom_image(self, main_ax: _AxesSubplot, zoom_coord: float | None):

        main_ax.set_xlim(zoom_coord[0], zoom_coord[1])
        main_ax.set_ylim(zoom_coord[3], zoom_coord[2])

    def _add_color_bar(self, plot, color_bar_label: str = ""):
        cb = plt.colorbar(
            plot,
            aspect=10,
            shrink=0.5,
            cmap=self.cmap,
            label=color_bar_label,
        )
        cb.outline.set_visible(False)

    def _add_title(self):
        plt.title(self.title)

    def _get_query_clusters_index(self):
        index_query = []
        full_labels = self.adata[0].obs[self.use_label].cat.categories

        for query in self.list_clusters:
            index_query.append(np.where(np.array(full_labels) == query)[0][0])

        return index_query

    def _save_output(self):

        self.fig.savefig(
            fname=self.fname, bbox_inches="tight", pad_inches=0, dpi=self.dpi
        )


################################################################
#                                                              #
#                      Gene plot class                         #
#                                                              #
################################################################


class GenePlot(SpatialBasePlot):
    def __init__(
        self,
        adata: AnnData,
        # plotting param
        title: Optional["str"] = None,
        figsize: tuple[float, float] | None = None,
        cmap: str | None = "Spectral_r",
        use_label: str | None = None,
        list_clusters: list | None = None,
        ax: matplotlib.axes.Axes | None = None,
        fig: matplotlib.figure.Figure | None = None,
        show_plot: bool | None = True,
        show_axis: bool | None = False,
        show_image: bool | None = True,
        show_color_bar: bool | None = True,
        color_bar_label: str | None = "",
        crop: bool | None = True,
        zoom_coord: float | None = None,
        margin: float | None = 100,
        size: float | None = 7,
        image_alpha: float | None = 1.0,
        cell_alpha: float | None = 1.0,
        use_raw: bool | None = False,
        fname: str | None = None,
        dpi: int | None = 120,
        # gene plot param
        gene_symbols: str | list | None = None,
        threshold: float | None = None,
        method: str = "CumSum",
        contour: bool = False,
        step_size: int | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs,
    ):
        super().__init__(
            adata=adata,
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
        )

        method_available = ["CumSum", "NaiveMean"]
        assert method in method_available, "Please choose available method in: " + str(
            method_available
        )
        self.method = method

        self.step_size = step_size

        if self.title is None:
            if gene_symbols is str:
                self.title = str(gene_symbols)
                gene_symbols = [gene_symbols]
            else:
                self.title = ", ".join(gene_symbols)

        self._add_title()

        self.gene_symbols = gene_symbols

        gene_values = self._get_gene_expression()

        self.available_ids = self._add_threshold(gene_values, threshold)

        self.vmin, self.vmax = vmin, vmax

        if contour:
            plot = self._plot_contour(gene_values[self.available_ids])
        else:
            plot = self._plot_genes(gene_values[self.available_ids])

        if show_color_bar:
            self._add_color_bar(plot, color_bar_label=color_bar_label)

        if fname is not None:
            self._save_output()

    def _get_gene_expression(self):

        # Gene plot option
        if len(self.gene_symbols) == 0:
            raise ValueError("Genes should be provided, please input genes")

        elif len(self.gene_symbols) == 1:

            if self.gene_symbols[0] not in self.query_adata.var_names:
                raise ValueError(
                    self.gene_symbols[0]
                    + " is not exist in the data, please try another gene"
                )

            colors = self.query_adata[:, self.gene_symbols].to_df().iloc[:, -1]

            return colors
        else:

            for gene in self.gene_symbols:
                if gene not in self.query_adata.var.index:
                    self.gene_symbols.remove(gene)
                    warnings.warn(
                        "We removed " + gene + " because they not exist in the data"
                    )
                if len(self.gene_symbols) == 0:
                    raise ValueError("All provided genes are not exist in the data")

            count_gene = self.query_adata[:, self.gene_symbols].to_df()

            if self.method is None:
                raise ValueError(
                    "Please provide method to combine genes by NaiveMean/CumSum"
                )

            if self.method == "NaiveMean":
                present_genes = (count_gene > 0).sum(axis=1) / len(self.gene_symbols)

                count_gene = (count_gene.mean(axis=1)) * present_genes

            elif self.method == "CumSum":
                count_gene = count_gene.cumsum(axis=1).iloc[:, -1]

            colors = count_gene

            return colors

    def _plot_genes(self, gene_values: pd.Series):

        if self.vmin is None and self.vmax is None:
            vmin = min(gene_values)
            vmax = max(gene_values)
        else:
            vmin, vmax = self.vmin, self.vmax

        # Plot scatter plot based on pixel of spots
        imgcol_new = self.query_adata.obsm["spatial"][:, 0] * self.scale_factor
        imgrow_new = self.query_adata.obsm["spatial"][:, 1] * self.scale_factor
        plot = self.ax.scatter(
            imgcol_new,
            imgrow_new,
            edgecolor="none",
            alpha=self.cell_alpha,
            s=self.size,
            marker="o",
            vmin=vmin,
            vmax=vmax,
            cmap=plt.get_cmap(self.cmap) if self.cmap is str else self.cmap,
            c=gene_values,
        )
        return plot

    def _plot_contour(self, gene_values: pd.Series):

        imgcol_new = self.query_adata.obsm["spatial"][:, 0] * self.scale_factor
        imgrow_new = self.query_adata.obsm["spatial"][:, 1] * self.scale_factor
        # Extracting x,y and values (z)
        z = gene_values
        y = imgrow_new
        x = imgcol_new

        # Interpolating values to get better coverage
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method="linear")

        if self.step_size is None:
            self.step_size = int(np.max(z) / 50)
            if self.step_size < 1:
                self.step_size = 1
        # Creating contour plot with a step size of 1

        cs = plt.contourf(
            xi,
            yi,
            zi,
            range(0, int(np.nanmax(zi)) + self.step_size, self.step_size),
            cmap=plt.get_cmap(self.cmap) if self.cmap is str else self.cmap,
            alpha=self.cell_alpha,
        )
        return cs

    def _add_threshold(self, gene_values, threshold):
        if threshold is None:
            return np.repeat(True, len(gene_values))
        else:
            return gene_values > threshold


################################################################
#                                                              #
#                      Feature plot class                      #
#                                                              #
################################################################


class FeaturePlot(SpatialBasePlot):
    def __init__(
        self,
        adata: AnnData,
        # plotting param
        title: Optional["str"] = None,
        figsize: tuple[float, float] | None = None,
        cmap: str | None = "Spectral_r",
        use_label: str | None = None,
        list_clusters: list | None = None,
        ax: matplotlib.axes.Axes | None = None,
        fig: matplotlib.figure.Figure | None = None,
        show_plot: bool | None = True,
        show_axis: bool | None = False,
        show_image: bool | None = True,
        show_color_bar: bool | None = True,
        color_bar_label: str | None = "",
        crop: bool | None = True,
        zoom_coord: float | None = None,
        margin: float | None = 100,
        size: float | None = 7,
        image_alpha: float | None = 1.0,
        cell_alpha: float | None = 1.0,
        use_raw: bool | None = False,
        fname: str | None = None,
        dpi: int | None = 120,
        # gene plot param
        feature: str = None,
        threshold: float | None = None,
        contour: bool = False,
        step_size: int | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs,
    ):
        super().__init__(
            adata=adata,
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
        )

        self.step_size = step_size

        self.title = feature
        self._add_title()

        self.feature = feature

        feature_values = self._get_feature_values()

        self.available_ids = self._add_threshold(feature_values, threshold)

        self.vmin, self.vmax = vmin, vmax

        if contour:
            plot = self._plot_contour(feature_values[self.available_ids])
        else:
            plot = self._plot_feature(feature_values[self.available_ids])

        if show_color_bar:
            self._add_color_bar(plot, color_bar_label=color_bar_label)

        if fname is not None:
            self._save_output()

    def _get_feature_values(self):

        if self.feature not in self.query_adata.obs:
            raise ValueError(
                self.feature + " is not in data.obs, please try another feature"
            )
        elif not isinstance(
            self.query_adata.obs[self.feature].values[0], numbers.Number
        ):
            raise ValueError(
                self.feature
                + " in data.obs is not continuous, please try another feature"
            )

        colors = self.query_adata.obs[self.feature]

        return colors

    def _plot_feature(self, feature_values: pd.Series):

        if self.vmin is None and self.vmax is None:
            vmin = min(feature_values)
            vmax = max(feature_values)
        else:
            vmin, vmax = self.vmin, self.vmax

        # Plot scatter plot based on pixel of spots
        imgcol_new = self.query_adata.obsm["spatial"][:, 0] * self.scale_factor
        imgrow_new = self.query_adata.obsm["spatial"][:, 1] * self.scale_factor
        plot = self.ax.scatter(
            imgcol_new,
            imgrow_new,
            edgecolor="none",
            alpha=self.cell_alpha,
            s=self.size,
            marker="o",
            vmin=vmin,
            vmax=vmax,
            cmap=plt.get_cmap(self.cmap) if self.cmap is str else self.cmap,
            c=feature_values,
        )
        return plot

    def _plot_contour(self, feature_values: pd.Series):

        imgcol_new = self.query_adata.obsm["spatial"][:, 0] * self.scale_factor
        imgrow_new = self.query_adata.obsm["spatial"][:, 1] * self.scale_factor
        # Extracting x,y and values (z)
        z = feature_values
        y = imgrow_new
        x = imgcol_new

        # Interpolating values to get better coverage
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method="linear")

        if self.step_size is None:
            self.step_size = int(np.max(z) / 50)
            if self.step_size < 1:
                self.step_size = 1
        # Creating contour plot with a step size of 1

        cs = plt.contourf(
            xi,
            yi,
            zi,
            range(0, int(np.nanmax(zi)) + self.step_size, self.step_size),
            cmap=plt.get_cmap(self.cmap) if self.cmap is str else self.cmap,
            alpha=self.cell_alpha,
        )
        return cs

    def _add_threshold(self, feature_values, threshold):
        if threshold is None:
            return np.repeat(True, len(feature_values))
        else:
            return feature_values > threshold


# Cluster plot class
class ClusterPlot(SpatialBasePlot):
    def __init__(
        self,
        adata: AnnData,
        # plotting param
        title: Optional["str"] = None,
        figsize: tuple[float, float] | None = None,
        cmap: str | None = "default",
        use_label: str | None = None,
        list_clusters: list | None = None,
        ax: matplotlib.axes.Axes | None = None,
        fig: matplotlib.figure.Figure | None = None,
        show_plot: bool | None = True,
        show_axis: bool | None = False,
        show_image: bool | None = True,
        show_color_bar: bool | None = True,
        crop: bool | None = True,
        zoom_coord: float | None = None,
        margin: float | None = 100,
        size: float | None = 5,
        image_alpha: float | None = 1.0,
        cell_alpha: float | None = 1.0,
        fname: str | None = None,
        dpi: int | None = 120,
        # cluster plot param
        show_subcluster: bool | None = False,
        show_cluster_labels: bool | None = False,
        show_trajectories: bool | None = False,
        reverse: bool | None = False,
        show_node: bool | None = False,
        threshold_spots: int | None = 5,
        text_box_size: float | None = 5,
        color_bar_size: float | None = 10,
        bbox_to_anchor: tuple[float, float] | None = (1, 1),
        # trajectory
        trajectory_node_size: int | None = 10,
        trajectory_alpha: float | None = 1.0,
        trajectory_width: float | None = 2.5,
        trajectory_edge_color: str | None = "#f4efd3",
        trajectory_arrowsize: int | None = 17,
    ):
        super().__init__(
            adata=adata,
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
            fname=fname,
            dpi=dpi,
        )

        self.cmap_ = self._get_cmap(self.cmap)

        self._add_cluster_colors()

        self._plot_clusters()

        self.threshold_spots = threshold_spots
        self.text_box_size = text_box_size
        self.color_bar_size = color_bar_size
        self.reverse = reverse
        self.show_node = show_node

        if show_color_bar:
            self._add_cluster_bar(bbox_to_anchor)

        if show_cluster_labels:
            self._add_cluster_labels()

        if show_subcluster:
            self._add_sub_clusters()

        if show_trajectories:
            self.trajectory_node_size = trajectory_node_size
            self.trajectory_alpha = trajectory_alpha
            self.trajectory_width = trajectory_width
            self.trajectory_edge_color = trajectory_edge_color
            self.trajectory_arrowsize = trajectory_arrowsize

            self._add_trajectories()

        if fname is not None:
            self._save_output()

    def _add_cluster_colors(self):
        if self.use_label + "_colors" not in self.adata[0].uns:
            # self.adata[0].uns[self.use_label + "_set"] = []
            self.adata[0].uns[self.use_label + "_colors"] = []

            for i, cluster in enumerate(self.adata[0].obs.groupby(self.use_label)):
                self.adata[0].uns[self.use_label + "_colors"].append(
                    matplotlib.colors.to_hex(self.cmap_(i / (self.cmap_n - 1)))
                )
                # self.adata[0].uns[self.use_label + "_set"].append( cluster[0] )

    def _plot_clusters(self):
        # Plot scatter plot based on pixel of spots

        for i, cluster in enumerate(self.query_adata.obs.groupby(self.use_label)):

            # Plot scatter plot based on pixel of spots
            subset_spatial = self.query_adata.obsm["spatial"][
                check_sublist(list(self.query_adata.obs.index), list(cluster[1].index))
            ]

            if self.use_label + "_colors" in self.adata[0].uns:
                # label_set = self.adata[0].uns[self.use_label+'_set']
                label_set = (
                    self.adata[0].obs[self.use_label].cat.categories.values.astype(str)
                )
                col_index = np.where(label_set == cluster[0])[0][0]
                color = self.adata[0].uns[self.use_label + "_colors"][col_index]
            else:
                color = self.cmap_(self.query_indexes[i] / (self.cmap_n - 1))

            imgcol_new = subset_spatial[:, 0] * self.scale_factor
            imgrow_new = subset_spatial[:, 1] * self.scale_factor
            _ = self.ax.scatter(
                imgcol_new,
                imgrow_new,
                c=[color],
                label=cluster[0],
                edgecolor="none",
                alpha=self.cell_alpha,
                s=self.size,
                marker="o",
            )

    def _get_cmap(self, cmap):
        cmap_, cmap_n = get_cmap(cmap)
        self.cmap_n = cmap_n
        return cmap_

    def _add_cluster_bar(self, bbox_to_anchor):
        lgnd = self.ax.legend(
            bbox_to_anchor=bbox_to_anchor,
            labelspacing=0.05,
            fontsize=self.color_bar_size,
            handleheight=1.0,
            edgecolor="white",
        )
        for handle in lgnd.legendHandles:
            handle.set_sizes([20.0])

    def _add_cluster_labels(self):

        for i, label in enumerate(self.list_clusters):

            label_index = list(
                self.query_adata.obs[
                    self.query_adata.obs[self.use_label] == str(label)
                ].index
            )
            subset_spatial = self.query_adata.obsm["spatial"][
                check_sublist(list(self.query_adata.obs.index), label_index)
            ]

            imgcol_new = subset_spatial[:, 0] * self.scale_factor
            imgrow_new = subset_spatial[:, 1] * self.scale_factor

            centroids = [centroidpython(imgcol_new, imgrow_new)]

            if centroids[0][0] < 1500:
                x = -100
                y = 50
            else:
                x = 100
                y = -50

            colors = self.adata[0].uns[self.use_label + "_colors"]
            index = self.query_indexes[i]
            self.ax.text(
                centroids[0][0] + x,
                centroids[0][1] + y,
                label,
                color="black",
                fontsize=self.text_box_size,
                zorder=3,
                bbox=dict(
                    facecolor=colors[index],
                    boxstyle="round",
                    alpha=1.0,
                ),
            )

    def _add_sub_clusters(self):

        if "sub_cluster_labels" not in self.query_adata.obs.columns:
            raise ValueError("Please run stlearn.spatial.cluster.localization")

        for i, label in enumerate(self.list_clusters):
            label_index = list(
                self.query_adata.obs[
                    self.query_adata.obs[self.use_label] == str(label)
                ].index
            )
            subset_spatial = self.query_adata.obsm["spatial"][
                check_sublist(list(self.query_adata.obs.index), label_index)
            ]

            imgcol_new = subset_spatial[:, 0] * self.scale_factor
            imgrow_new = subset_spatial[:, 1] * self.scale_factor

            if (
                len(
                    self.query_adata.obs[
                        self.query_adata.obs[self.use_label] == str(label)
                    ]["sub_cluster_labels"].unique()
                )
                < 2
            ):
                centroids = [centroidpython(imgcol_new, imgrow_new)]
                classes = np.array(
                    self.query_adata.obs[
                        self.query_adata.obs[self.use_label] == str(label)
                    ]["sub_cluster_labels"].unique()
                )

            else:
                from sklearn.neighbors import NearestCentroid

                clf = NearestCentroid()
                clf.fit(
                    np.column_stack((imgcol_new, imgrow_new)),
                    self.query_adata.obs[
                        self.query_adata.obs[self.use_label] == str(label)
                    ]["sub_cluster_labels"],
                )

                centroids = clf.centroids_
                classes = clf.classes_

            for j, label in enumerate(classes):
                if (
                    len(
                        self.query_adata.obs[
                            self.query_adata.obs["sub_cluster_labels"] == label
                        ]
                    )
                    > self.threshold_spots
                ):
                    if centroids[j][0] < 1500:
                        x = -100
                        y = 50
                    else:
                        x = 100
                        y = -50
                    colors = self.adata[0].uns[self.use_label + "_colors"]
                    index = self.query_indexes[i]

                    self.ax.text(
                        centroids[j][0] + x,
                        centroids[j][1] + y,
                        label,
                        color="black",
                        fontsize=5,
                        zorder=3,
                        bbox=dict(
                            facecolor=colors[index],
                            boxstyle="round",
                            alpha=1.0,
                        ),
                        clip_on=True,
                    )

    def _add_trajectories(self):
        used_colors = self.adata[0].uns[self.use_label + "_colors"]
        cmaps = matplotlib.colors.LinearSegmentedColormap.from_list("", used_colors)

        cmap = plt.get_cmap(cmaps)

        if "PTS_graph" not in self.adata[0].uns:
            raise ValueError("Please run stlearn.spatial.trajectory.pseudotimespace!")

        tmp = _read_graph(self.adata[0], "PTS_graph")

        G = tmp.copy()

        remove = [edge for edge in G.edges if 9999 in edge]
        G.remove_edges_from(remove)
        G.remove_node(9999)
        centroid_dict = self.adata[0].uns["centroid_dict"]
        centroid_dict = {int(key): centroid_dict[key] for key in centroid_dict}
        if self.reverse:
            nx.draw_networkx_edges(
                G,
                pos=centroid_dict,
                node_size=self.trajectory_node_size,
                alpha=self.trajectory_alpha,
                width=self.trajectory_width,
                edge_color=self.trajectory_edge_color,
                arrowsize=self.trajectory_arrowsize,
                arrowstyle="<|-",
                connectionstyle="arc3,rad=0.2",
            )
        else:
            nx.draw_networkx_edges(
                G,
                pos=centroid_dict,
                node_size=self.trajectory_node_size,
                alpha=self.trajectory_alpha,
                width=self.trajectory_width,
                edge_color=self.trajectory_edge_color,
                arrowsize=self.trajectory_arrowsize,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.2",
            )

        if self.show_node:
            for x, y in centroid_dict.items():

                if x in get_node(self.list_clusters, self.adata[0].uns["split_node"]):
                    self.ax.text(
                        y[0],
                        y[1],
                        get_cluster(str(x), self.adata[0].uns["split_node"]),
                        color="black",
                        fontsize=8,
                        zorder=100,
                        bbox=dict(
                            facecolor=cmap(
                                int(
                                    get_cluster(str(x), self.adata[0].uns["split_node"])
                                )
                                / (len(used_colors) - 1)
                            ),
                            boxstyle="circle",
                            alpha=1,
                        ),
                    )


################################################################
#                                                              #
#                      SubCluster plot class                   #
#                                                              #
################################################################


class SubClusterPlot(SpatialBasePlot):
    def __init__(
        self,
        adata: AnnData,
        # plotting param
        title: Optional["str"] = None,
        figsize: tuple[float, float] | None = None,
        cmap: str | None = "jet",
        use_label: str | None = None,
        list_clusters: list | None = None,
        ax: matplotlib.axes.Axes | None = None,
        fig: matplotlib.figure.Figure | None = None,
        show_plot: bool | None = True,
        show_axis: bool | None = False,
        show_image: bool | None = True,
        show_color_bar: bool | None = True,
        crop: bool | None = True,
        zoom_coord: float | None = None,
        margin: float | None = 100,
        size: float | None = 5,
        image_alpha: float | None = 1.0,
        cell_alpha: float | None = 1.0,
        fname: str | None = None,
        dpi: int | None = 120,
        # subcluster plot param
        cluster: int | None = 0,
        threshold_spots: int | None = 5,
        text_box_size: float | None = 5,
        bbox_to_anchor: tuple[float, float] | None = (1, 1),
        **kwargs,
    ):
        super().__init__(
            adata=adata,
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
            fname=fname,
            dpi=dpi,
        )

        self.text_box_size = text_box_size
        self.cluster = cluster

        subset = self._plot_subclusters(threshold_spots)

        self._add_subclusters_label(subset)

        if fname is not None:
            self._save_output()

    def _plot_subclusters(self, threshold_spots):
        subset = (
            self.adata[0]
            .obs[self.adata[0].obs[self.use_label] == str(self.cluster)]
            .copy()
        )

        meaningful_sub = []
        for i in subset["sub_cluster_labels"].unique():
            if len(subset[subset["sub_cluster_labels"] == str(i)]) > threshold_spots:
                meaningful_sub.append(i)

        subset = subset[subset["sub_cluster_labels"].isin(meaningful_sub)]

        colors = subset["sub_cluster_labels"]
        sub_anndata = self.adata[0][subset.index, :].copy()
        self.imgcol_new = sub_anndata.obsm["spatial"][:, 0] * self.scale_factor
        self.imgrow_new = sub_anndata.obsm["spatial"][:, 1] * self.scale_factor

        keys = list(np.sort(colors.unique()))
        self.vals = np.arange(len(keys))
        self.mapping = dict(zip(keys, self.vals))

        colors = colors.replace(self.mapping)

        self.ax.scatter(
            self.imgcol_new,
            self.imgrow_new,
            edgecolor="none",
            s=self.size,
            marker="o",
            cmap=plt.get_cmap(self.cmap) if self.cmap is str else self.cmap,
            c=colors,
            alpha=self.cell_alpha,
        )

        return subset

    def _add_subclusters_label(self, subset):
        if len(subset["sub_cluster_labels"].unique()) < 2:
            print("lower than 2")
            centroids = [centroidpython(subset[["imagecol", "imagerow"]].values)]
            classes = np.array([subset["sub_cluster_labels"][0]])

        else:
            from sklearn.neighbors import NearestCentroid

            clf = NearestCentroid()
            clf.fit(
                np.column_stack((self.imgcol_new, self.imgrow_new)),
                subset["sub_cluster_labels"],
            )

            centroids = clf.centroids_
            classes = clf.classes_

        norm = matplotlib.colors.Normalize(vmin=min(self.vals), vmax=max(self.vals))

        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=self.cmap)

        for i, label in enumerate(classes):
            if centroids[i][0] < 1000:
                x = -100
                y = 100
            else:
                x = 100
                y = -100

            self.ax.text(
                centroids[i][0] + x,
                centroids[i][1] + y,
                label,
                color="white",
                fontsize=self.text_box_size,
                zorder=3,
                bbox=dict(
                    facecolor=matplotlib.colors.to_hex(m.to_rgba(self.mapping[label])),
                    boxstyle="round",
                    alpha=0.5,
                ),
            )


################################################################
#                                                              #
#                      Cci Plot class                          #
#                                                              #
################################################################


class CciPlot(GenePlot):
    def __init__(
        self,
        adata: AnnData,
        # plotting param
        title: Optional["str"] = None,
        figsize: tuple[float, float] | None = None,
        cmap: str | None = "Spectral_r",
        use_label: str | None = None,
        list_clusters: list | None = None,
        ax: matplotlib.axes.Axes | None = None,
        fig: matplotlib.figure.Figure | None = None,
        show_plot: bool | None = True,
        show_axis: bool | None = False,
        show_image: bool | None = True,
        show_color_bar: bool | None = True,
        crop: bool | None = True,
        zoom_coord: float | None = None,
        margin: float | None = 100,
        size: float | None = 7,
        image_alpha: float | None = 1.0,
        cell_alpha: float | None = 1.0,
        use_raw: bool | None = False,
        fname: str | None = None,
        dpi: int | None = 120,
        # cci_rank param
        use_het: str | None = "het",
        contour: bool = False,
        step_size: int | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs,
    ):
        super().__init__(
            adata=adata,
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
            gene_symbols=use_het,
            contour=contour,
            step_size=step_size,
            vmin=vmin,
            vmax=vmax,
        )

        self.title = title

        self._add_title()

    def _get_gene_expression(self):
        return self.query_adata.obsm[self.gene_symbols[0]]


class LrResultPlot(GenePlot):
    def __init__(
        self,
        adata: AnnData,
        use_lr: Optional["str"] = None,
        use_result: Optional["str"] = "lr_sig_scores",
        # plotting param
        title: Optional["str"] = None,
        figsize: tuple[float, float] | None = None,
        cmap: str | None = "Spectral_r",
        list_clusters: list | None = None,
        ax: matplotlib.axes.Axes | None = None,
        fig: matplotlib.figure.Figure | None = None,
        show_plot: bool | None = True,
        show_axis: bool | None = False,
        show_image: bool | None = True,
        show_color_bar: bool | None = True,
        crop: bool | None = True,
        zoom_coord: float | None = None,
        margin: float | None = 100,
        size: float | None = 7,
        image_alpha: float | None = 1.0,
        cell_alpha: float | None = 1.0,
        use_raw: bool | None = False,
        fname: str | None = None,
        dpi: int | None = 120,
        # cci_rank param
        contour: bool = False,
        step_size: int | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs,
    ):
        # Making sure cci_rank has been run first #
        if "lr_summary" not in adata.uns:
            raise Exception(
                "To visualise LR interaction results, must run st.pl.cci.run first."
            )

        # By default, using the LR with most significant spots #
        if use_lr is None:
            use_lr = adata.uns["lr_summary"].index.values[0]
        elif use_lr not in adata.uns["lr_summary"].index:
            raise Exception(
                f"use_lr must be one of:\n" f'{adata.uns["lr_summary"].index}'
            )
        else:
            use_lr = str(use_lr)

        # Checking is a valid result #
        res_info = ["lr_scores", "p_vals", "p_adjs", "-log10(p_adjs)", "lr_sig_scores"]
        if use_result not in res_info:
            raise Exception(f"use_result must be one of:\n{res_info}")
        else:
            self.use_result = use_result

        super().__init__(
            adata=adata,
            title=title,
            figsize=figsize,
            cmap=cmap,
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
            gene_symbols=use_lr,
            contour=contour,
            step_size=step_size,
            vmin=vmin,
            vmax=vmax,
        )

    def _get_gene_expression(self):
        use_lr = self.gene_symbols[0]
        index = np.where(self.query_adata.uns["lr_summary"].index.values == use_lr)[0][
            0
        ]
        return self.query_adata.obsm[self.use_result][:, index]
