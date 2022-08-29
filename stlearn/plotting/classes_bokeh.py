from __future__ import division
import numpy as np
import pandas as pd
from PIL import Image
from stlearn.tools.microenv.cci.het import get_edges

from bokeh.plotting import (
    figure,
    show,
    ColumnDataSource,
    curdoc,
)
from bokeh.models import (
    BoxSelectTool,
    LassoSelectTool,
    CustomJS,
    Div,
    Paragraph,
    LinearColorMapper,
    Slider,
    Panel,
    Tabs,
    Select,
    AutocompleteInput,
    ColorBar,
    TextInput,
    BasicTicker,
    CrosshairTool,
    HoverTool,
    ZoomOutTool,
    CheckboxGroup,
    Arrow,
    VeeHead,
    Button,
    Dropdown,
    Div,
)

from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from anndata import AnnData
from bokeh.palettes import (
    Spectral11,
    Viridis256,
    Reds256,
    Blues256,
    Magma256,
    Category20,
)
from bokeh.layouts import column, row, grid
from collections import OrderedDict
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from stlearn.classes import Spatial
from typing import Optional
from stlearn.utils import _read_graph
import scanpy as sc


class BokehGenePlot(Spatial):
    def __init__(
        self,
        # plotting param
        adata: AnnData,
    ):
        super().__init__(
            adata,
        )
        # Open image, and make sure it's RGB*A*
        image = (self.img * 255).astype(np.uint8)

        img_pillow = Image.fromarray(image).convert("RGBA")

        self.xdim, self.ydim = img_pillow.size

        # Create an array representation for the image `img`, and an 8-bit "4
        # layer/RGBA" version of it `view`.
        self.image = np.empty((self.ydim, self.xdim), dtype=np.uint32)
        view = self.image.view(dtype=np.uint8).reshape((self.ydim, self.xdim, 4))
        # Copy the RGBA image into view, flipping it so it comes right-side up
        # with a lower-left origin
        view[:, :, :] = np.flipud(np.asarray(img_pillow))

        # Display the 32-bit RGBA image
        self.dim = max(self.xdim, self.ydim)
        gene_list = list(adata.var_names)

        self.data_alpha = Slider(
            title="Spot alpha", value=1.0, start=0, end=1.0, step=0.1
        )
        self.tissue_alpha = Slider(
            title="Tissue alpha", value=1.0, start=0, end=1.0, step=0.1
        )
        self.spot_size = Slider(
            title="Spot size", value=5.0, start=0, end=5.0, step=1.0
        )

        self.gene_select = AutocompleteInput(
            title="Gene:", value=gene_list[0], completions=gene_list, min_characters=1
        )

        color_list = ["Spectral", "Viridis", "Reds", "Blues", "Magma"]
        self.cmap_select = Select(
            title="Select color map:", value=color_list[0], options=color_list
        )

        self.output_backend = Select(
            title="Select output backend:", value="webgl", options=["webgl", "svg"]
        )

        self.menu = []

        for col in adata.obs.columns:
            if adata.obs[col].dtype.name == "category":
                if col != "sub_cluster_labels":
                    self.menu.append(col)
        if len(self.menu) != 0:
            self.use_label = Select(
                title="Select use_label:", value=self.menu[0], options=self.menu
            )
            inputs = column(
                self.gene_select,
                self.data_alpha,
                self.tissue_alpha,
                self.spot_size,
                self.cmap_select,
                self.use_label,
                self.output_backend,
            )
            self.layout = column(row(inputs, self.make_fig()), self.add_violin())
        else:
            inputs = column(
                self.gene_select,
                self.data_alpha,
                self.tissue_alpha,
                self.spot_size,
                self.cmap_select,
                self.output_backend,
            )
            self.layout = row(inputs, self.make_fig())

        # Make a tab with the layout
        # self.tab = Tabs(tabs = [Panel(child=self.layout, title="Gene plot")])

        def modify_fig(doc):

            doc.add_root(row(self.layout, width=800))

            self.data_alpha.on_change("value", self.update_data)
            self.tissue_alpha.on_change("value", self.update_data)
            self.spot_size.on_change("value", self.update_data)
            self.gene_select.on_change("value", self.update_data)
            self.cmap_select.on_change("value", self.update_data)
            self.output_backend.on_change("value", self.update_data)
            if len(self.menu) != 0:
                self.use_label.on_change("value", self.update_data)

        handler = FunctionHandler(modify_fig)

        self.app = Application(handler)

    def make_fig(self):

        fig = figure(
            title=self.gene_select.value,
            x_range=(0, self.dim - 150),
            y_range=(self.dim, 0),
            output_backend=self.output_backend.value,
            name="GenePlot",
            active_scroll="wheel_zoom",
        )

        colors = self._get_gene_expression([self.gene_select.value])

        s1 = ColumnDataSource(data=dict(x=self.imagecol, y=self.imagerow, color=colors))

        fig.image_rgba(
            image=[self.image],
            x=0,
            y=self.xdim,
            dw=self.ydim,
            dh=self.xdim,
            global_alpha=self.tissue_alpha.value,
        )
        if self.cmap_select.value == "Spectral":
            cmap = Spectral11
        elif self.cmap_select.value == "Viridis":
            cmap = Viridis256
        elif self.cmap_select.value == "Reds":
            cmap = list(Reds256)
            cmap.reverse()
        elif self.cmap_select.value == "Blues":
            cmap = list(Blues256)
            cmap.reverse()
        elif self.cmap_select.value == "Magma":
            cmap = Magma256

        color_mapper = LinearColorMapper(
            palette=cmap, low=min(s1.data["color"]), high=max(s1.data["color"])
        )

        fig.circle(
            "x",
            "y",
            color={"field": "color", "transform": color_mapper},
            size=self.spot_size.value,
            source=s1,
            fill_alpha=self.data_alpha.value,
            line_alpha=self.data_alpha.value,
        )

        color_bar = ColorBar(
            color_mapper=color_mapper, ticker=BasicTicker(), location=(-20, 0)
        )
        fig.add_layout(color_bar, "right")

        fig.toolbar.logo = None
        fig.xaxis.visible = False
        fig.yaxis.visible = False
        fig.xgrid.grid_line_color = None
        fig.ygrid.grid_line_color = None
        fig.outline_line_alpha = 0
        fig.add_tools(LassoSelectTool())
        fig.add_tools(BoxSelectTool())
        fig.add_tools(HoverTool())

        hover = fig.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict(
            [
                ("Spot", "$index"),
                ("X location", "@x{1.11}"),
                ("Y location", "@y{1.11}"),
                ("Gene expression", "@color{1.11}"),
            ]
        )

        return fig

    def add_violin(self):
        violin = self.create_violin(
            self.adata[0],
            gene_symbol=self.gene_select.value,
            use_label=self.use_label.value,
        )

        violin = (np.array(violin)).astype(np.uint8)

        img_pillow2 = Image.fromarray(violin).convert("RGBA")

        xdim, ydim = img_pillow2.size

        # Create an array representation for the image `img`, and an 8-bit "4
        # layer/RGBA" version of it `view`.
        image2 = np.empty((ydim, xdim), dtype=np.uint32)
        view2 = image2.view(dtype=np.uint8).reshape((ydim, xdim, 4))
        # Copy the RGBA image into view, flipping it so it comes right-side up
        # with a lower-left origin
        view2[:, :, :] = np.flipud(np.asarray(img_pillow2))

        p = figure(
            plot_width=910,
            plot_height=int(910 / xdim * ydim) + 5,
            output_backend=self.output_backend.value,
        )

        # must give a vector of images
        p.image_rgba(image=[image2], x=0, y=1, dw=xdim, dh=ydim)

        p.toolbar.logo = None
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.outline_line_alpha = 0

        return p

    def update_data(self, attrname, old, new):

        if len(self.menu) != 0:
            self.layout.children[0].children[1] = self.make_fig()
            self.layout.children[1] = self.add_violin()
        else:
            self.layout.children[1] = self.make_fig()

    def _get_gene_expression(self, gene_symbols):

        if gene_symbols[0] not in self.adata[0].var_names:
            raise ValueError(
                gene_symbols[0] + " is not exist in the data, please try another gene"
            )

        colors = self.adata[0][:, gene_symbols].to_df().iloc[:, -1]

        return colors

    def fig2img(self, fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        import io

        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches="tight", pad_inches=0, dpi=120)
        buf.seek(0)
        img = Image.open(buf)
        return img

    def create_violin(self, adata, gene_symbol, use_label):
        import matplotlib.pyplot as plt

        plt.rc("font", size=5)

        fig, ax = plt.subplots(figsize=(8, 5))
        sc.pl.violin(
            adata, keys=gene_symbol, groupby=use_label, rotation=45, ax=ax, show=False
        )

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

        plt.close(fig)

        violin = self.fig2img(fig)

        return violin


class BokehClusterPlot(Spatial):
    def __init__(
        self,
        # plotting param
        adata: AnnData,
    ):
        super().__init__(adata)

        # Open image, and make sure it's RGB*A*
        image = (self.img * 255).astype(np.uint8)

        img_pillow = Image.fromarray(image).convert("RGBA")

        self.xdim, self.ydim = img_pillow.size

        # Create an array representation for the image `img`, and an 8-bit "4
        # layer/RGBA" version of it `view`.
        self.image = np.empty((self.ydim, self.xdim), dtype=np.uint32)
        view = self.image.view(dtype=np.uint8).reshape((self.ydim, self.xdim, 4))
        # Copy the RGBA image into view, flipping it so it comes right-side up
        # with a lower-left origin
        view[:, :, :] = np.flipud(np.asarray(img_pillow))

        # Display the 32-bit RGBA image
        self.dim = max(self.xdim, self.ydim)

        menu = []

        for col in adata.obs.columns:
            if adata.obs[col].dtype.name == "category":
                if col != "sub_cluster_labels":
                    menu.append(col)

        self.use_label = Select(title="Select use_label:", value=menu[0], options=menu)

        # Initialize the color
        from stlearn.plotting.cluster_plot import cluster_plot

        if len(adata.obs[self.use_label.value].cat.categories) <= 20:
            cluster_plot(adata, use_label=self.use_label.value, show_plot=False)
        else:
            cluster_plot(
                adata,
                use_label=self.use_label.value,
                show_plot=False,
                cmap="default_102",
            )

        self.data_alpha = Slider(
            title="Spot alpha", value=1.0, start=0, end=1.0, step=0.1
        )

        self.tissue_alpha = Slider(
            title="Tissue alpha", value=1.0, start=0, end=1.0, step=0.1
        )

        self.spot_size = Slider(
            title="Spot size", value=5.0, start=0, end=5.0, step=1.0
        )

        self.checkbox_group = CheckboxGroup(
            labels=["Show spatial trajectories"], active=[]
        )

        self.p = Paragraph(text="""Choose clusters:""", width=400, height=20)

        self.list_cluster = CheckboxGroup(
            labels=list(self.adata[0].obs[self.use_label.value].cat.categories),
            active=list(
                np.array(
                    range(0, len(self.adata[0].obs[self.use_label.value].unique()))
                )
            ),
        )

        self.n_top_genes = Slider(
            title="Number of top genes", value=5, start=1, end=10, step=1
        )

        color_list = ["bwr", "RdBu_r", "viridis", "Reds", "Blues", "magma"]
        self.cmap_select = Select(
            title="Select color map:", value=color_list[0], options=color_list
        )

        plot_types = ["dotplot", "stacked_violin", "matrixplot"]
        self.plot_select = Select(
            title="Select DEA plot type:", value=plot_types[0], options=plot_types
        )

        self.output_backend = Select(
            title="Select output backend:", value="webgl", options=["webgl", "svg"]
        )

        self.min_logfoldchange = TextInput(value="3", title="Min log fold-change")

        if "PTS_graph" in adata.uns:
            if "rank_genes_groups" in self.adata[0].uns:
                self.inputs = column(
                    self.use_label,
                    self.data_alpha,
                    self.tissue_alpha,
                    self.spot_size,
                    self.p,
                    self.list_cluster,
                    self.checkbox_group,
                    self.n_top_genes,
                    self.min_logfoldchange,
                    self.cmap_select,
                    self.plot_select,
                    self.output_backend,
                )
            else:
                self.inputs = column(
                    self.use_label,
                    self.data_alpha,
                    self.tissue_alpha,
                    self.spot_size,
                    self.p,
                    self.list_cluster,
                    self.checkbox_group,
                    self.output_backend,
                )

        else:
            if "rank_genes_groups" in self.adata[0].uns:
                self.inputs = column(
                    self.use_label,
                    self.data_alpha,
                    self.tissue_alpha,
                    self.spot_size,
                    self.p,
                    self.list_cluster,
                    self.n_top_genes,
                    self.min_logfoldchange,
                    self.cmap_select,
                    self.plot_select,
                    self.output_backend,
                )
            else:
                self.inputs = column(
                    self.use_label,
                    self.data_alpha,
                    self.tissue_alpha,
                    self.spot_size,
                    self.p,
                    self.list_cluster,
                    self.output_backend,
                )

        if "rank_genes_groups" in self.adata[0].uns:
            if (
                self.use_label.value
                == self.adata[0].uns["rank_genes_groups"]["params"]["groupby"]
            ):
                self.layout = column(row(self.inputs, self.make_fig()), self.add_dea())
            else:
                self.layout = column(row(self.inputs, self.make_fig()), Div(text=""))
        else:
            self.layout = row(self.inputs, self.make_fig())

        def modify_fig(doc):
            doc.add_root(row(self.layout, width=800))
            self.use_label.on_change("value", self.update_list)
            self.use_label.on_change("value", self.update_data)

            self.data_alpha.on_change("value", self.update_data)
            self.tissue_alpha.on_change("value", self.update_data)
            self.spot_size.on_change("value", self.update_data)
            self.list_cluster.on_change("active", self.update_data)
            if "PTS_graph" in self.adata[0].uns:
                self.checkbox_group.on_change("active", self.update_data)

            self.output_backend.on_change("value", self.update_data)
            if "rank_genes_groups" in self.adata[0].uns:
                self.n_top_genes.on_change("value", self.update_data)
                self.cmap_select.on_change("value", self.update_data)
                self.plot_select.on_change("value", self.update_data)
                self.min_logfoldchange.on_change("value", self.update_data)

        handler = FunctionHandler(modify_fig)
        self.app = Application(handler)

    def update_list(self, attrname, old, name):

        # Initialize the color
        from stlearn.plotting.cluster_plot import cluster_plot

        cluster_plot(self.adata[0], use_label=self.use_label.value, show_plot=False)

        # self.list_cluster = CheckboxGroup(
        #     labels=list(self.adata[0].obs[self.use_label.value].cat.categories),
        #     active=list(
        #         np.array(range(0, len(self.adata[0].obs[self.use_label.value].unique())))
        #     ),
        # )
        self.list_cluster.labels = list(
            self.adata[0].obs[self.use_label.value].cat.categories
        )
        self.list_cluster.active = list(
            np.array(range(0, len(self.adata[0].obs[self.use_label.value].unique())))
        )

    def update_data(self, attrname, old, new):

        if "rank_genes_groups" in self.adata[0].uns:
            if (
                self.use_label.value
                == self.adata[0].uns["rank_genes_groups"]["params"]["groupby"]
            ):
                self.layout.children[0].children[1] = self.make_fig()
                self.layout.children[1] = self.add_dea()
            else:
                if len(self.layout.children) > 1:
                    self.layout.children[1] = Div(text="")
                    self.layout.children[0].children[1] = self.make_fig()
        else:
            self.layout.children[1] = self.make_fig()

    def make_fig(self):
        fig = figure(
            title="Cluster plot",
            x_range=(0, self.dim - 150),
            y_range=(self.dim, 0),
            output_backend=self.output_backend.value
            # Specifying xdim/ydim isn't quire right :-(
            # width=xdim, height=ydim,
        )

        fig.image_rgba(
            image=[self.image],
            x=0,
            y=self.xdim,
            dw=self.ydim,
            dh=self.xdim,
            global_alpha=self.tissue_alpha.value,
        )

        # Get query clusters
        command = []
        for i in self.list_cluster.active:
            command.append(
                self.use_label.value
                + ' == "'
                + self.adata[0].obs[self.use_label.value].cat.categories[i]
                + '"'
            )
        tmp = self.adata[0].obs.query(" or ".join(command))

        tmp_adata = self.adata[0][tmp.index, :]

        x = tmp_adata.obsm["spatial"][:, 0] * self.scale_factor
        y = tmp_adata.obsm["spatial"][:, 1] * self.scale_factor

        category_items = self.adata[0].obs[self.use_label.value].cat.categories
        palette = self.adata[0].uns[self.use_label.value + "_colors"]
        colormap = dict(zip(category_items, palette))
        color = list(tmp[self.use_label.value].map(colormap))
        cluster = list(tmp[self.use_label.value])

        s1 = ColumnDataSource(data=dict(x=x, y=y, color=color, cluster=cluster))
        if len(category_items[0]) > 5:
            fig.scatter(
                x="x",
                y="y",
                source=s1,
                size=self.spot_size.value,
                color="color",
                # legend_group="cluster",
                fill_alpha=self.data_alpha.value,
                line_alpha=self.data_alpha.value,
            )
        else:
            fig.scatter(
                x="x",
                y="y",
                source=s1,
                size=self.spot_size.value,
                color="color",
                legend_group="cluster",
                fill_alpha=self.data_alpha.value,
                line_alpha=self.data_alpha.value,
            )

        if 0 in self.checkbox_group.active:
            tmp = _read_graph(self.adata[0], "PTS_graph")
            G = tmp.copy()

            remove = [edge for edge in G.edges if 9999 in edge]
            G.remove_edges_from(remove)
            G.remove_node(9999)
            centroid_dict = self.adata[0].uns["centroid_dict"]
            centroid_dict = {int(key): centroid_dict[key] for key in centroid_dict}

            set_x = []
            set_y = []
            for edges in G.edges:
                fig.add_layout(
                    Arrow(
                        end=VeeHead(fill_color="#f4efd3", line_alpha=0, line_width=5),
                        x_start=centroid_dict[edges[0]][0],
                        y_start=centroid_dict[edges[0]][1],
                        x_end=centroid_dict[edges[1]][0],
                        y_end=centroid_dict[edges[1]][1],
                        line_color="#f4efd3",
                        line_width=5,
                    )
                )
                set_x.append(centroid_dict[edges[0]][0])
                set_y.append(centroid_dict[edges[0]][1])

            fig.circle(
                x=set_x,
                y=set_y,
                radius=50,
                color="#f4efd3",
                fill_alpha=0.3,
                line_alpha=0,
            )

        fig.toolbar.logo = None
        fig.xaxis.visible = False
        fig.yaxis.visible = False
        fig.xgrid.grid_line_color = None
        fig.ygrid.grid_line_color = None
        fig.outline_line_alpha = 0
        fig.add_tools(LassoSelectTool())
        fig.add_tools(ZoomOutTool())
        fig.add_tools(HoverTool())
        fig.add_tools(BoxSelectTool())

        hover = fig.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict(
            [
                ("Spot", "$index"),
                ("X location", "@x{1.11}"),
                ("Y location", "@y{1.11}"),
                ("Cluster", "@cluster"),
            ]
        )

        return fig

    def add_dea(self):
        dea = self.create_dea(self.adata[0])

        dea = (np.array(dea)).astype(np.uint8)

        img_pillow2 = Image.fromarray(dea).convert("RGBA")

        xdim, ydim = img_pillow2.size

        # Create an array representation for the image `img`, and an 8-bit "4
        # layer/RGBA" version of it `view`.
        image2 = np.empty((ydim, xdim), dtype=np.uint32)
        view2 = image2.view(dtype=np.uint8).reshape((ydim, xdim, 4))
        # Copy the RGBA image into view, flipping it so it comes right-side up
        # with a lower-left origin
        view2[:, :, :] = np.flipud(np.asarray(img_pillow2))

        p = figure(
            plot_width=910,
            plot_height=int(910 / xdim * ydim) + 5,
            output_backend=self.output_backend.value,
        )

        # must give a vector of images
        p.image_rgba(image=[image2], x=0, y=1, dw=xdim, dh=ydim)

        p.toolbar.logo = None
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        p.outline_line_alpha = 0

        return p

    def fig2img(self, fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        import io

        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches="tight", pad_inches=0, dpi=150)
        buf.seek(0)
        img = Image.open(buf)
        return img

    def create_dea(self, adata):
        import matplotlib.pyplot as plt

        plt.rc("font", size=12)

        fig, ax = plt.subplots(figsize=(12, 5))
        sc.tl.dendrogram(adata, groupby=self.use_label.value)
        if self.plot_select.value == "matrixplot":
            sc.pl.rank_genes_groups_matrixplot(
                adata,
                n_genes=self.n_top_genes.value,
                cmap=self.cmap_select.value,
                show=False,
                ax=ax,
                standard_scale="var",
                min_logfoldchange=float(self.min_logfoldchange.value),
                groups=self.adata[0]
                .obs[self.use_label.value]
                .cat.categories[self.list_cluster.active],
            )
        elif self.plot_select.value == "stacked_violin":
            sc.pl.rank_genes_groups_stacked_violin(
                adata,
                n_genes=self.n_top_genes.value,
                cmap=self.cmap_select.value,
                show=False,
                ax=ax,
                standard_scale="var",
                min_logfoldchange=float(self.min_logfoldchange.value),
                groups=self.adata[0]
                .obs[self.use_label.value]
                .cat.categories[self.list_cluster.active],
            )
        else:
            sc.pl.rank_genes_groups_dotplot(
                adata,
                n_genes=self.n_top_genes.value,
                cmap=self.cmap_select.value,
                show=False,
                ax=ax,
                standard_scale="var",
                min_logfoldchange=float(self.min_logfoldchange.value),
                groups=self.adata[0]
                .obs[self.use_label.value]
                .cat.categories[self.list_cluster.active],
            )

        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

        plt.close(fig)

        dea = self.fig2img(fig)

        return dea


class BokehLRPlot(Spatial):
    def __init__(
        self,
        # plotting param
        adata: AnnData,
    ):
        super().__init__(
            adata,
        )
        # Open image, and make sure it's RGB*A*
        image = (self.img * 255).astype(np.uint8)

        img_pillow = Image.fromarray(image).convert("RGBA")

        self.xdim, self.ydim = img_pillow.size

        # Create an array representation for the image `img`, and an 8-bit "4
        # layer/RGBA" version of it `view`.
        self.image = np.empty((self.ydim, self.xdim), dtype=np.uint32)
        view = self.image.view(dtype=np.uint8).reshape((self.ydim, self.xdim, 4))
        # Copy the RGBA image into view, flipping it so it comes right-side up
        # with a lower-left origin
        view[:, :, :] = np.flipud(np.asarray(img_pillow))

        # Display the 32-bit RGBA image
        self.dim = max(self.xdim, self.ydim)
        # available_het = [] #Checking available LR scores
        # for key in list(self.adata[0].obsm.keys()):
        #     if len(self.adata[0].obsm[key].shape) == 1:
        #         available_het.append(key)
        lrs = list(adata.uns["lr_summary"].index.values.astype(str))

        self.data_alpha = Slider(
            title="Spot alpha", value=1.0, start=0, end=1.0, step=0.1
        )
        self.tissue_alpha = Slider(
            title="Tissue alpha", value=1.0, start=0, end=1.0, step=0.1
        )
        self.spot_size = Slider(
            title="Spot size", value=5.0, start=0, end=5.0, step=1.0
        )

        # self.het_select = AutocompleteInput(
        #     title="Het:",
        #     value=available_het[0],
        #     completions=available_het,
        #     min_characters=1,
        # )
        # self.lr_select = AutocompleteInput(
        #     title="Ligand-receptor:",
        #     value=lrs[0],
        #     completions=lrs,
        #     min_characters=1,
        # )
        self.lr_select = Select(
            title="Ligand-receptor:",
            value=lrs[0],
            options=lrs,
        )

        self.output_backend = Select(
            title="Select output backend:", value="webgl", options=["webgl", "svg"]
        )

        inputs = column(
            # self.het_select,
            self.lr_select,
            self.data_alpha,
            self.tissue_alpha,
            self.spot_size,
            self.output_backend,
        )

        self.layout = row(inputs, self.make_fig())

        # Make a tab with the layout
        # self.tab = Tabs(tabs = [Panel(child=self.layout, title="Gene plot")])

        def modify_fig(doc):

            doc.add_root(row(self.layout, width=800))

            self.data_alpha.on_change("value", self.update_data)
            self.tissue_alpha.on_change("value", self.update_data)
            self.spot_size.on_change("value", self.update_data)
            # self.het_select.on_change("value", self.update_data)
            self.lr_select.on_change("value", self.update_data)

        handler = FunctionHandler(modify_fig)

        self.app = Application(handler)

    def make_fig(self):

        fig = figure(
            title=self.lr_select.value,  # self.het_select.value,
            x_range=(0, self.dim - 150),
            y_range=(self.dim, 0),
            output_backend=self.output_backend.value,
        )

        colors = self._get_lr(self.lr_select.value)
        # colors = self._get_het(self.het_select.value)

        s1 = ColumnDataSource(data=dict(x=self.imagecol, y=self.imagerow, color=colors))

        fig.image_rgba(
            image=[self.image],
            x=0,
            y=self.xdim,
            dw=self.ydim,
            dh=self.xdim,
            global_alpha=self.tissue_alpha.value,
        )

        color_mapper = LinearColorMapper(
            palette=Spectral11, low=min(s1.data["color"]), high=max(s1.data["color"])
        )

        fig.circle(
            "x",
            "y",
            color={"field": "color", "transform": color_mapper},
            size=self.spot_size.value,
            source=s1,
            fill_alpha=self.data_alpha.value,
            line_alpha=self.data_alpha.value,
        )

        color_bar = ColorBar(
            color_mapper=color_mapper, ticker=BasicTicker(), location=(10, 0)
        )
        fig.add_layout(color_bar, "right")

        fig.toolbar.logo = None
        fig.xaxis.visible = False
        fig.yaxis.visible = False
        fig.xgrid.grid_line_color = None
        fig.ygrid.grid_line_color = None
        fig.outline_line_alpha = 0
        fig.add_tools(LassoSelectTool())
        fig.add_tools(BoxSelectTool())
        fig.add_tools(HoverTool())

        hover = fig.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict(
            [
                ("Spot", "$index"),
                ("X location", "@x{1.11}"),
                ("Y location", "@y{1.11}"),
                ("Values", "@color{1.11}"),
            ]
        )

        return fig

    def update_data(self, attrname, old, new):
        self.layout.children[1] = self.make_fig()

    def _get_het(self, het):

        if het not in self.adata[0].obsm:
            raise ValueError(het + " is not exist in the data, please try another het")

        colors = self.adata[0].obsm[het]

        return colors

    def _get_lr(self, lr):
        if lr not in self.adata[0].uns["lr_summary"].index:
            raise ValueError(lr + " is not exist in the data, please try another het")

        lr_bool = self.adata[0].uns["lr_summary"].index.values.astype(str) == lr
        lr_index = np.where(lr_bool)[0][0]
        colors = self.adata[0].obsm["lr_sig_scores"][:, lr_index]

        return colors


class BokehSpatialCciPlot(Spatial):
    def __init__(
        self,
        # plotting param
        adata: AnnData,
    ):
        super().__init__(
            adata,
        )
        # Open image, and make sure it's RGB*A*
        image = (self.img * 255).astype(np.uint8)

        img_pillow = Image.fromarray(image).convert("RGBA")

        self.xdim, self.ydim = img_pillow.size

        # Create an array representation for the image `img`, and an 8-bit "4
        # layer/RGBA" version of it `view`.
        self.image = np.empty((self.ydim, self.xdim), dtype=np.uint32)
        view = self.image.view(dtype=np.uint8).reshape((self.ydim, self.xdim, 4))
        # Copy the RGBA image into view, flipping it so it comes right-side up
        # with a lower-left origin
        view[:, :, :] = np.flipud(np.asarray(img_pillow))

        # Display the 32-bit RGBA image
        self.dim = max(self.xdim, self.ydim)
        # available_het = [] #Checking available LR scores
        # for key in list(self.adata[0].obsm.keys()):
        #     if len(self.adata[0].obsm[key].shape) == 1:
        #         available_het.append(key)
        lrs = list(adata.uns["lr_summary"].index.values.astype(str))

        self.data_alpha = Slider(
            title="Spot alpha", value=1.0, start=0, end=1.0, step=0.1
        )
        self.tissue_alpha = Slider(
            title="Tissue alpha", value=1.0, start=0, end=1.0, step=0.1
        )
        self.spot_size = Slider(
            title="Spot size", value=5.0, start=0, end=10.0, step=1.0
        )
        self.arrow_size = Slider(
            title="Arrow size", value=1.0, start=0, end=10.0, step=0.5
        )

        # Getting the annnotations for which CCI has been performed.. #
        annots = [
            opt.replace("lr_cci_", "")
            for opt in adata.uns.keys()
            if opt.startswith("lr_cci_") and "raw" not in opt
        ]
        self.annot_select = Select(
            title="Cell-type annotation select:",
            value=annots[0],
            options=annots,
        )
        self.lr_select = Select(
            title="Ligand-receptor:",
            value=lrs[0],
            options=lrs,
        )

        self.list_cluster = CheckboxGroup(
            labels=list(self.adata[0].obs[self.annot_select.value].cat.categories),
            active=list(
                np.array(
                    range(0, len(self.adata[0].obs[self.annot_select.value].unique()))
                )
            ),
        )

        self.output_backend = Select(
            title="Select output backend:", value="webgl", options=["webgl", "svg"]
        )

        inputs = column(
            # self.het_select,
            self.annot_select,
            self.lr_select,
            self.data_alpha,
            self.tissue_alpha,
            self.spot_size,
            self.arrow_size,
            self.output_backend,
        )

        self.layout = row(inputs, self.make_fig())

        # Make a tab with the layout
        # self.tab = Tabs(tabs = [Panel(child=self.layout, title="Gene plot")])

        def modify_fig(doc):

            doc.add_root(row(self.layout, width=800))

            self.data_alpha.on_change("value", self.update_data)
            self.tissue_alpha.on_change("value", self.update_data)
            self.spot_size.on_change("value", self.update_data)
            self.arrow_size.on_change("value", self.update_data)
            # self.het_select.on_change("value", self.update_data)
            self.annot_select.on_change("value", self.update_data)
            self.lr_select.on_change("value", self.update_data)

        handler = FunctionHandler(modify_fig)

        self.app = Application(handler)

    def make_fig(self):

        fig = figure(
            title="Spatial CCI plot",
            x_range=(0, self.dim - 150),
            y_range=(self.dim, 0),
            output_backend=self.output_backend.value,
        )

        fig.image_rgba(
            image=[self.image],
            x=0,
            y=self.xdim,
            dw=self.ydim,
            dh=self.xdim,
            global_alpha=self.tissue_alpha.value,
        )

        # Get query clusters
        selected = self.annot_select.value.strip("raw_")
        command = []
        for i in self.list_cluster.active:
            command.append(
                selected + ' == "' + self.adata[0].obs[selected].cat.categories[i] + '"'
            )
        tmp = self.adata[0].obs.query(" or ".join(command))

        tmp_adata = self.adata[0][tmp.index, :]

        x = tmp_adata.obsm["spatial"][:, 0] * self.scale_factor
        y = tmp_adata.obsm["spatial"][:, 1] * self.scale_factor

        category_items = self.adata[0].obs[selected].cat.categories
        palette = self.adata[0].uns[selected + "_colors"]
        colormap = dict(zip(category_items, palette))
        color = list(tmp[selected].map(colormap))
        cluster = list(tmp[selected])

        s1 = ColumnDataSource(data=dict(x=x, y=y, color=color, cluster=cluster))
        if len(category_items[0]) > 5:
            fig.scatter(
                x="x",
                y="y",
                source=s1,
                size=self.spot_size.value,
                color="color",
                # legend_group="cluster",
                fill_alpha=self.data_alpha.value,
                line_alpha=self.data_alpha.value,
            )
        else:
            fig.scatter(
                x="x",
                y="y",
                source=s1,
                size=self.spot_size.value,
                color="color",
                legend_group="cluster",
                fill_alpha=self.data_alpha.value,
                line_alpha=self.data_alpha.value,
            )

        #### Adding in the arrows for the interaction edges !!! #####
        forward_edges, reverse_edges = self._get_cci_lr_edges()
        self._add_edges(
            fig, self.adata[0], forward_edges, self.arrow_size.value, forward=True
        )
        self._add_edges(
            fig, self.adata[0], reverse_edges, self.arrow_size.value, forward=False
        )

        fig.toolbar.logo = None
        fig.xaxis.visible = False
        fig.yaxis.visible = False
        fig.xgrid.grid_line_color = None
        fig.ygrid.grid_line_color = None
        fig.outline_line_alpha = 0
        fig.add_tools(LassoSelectTool())
        fig.add_tools(ZoomOutTool())
        fig.add_tools(HoverTool())
        fig.add_tools(BoxSelectTool())

        hover = fig.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict(
            [
                ("Spot", "$index"),
                ("X location", "@x{1.11}"),
                ("Y location", "@y{1.11}"),
                ("Cluster", "@cluster"),
            ]
        )

        return fig

    def update_data(self, attrname, old, new):
        self.layout.children[1] = self.make_fig()

    def _get_cci_lr_edges(self):
        """Gets edge list of significant interactions for LR pair."""

        adata = self.adata[0]
        lr = self.lr_select.value
        selected = self.annot_select.value

        # Extracting the data #
        l, r = lr.split("_")
        lr_index = np.where(adata.uns["lr_summary"].index.values == lr)[0][0]
        L_bool = adata[:, l].X.toarray()[:, 0] > 0
        R_bool = adata[:, r].X.toarray()[:, 0] > 0
        sig_bool = adata.obsm["lr_sig_scores"][:, lr_index] > 0
        int_df = adata.uns[f"per_lr_cci_{selected}"][lr]

        ###### Getting the edges, from sig-L->R and sig-R<-L ##########
        forward_edges, reverse_edges = get_edges(adata, L_bool, R_bool, sig_bool)

        ###### Subsetting to cell types with significant interactions ########
        spot_bcs = adata.obs_names.values.astype(str)
        spot_labels = adata.obs[selected].values.astype(str)
        label_set = int_df.index.values.astype(str)
        interact_bool = int_df.values > 0

        ###### Subsetting to only significant CCIs ########
        edges_sub = [[], []]  # forward, reverse
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
                if interact_bool[celltypes[k_[0]], celltypes[k_[1]]]:
                    edges_sub[i].append(edge)

        return edges_sub

    @staticmethod
    def _add_edges(fig, adata, edges, arrow_size, forward=True, scale_factor=1):
        """Gets edges for input."""
        for i, edge in enumerate(edges):
            cols = ["imagecol", "imagerow"]
            if forward:
                edge0, edge1 = edge
            else:
                edge0, edge1 = edge[::-1]

            # Arrow details #
            x1, y1 = adata.obs.loc[edge0, cols].values.astype(float) * scale_factor
            x2, y2 = adata.obs.loc[edge1, cols].values.astype(float) * scale_factor

            fig.add_layout(
                Arrow(
                    end=VeeHead(size=arrow_size),
                    line_color="black",
                    x_start=x1,
                    y_start=y1,
                    x_end=x2,
                    y_end=y2,
                )
            )

    def update_list(self, attrname, old, name):

        # Initialize the color
        from stlearn.plotting.cluster_plot import cluster_plot

        selected = self.annot_select.value.strip("raw_")
        cluster_plot(self.adata[0], use_label=selected, show_plot=False)

        # self.list_cluster = CheckboxGroup(
        #     labels=list(self.adata[0].obs[self.use_label.value].cat.categories),
        #     active=list(
        #         np.array(range(0, len(self.adata[0].obs[self.use_label.value].unique())))
        #     ),
        # )
        self.list_cluster.labels = list(self.adata[0].obs[selected].cat.categories)
        self.list_cluster.active = list(
            np.array(range(0, len(self.adata[0].obs[selected].unique())))
        )


class Annotate(Spatial):
    def __init__(
        self,
        # plotting param
        adata: AnnData,
    ):
        super().__init__(adata)
        # Open image, and make sure it's RGB*A*
        image = (self.img * 255).astype(np.uint8)

        img_pillow = Image.fromarray(image).convert("RGBA")

        self.xdim, self.ydim = img_pillow.size

        # Create an array representation for the image `img`, and an 8-bit "4
        # layer/RGBA" version of it `view`.
        self.image = np.empty((self.ydim, self.xdim), dtype=np.uint32)
        view = self.image.view(dtype=np.uint8).reshape((self.ydim, self.xdim, 4))
        # Copy the RGBA image into view, flipping it so it comes right-side up
        # with a lower-left origin
        view[:, :, :] = np.flipud(np.asarray(img_pillow))

        # Display the 32-bit RGBA image
        self.dim = max(self.xdim, self.ydim)

        self.data_alpha = Slider(
            title="Spot alpha", value=1.0, start=0, end=1.0, step=0.1
        )

        self.tissue_alpha = Slider(
            title="Tissue alpha", value=1.0, start=0, end=1.0, step=0.1
        )

        self.spot_size = Slider(
            title="Spot size", value=5.0, start=0, end=5.0, step=1.0
        )

        self.checkbox_group = CheckboxGroup(
            labels=["Show spatial trajectories"], active=[]
        )

        inputs = column(
            self.data_alpha,
            self.tissue_alpha,
            self.spot_size,
        )
        self.layout = row(inputs, self.make_fig())

        def modify_fig(doc):
            doc.add_root(row(self.layout, width=800))
            self.data_alpha.on_change("value", self.update_data)
            self.tissue_alpha.on_change("value", self.update_data)
            self.spot_size.on_change("value", self.update_data)

        handler = FunctionHandler(modify_fig)
        self.app = Application(handler)

    def update_data(self, attrname, old, new):
        self.layout.children[1] = self.make_fig()

    def make_fig(self):
        fig = figure(
            x_range=(0, self.dim - 150), y_range=(self.dim, 0), output_backend="webgl"
        )

        colors = np.repeat("black", len(self.imagecol))
        adding_colors = np.array(Category20[20])[
            list(sorted(list(range(0, 20)), key=lambda x: [x % 2, x]))
        ]
        s1 = ColumnDataSource(
            data=dict(x=self.imagecol, y=self.imagerow, colors=colors)
        )

        fig.image_rgba(
            image=[self.image],
            x=0,
            y=self.xdim,
            dw=self.ydim,
            dh=self.xdim,
            global_alpha=self.tissue_alpha.value,
        )

        fig.circle(
            "x",
            "y",
            size=self.spot_size.value,
            color="colors",
            source=s1,
            fill_alpha=self.data_alpha.value,
            line_alpha=self.data_alpha.value,
        )

        fig.toolbar.logo = None
        fig.xaxis.visible = False
        fig.yaxis.visible = False
        fig.xgrid.grid_line_color = None
        fig.ygrid.grid_line_color = None
        fig.outline_line_alpha = 0
        fig.add_tools(LassoSelectTool())
        fig.add_tools(BoxSelectTool())

        self.s2 = ColumnDataSource(data=dict(spot=[], label=[]))
        columns = [
            TableColumn(field="spot", title="Spots"),
            TableColumn(field="label", title="Labels"),
        ]
        table = DataTable(
            source=self.s2,
            columns=columns,
            width=400,
            height=200,
            reorderable=False,
            editable=True,
        )

        color_index = ColumnDataSource(data=dict(index=[0]))
        savebutton = Button(label="Add new group", button_type="success", width=200)
        savebutton.js_on_click(
            CustomJS(
                args=dict(
                    source_data=s1,
                    source_data_2=self.s2,
                    adding_colors=adding_colors,
                    color_index=color_index,
                    table=table,
                ),
                code="""

                function addRowToAccumulator(accumulator, spots, labels, index) {
                    accumulator['spot'][index] = spots;
                    accumulator['label'][index] = labels;

                    return accumulator;
                }

                var inds = source_data.selected.indices;
                var data = source_data.data;
                var add_color = adding_colors
                var colors = source_data.data.colors
                var i = 0;
                for (i; i < inds.length; i++) {
                    colors[inds[i]] = add_color[color_index.data.index[0]]
                }

                source_data.change.emit();

                var new_data =  source_data_2.data;

                new_data = addRowToAccumulator(new_data,inds,color_index.data.index[0].toString(),color_index.data.index[0])

                source_data_2.data = new_data

                source_data_2.change.emit();
                color_index.data.index[0] += 1
                color_index.change.emit();
                """,
            )
        )
        submitbutton = Button(label="Submit", button_type="success", width=200)

        def change_click():
            self.adata[0].uns["annotation"] = self.s2.to_df()
            empty_array = np.empty(len(self.adata[0]))
            empty_array[:] = np.NaN
            empty_array = empty_array.astype(object)
            for i in range(0, len(self.adata[0].uns["annotation"])):
                empty_array[
                    [np.array(self.adata[0].uns["annotation"]["spot"][i])]
                ] = str(self.adata[0].uns["annotation"]["label"][i])

            empty_array = pd.Series(empty_array).fillna("other")
            self.adata[0].obs["annotation"] = pd.Categorical(empty_array)

        submitbutton.on_click(change_click)
        submitbutton.js_on_click(
            CustomJS(
                args={},
                code="""
                alert("The annotated labels stored in adata.obs.annotation")
                try {
                        $( "#cluster_plot" ).removeClass( "disabled" )
                    }
                    catch (e) {}
                """,
            )
        )

        layout = column([fig, row(table, column(savebutton, submitbutton))])

        return layout
