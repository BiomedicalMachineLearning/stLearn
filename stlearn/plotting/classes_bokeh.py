from __future__ import division
import numpy as np
from PIL import Image
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
)

from anndata import AnnData
from bokeh.palettes import Spectral11
from bokeh.layouts import column, row
from collections import OrderedDict
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from ..classes import Spatial
from typing import Optional
from ..utils import _read_graph


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
        inputs = column(
            self.gene_select, self.data_alpha, self.tissue_alpha, self.spot_size
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

        handler = FunctionHandler(modify_fig)

        self.app = Application(handler)

    def make_fig(self):

        fig = figure(
            title=self.gene_select.value,
            x_range=(0, self.dim - 150),
            y_range=(self.dim, 0),
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
                ("Gene expression", "@color{1.11}"),
            ]
        )

        return fig

    def update_data(self, attrname, old, new):
        self.layout.children[1] = self.make_fig()

    def _get_gene_expression(self, gene_symbols):

        if gene_symbols[0] not in self.adata[0].var_names:
            raise ValueError(
                gene_symbols[0] + " is not exist in the data, please try another gene"
            )

        colors = self.adata[0][:, gene_symbols].to_df().iloc[:, -1]

        return colors


class BokehClusterPlot(Spatial):
    def __init__(
        self,
        # plotting param
        adata: AnnData,
        use_label: Optional[str] = None,
    ):
        super().__init__(adata, use_label=use_label)
        self.use_label = use_label
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

        p = Paragraph(text="""Choose clusters:""", width=400, height=20)
        self.list_cluster = CheckboxGroup(
            labels=list(self.adata[0].obs[self.use_label].cat.categories),
            active=list(
                np.array(range(0, len(self.adata[0].obs[self.use_label].unique())))
            ),
        )

        inputs = column(
            self.data_alpha,
            self.tissue_alpha,
            self.spot_size,
            p,
            self.list_cluster,
            self.checkbox_group,
        )
        self.layout = row(inputs, self.make_fig())

        def modify_fig(doc):
            doc.add_root(row(self.layout, width=800))
            self.data_alpha.on_change("value", self.update_data)
            self.tissue_alpha.on_change("value", self.update_data)
            self.spot_size.on_change("value", self.update_data)
            self.list_cluster.on_change("active", self.update_data)
            self.checkbox_group.on_change("active", self.update_data)

        handler = FunctionHandler(modify_fig)
        self.app = Application(handler)

    def update_data(self, attrname, old, new):
        self.layout.children[1] = self.make_fig()

    def make_fig(self):
        fig = figure(
            title="Cluster plot",
            x_range=(0, self.dim),
            y_range=(self.dim, 0),
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
                self.use_label
                + ' == "'
                + self.adata[0].obs[self.use_label].cat.categories[i]
                + '"'
            )
        tmp = self.adata[0].obs.query(" or ".join(command))

        tmp_adata = self.adata[0][tmp.index, :]

        x = tmp_adata.obsm["spatial"][:, 0] * self.scale_factor
        y = tmp_adata.obsm["spatial"][:, 1] * self.scale_factor

        category_items = self.adata[0].obs[self.use_label].cat.categories
        palette = self.adata[0].uns[self.use_label + "_colors"]
        colormap = dict(zip(category_items, palette))
        color = list(tmp[self.use_label].map(colormap))
        cluster = list(tmp[self.use_label])

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


class BokehCciPlot(Spatial):
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
        available_het = []
        for key in list(self.adata[0].obsm.keys()):
            if len(self.adata[0].obsm[key].shape) == 1:
                available_het.append(key)

        self.data_alpha = Slider(
            title="Spot alpha", value=1.0, start=0, end=1.0, step=0.1
        )
        self.tissue_alpha = Slider(
            title="Tissue alpha", value=1.0, start=0, end=1.0, step=0.1
        )
        self.spot_size = Slider(
            title="Spot size", value=5.0, start=0, end=5.0, step=1.0
        )

        self.het_select = AutocompleteInput(
            title="Het:",
            value=available_het[0],
            completions=available_het,
            min_characters=1,
        )
        inputs = column(
            self.het_select, self.data_alpha, self.tissue_alpha, self.spot_size
        )

        self.layout = row(inputs, self.make_fig())

        # Make a tab with the layout
        # self.tab = Tabs(tabs = [Panel(child=self.layout, title="Gene plot")])

        def modify_fig(doc):

            doc.add_root(row(self.layout, width=800))

            self.data_alpha.on_change("value", self.update_data)
            self.tissue_alpha.on_change("value", self.update_data)
            self.spot_size.on_change("value", self.update_data)
            self.het_select.on_change("value", self.update_data)

        handler = FunctionHandler(modify_fig)

        self.app = Application(handler)

    def make_fig(self):

        fig = figure(
            title=self.het_select.value,
            x_range=(0, self.dim - 150),
            y_range=(self.dim, 0),
        )

        colors = self._get_het(self.het_select.value)

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
