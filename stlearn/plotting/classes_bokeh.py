from __future__ import division
import numpy as np
import pandas as pd
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
    Button,
    Dropdown,
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

        inputs = column(
            self.gene_select,
            self.data_alpha,
            self.tissue_alpha,
            self.spot_size,
            self.cmap_select,
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

        handler = FunctionHandler(modify_fig)

        self.app = Application(handler)

    def make_fig(self):

        fig = figure(
            title=self.gene_select.value,
            x_range=(0, self.dim - 150),
            y_range=(self.dim, 0),
            output_backend="svg",
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

        cluster_plot(adata, use_label=self.use_label.value, show_plot=False)

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

        self.inputs = column(
            self.use_label,
            self.data_alpha,
            self.tissue_alpha,
            self.spot_size,
            self.p,
            self.list_cluster,
            self.checkbox_group,
        )
        self.layout = row(self.inputs, self.make_fig())

        def modify_fig(doc):
            doc.add_root(row(self.layout, width=800))
            self.use_label.on_change("value", self.update_list)
            self.use_label.on_change("value", self.update_data)

            self.data_alpha.on_change("value", self.update_data)
            self.tissue_alpha.on_change("value", self.update_data)
            self.spot_size.on_change("value", self.update_data)
            self.list_cluster.on_change("active", self.update_data)
            self.checkbox_group.on_change("active", self.update_data)

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

        self.inputs = column(
            self.use_label,
            self.data_alpha,
            self.tissue_alpha,
            self.spot_size,
            self.p,
            self.list_cluster,
            self.checkbox_group,
        )

        self.layout.children[0] = self.inputs

    def update_data(self, attrname, old, new):

        self.layout.children[1] = self.make_fig()

    def make_fig(self):
        fig = figure(
            title="Cluster plot",
            x_range=(0, self.dim),
            y_range=(self.dim, 0),
            output_backend="svg"
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
            output_backend="svg",
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
            x_range=(0, self.dim - 150), y_range=(self.dim, 0), output_backend="svg"
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
