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
)

from anndata import AnnData
from bokeh.palettes import Spectral11
from bokeh.layouts import column, row
from collections import OrderedDict
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from ..classes import Spatial

class BokehBasePlot(Spatial):
    def __init__(self,
        # plotting param
        adata: AnnData,
                ):
        super().__init__(
            adata,
        )
        # Open image, and make sure it's RGB*A*
        image = (self.img* 255).astype(np.uint8)

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
        self.header = Div(
                text="""<h2>Gene plot: </h2>""", width=400, height=50, sizing_mode="fixed"
            )
        self.data_alpha = Slider(title="Spot alpha", value=1.0, start=0, end=1.0, step=0.1)
        self.tissue_alpha = Slider(title="Tissue alpha", value=1.0, start=0, end=1.0, step=0.1)
        self.spot_size = Slider(title="Spot size", value=5.0, start=0, end=5.0, step=1.0)

        self.gene_select = AutocompleteInput(
                title="Gene:", value=gene_list[0], completions=gene_list, min_characters=1
            )
        inputs = column(self.header, self.gene_select, self.data_alpha, self.tissue_alpha,
                       self.spot_size)
        
        self.layout = row(inputs, self.make_fig())

        # Make a tab with the layout
        #self.tab = Tabs(tabs = [Panel(child=self.layout, title="Gene plot")])
        
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
            title="Gene plot",
            x_range=(0, self.dim - 150),
            y_range=(self.dim, 0),
        )

        colors = self._get_gene_expression([self.gene_select.value])

        s1 = ColumnDataSource(data=dict(x=self.imagecol, y=self.imagerow, color=colors))

        fig.image_rgba(
            image=[self.image], x=0, y=self.xdim, dw=self.ydim, dh=self.xdim, global_alpha=self.tissue_alpha.value
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

    def update_data(self,attrname, old, new):
        self.layout.children[1] = self.make_fig()


    def _get_gene_expression(self,gene_symbols):

        if gene_symbols[0] not in self.adata[0].var_names:
            raise ValueError(
                gene_symbols[0] + " is not exist in the data, please try another gene"
            )

        colors = self.adata[0][:, gene_symbols].to_df().iloc[:, -1]

        return colors