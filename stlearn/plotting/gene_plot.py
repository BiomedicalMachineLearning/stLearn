import matplotlib
from anndata import AnnData
from bokeh.io import output_notebook
from bokeh.plotting import show

from stlearn.plotting._docs import doc_gene_plot, doc_spatial_base_plot
from stlearn.plotting.classes import GenePlot
from stlearn.plotting.classes_bokeh import BokehGenePlot
from stlearn.utils import _docs_params


@_docs_params(spatial_base_plot=doc_spatial_base_plot, gene_plot=doc_gene_plot)
def gene_plot(
    adata: AnnData,
    gene_symbols: str | list | None = None,
    threshold: float | None = None,
    method: str = "CumSum",
    contour: bool = False,
    step_size: int | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "Spectral_r",
    use_label: str | None = None,
    list_clusters: list | None = None,
    ax: matplotlib.axes.Axes | None = None,
    fig: matplotlib.figure.Figure | None = None,
    show_plot: bool = True,
    show_axis: bool = False,
    show_image: bool = True,
    show_color_bar: bool = True,
    color_bar_label: str = "",
    zoom_coord: tuple[float, float, float, float] | None = None,
    crop: bool = True,
    margin: float = 100,
    size: float = 7,
    image_alpha: float = 1.0,
    cell_alpha: float = 0.7,
    use_raw: bool = False,
    fname: str | None = None,
    dpi: int = 120,
    vmin: float | None = None,
    vmax: float | None = None,
) -> AnnData | None:
    """\
    Allows the visualization of a single gene or multiple genes as the values
    of dot points or contour in the Spatial transcriptomics array.


    Parameters
    -------------------------------------
    {spatial_base_plot}
    {gene_plot}

    Examples
    -------------------------------------
    >>> import stlearn as st
    >>> adata = st.datasets.visium_sge(sample_id="V1_Breast_Cancer_Block_A_Section_1")
    >>> genes = ["BRCA1","BRCA2"]
    >>> st.pl.gene_plot(adata, gene_symbols = genes)

    """
    GenePlot(
        adata,
        gene_symbols=gene_symbols,
        threshold=threshold,
        method=method,
        contour=contour,
        step_size=step_size,
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
        color_bar_label=color_bar_label,
        zoom_coord=zoom_coord,
        crop=crop,
        margin=margin,
        size=size,
        image_alpha=image_alpha,
        cell_alpha=cell_alpha,
        use_raw=use_raw,
        fname=fname,
        dpi=dpi,
        vmin=vmin,
        vmax=vmax,
    )
    return adata


def gene_plot_interactive(adata: AnnData):
    bokeh_object = BokehGenePlot(adata)
    output_notebook()
    show(bokeh_object.app, notebook_handle=True)
