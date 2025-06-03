
# from .utils import get_img_from_fig, checkType
import scanpy
from anndata import AnnData


def non_spatial_plot(
        adata: AnnData,
        use_label: str = "louvain",
) -> AnnData | None:
    """\
    A wrap function to plot all the non-spatial plot from scanpy.

    This function will produce 3 plots: PAGA graph, cluster plot in PAGA space and
    DPT in PAGA space.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_label
        Use label result of cluster method.
    dpi
        Set dpi as the resolution for the plot.
    Returns
    -------
    Nothing
    """

    # plt.rcParams['figure.dpi'] = dpi

    if "paga" in adata.uns.keys():
        # adata.uns[use_label+"_colors"] = adata.uns["tmp_color"]

        print("PAGA plot:")

        scanpy.pl.paga(adata, color=use_label)

        scanpy.tl.draw_graph(adata, init_pos="paga")
        # adata.uns[use_label+"_colors"] = adata.uns["tmp_color"]

        print("Gene expression (reduced dimension) plot:")
        scanpy.pl.draw_graph(adata, color=use_label, legend_loc="on data")

        print("Diffusion pseudotime plot:")
        scanpy.pl.draw_graph(adata, color="dpt_pseudotime")

    else:

        scanpy.pl.draw_graph(adata)
        # adata.uns[use_label+"_colors"] = adata.uns["tmp_color"]

        scanpy.pl.draw_graph(adata, color=use_label, legend_loc="on data")
