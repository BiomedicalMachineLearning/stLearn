
import pandas as pd
from anndata import AnnData


def stack_3d_plot(
        adata: AnnData,
        slides,
        height,
        width,
        cmap="viridis",
        slide_col="sample_id",
        use_label=None,
        gene_symbol=None,
) -> AnnData | None:
    """\
    Clustering plot for spatial transcriptomics data. Also, it has a function to
    display trajectory inference.

    Parameters
    ----------
    adata:
        Annotated data matrix.
    slides:
        A list of slide id
    width:
        Width of the plot
    height:
        Height of the plot
    cmap:
        Color map
    slide_col:
        Obs column to use for coloring.
    use_label:
        Choose label to plot (priotize)
    gene_symbol
        Choose gene symbol to plot
    Returns
    -------
    Nothing
    """
    try:
        import plotly.express as px
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please install plotly by `pip install plotly`")

    assert (
            slide_col in adata.obs.columns
    ), "Please provide the right column for slide_id!"

    list_df = []
    for i, slide in enumerate(slides):
        tmp = adata.obs[adata.obs[slide_col] == slide][["imagecol", "imagerow"]]
        tmp["sample_id"] = slide
        tmp["z-dimension"] = i
        list_df.append(tmp)

    df = pd.concat(list_df)

    if use_label is not None:
        assert use_label in adata.obs.columns, "Please use the right `use_label`"
        df[use_label] = adata[df.index].obs[use_label].values

        fig = px.scatter_3d(
            df,
            x="imagecol",
            y="imagerow",
            z="z-dimension",
            color=use_label,
            width=width,
            height=height,
            color_continuous_scale=cmap,
        )
        fig.show()

    else:
        assert gene_symbol in adata.var_names, "Please use the right `gene_symbol`"
        df[gene_symbol] = adata[df.index][:, gene_symbol].X

        fig = px.scatter_3d(
            df,
            x="imagecol",
            y="imagerow",
            z="z-dimension",
            color=gene_symbol,
            width=width,
            height=height,
            color_continuous_scale=cmap,
        )
        fig.show()
