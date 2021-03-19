import matplotlib.pyplot as plt
from decimal import Decimal
from typing import Optional, Union
from anndata import AnnData


def DE_transition_plot(
    adata: AnnData,
    top_genes: int = 10,
    font_size: int = 6,
    name: str = None,
    dpi: int = 150,
    output: str = None,
) -> Optional[AnnData]:

    """\
    Differential expression between transition markers.

    Parameters
    ----------
    adata
        Annotated data matrix.
    top_genes
        Number of genes using to plot.
    font_size
        Size of the font.
    name
        Name of the output figure file.
    dpi
        DPI of the output figure.
    output
        Save the figure as file or not.
    Returns
    -------
    Figure object
    """

    trajectories = adata.uns["compare_result"]["trajectories"]
    pos_1 = (
        adata.uns[trajectories[0]]
        .set_index("gene")
        .loc[adata.uns["compare_result"]["pos_1"][:top_genes]]
        .iloc[::-1]
    )
    pos_2 = (
        adata.uns[trajectories[1]]
        .set_index("gene")
        .loc[adata.uns["compare_result"]["pos_2"][:top_genes]]
        .iloc[::-1]
    )
    neg_1 = (
        adata.uns[trajectories[0]]
        .set_index("gene")
        .loc[adata.uns["compare_result"]["neg_1"][:top_genes]]
        .iloc[::-1]
    )
    neg_2 = (
        adata.uns[trajectories[1]]
        .set_index("gene")
        .loc[adata.uns["compare_result"]["neg_2"][:top_genes]]
        .iloc[::-1]
    )

    y = range(top_genes)
    x1 = list(neg_1["score"])
    x2 = list(pos_1["score"])
    x3 = list(neg_2["score"])
    x4 = list(pos_2["score"])

    if len(x1) < top_genes:
        for i in range(len(x1), top_genes):
            x1.append(0)
    if len(x2) < top_genes:
        for i in range(len(x2), top_genes):
            x2.append(0)
    if len(x3) < top_genes:
        for i in range(len(x3), top_genes):
            x3.append(0)
    if len(x4) < top_genes:
        for i in range(len(x4), top_genes):
            x4.append(0)

    fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True)
    fig.subplots_adjust(wspace=0, hspace=0.8)
    axes[0][0].barh(y, x1, align="center", color="#fb687a")
    axes[0][1].barh(y, x2, align="center", color="#31a2fb")
    axes[1][0].barh(y, x3, align="center", color="#fb687a")
    axes[1][1].barh(y, x4, align="center", color="#31a2fb")

    for i, x in enumerate([x1, x2, x3, x4]):
        if all(value == 0 for value in x):
            if i == 0:
                axes[0][0].get_xaxis().set_ticks([])
            if i == 1:
                axes[0][1].get_xaxis().set_ticks([])
            if i == 2:
                axes[1][0].get_xaxis().set_ticks([])
            if i == 3:
                axes[1][1].get_xaxis().set_ticks([])

    axes[0][0].spines["left"].set_visible(False)
    axes[0][0].spines["right"].set_visible(False)
    axes[0][0].spines["top"].set_visible(False)
    axes[1][0].spines["left"].set_visible(False)
    axes[1][0].spines["right"].set_visible(False)
    axes[1][0].spines["top"].set_visible(False)

    axes[0][1].spines["right"].set_visible(False)
    axes[0][1].spines["top"].set_visible(False)
    axes[0][1].spines["left"].set_visible(False)
    axes[1][1].spines["right"].set_visible(False)
    axes[1][1].spines["top"].set_visible(False)
    axes[1][1].spines["left"].set_visible(False)

    axes[0][0].get_yaxis().set_ticks([])
    axes[1][0].get_yaxis().set_ticks([])
    axes[0][0].tick_params(axis="both", which="both", length=0)
    axes[1][0].tick_params(axis="both", which="both", length=0)

    axes[0][1].get_yaxis().set_ticks([])
    axes[1][1].get_yaxis().set_ticks([])
    axes[0][1].tick_params(axis="both", which="both", length=0)
    axes[1][1].tick_params(axis="both", which="both", length=0)

    rects = axes[0][1].patches
    for i, rect in enumerate(rects):
        try:
            gene_name = pos_1.index[i]
            p_value = "{:.2E}".format(Decimal(str(pos_1["p-value"][i])))
        except:
            gene_name = ""
            p_value = ""
        alignment = {"horizontalalignment": "left", "verticalalignment": "center"}
        axes[0][1].text(
            rect.get_x() + rect.get_width() + 0.01,
            rect.get_y() + rect.get_height() / 2.0,
            gene_name,
            **alignment,
            size=font_size
        )
        axes[0][1].text(
            rect.get_x() + 0.01,
            rect.get_y() + rect.get_height() / 2.0,
            p_value,
            color="w",
            **alignment,
            size=font_size
        )

    rects = axes[0][0].patches
    for i, rect in enumerate(rects):
        try:
            gene_name = neg_1.index[i]
            p_value = "{:.2E}".format(Decimal(str(neg_1["p-value"][i])))
        except:
            gene_name = ""
            p_value = ""
        alignment = {"horizontalalignment": "right", "verticalalignment": "center"}
        axes[0][0].text(
            rect.get_x() + rect.get_width() - 0.01,
            rect.get_y() + rect.get_height() / 2.0,
            gene_name,
            **alignment,
            size=font_size
        )
        axes[0][0].text(
            rect.get_x() - 0.01,
            rect.get_y() + rect.get_height() / 2.0,
            p_value,
            color="w",
            **alignment,
            size=font_size
        )

    rects = axes[1][1].patches
    for i, rect in enumerate(rects):
        try:
            gene_name = pos_2.index[i]
            p_value = "{:.2E}".format(Decimal(str(pos_2["p-value"][i])))
        except:
            gene_name = ""
            p_value = ""
        alignment = {"horizontalalignment": "left", "verticalalignment": "center"}
        axes[1][1].text(
            rect.get_x() + rect.get_width() + 0.01,
            rect.get_y() + rect.get_height() / 2.0,
            gene_name,
            **alignment,
            size=font_size
        )
        axes[1][1].text(
            rect.get_x() + 0.01,
            rect.get_y() + rect.get_height() / 2.0,
            p_value,
            color="w",
            **alignment,
            size=font_size
        )

    rects = axes[1][0].patches
    for i, rect in enumerate(rects):
        try:
            gene_name = neg_2.index[i]
            p_value = "{:.2E}".format(Decimal(str(neg_2["p-value"][i])))
        except:
            gene_name = ""
            p_value = ""
        alignment = {"horizontalalignment": "right", "verticalalignment": "center"}
        axes[1][0].text(
            rect.get_x() + rect.get_width() - 0.01,
            rect.get_y() + rect.get_height() / 2.0,
            gene_name,
            **alignment,
            size=font_size
        )
        axes[1][0].text(
            rect.get_x() - 0.01,
            rect.get_y() + rect.get_height() / 2.0,
            p_value,
            color="w",
            **alignment,
            size=font_size
        )

    plt.figtext(
        0.5,
        0.5,
        "Markers of " + trajectories[0] + " compared to " + trajectories[1],
        ha="center",
        va="center",
    )
    plt.figtext(
        0.5,
        0.0,
        "Markers of " + trajectories[1] + " compared to " + trajectories[0],
        ha="center",
        va="center",
    )
    plt.show()
    if output is not None:
        if name is not None:
            plt.savefig(output + "/" + name, dpi=dpi, bbox_inches="tight", pad_inches=0)
