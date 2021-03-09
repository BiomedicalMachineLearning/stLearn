import matplotlib.pyplot as plt
from decimal import Decimal
from anndata import AnnData
from typing import Optional, Union


def transition_markers_plot(
    adata: AnnData,
    top_genes: int = 10,
    trajectory: str = None,
    dpi: int = 150,
    output: str = None,
    name: str = None,
) -> Optional[AnnData]:
    """\
    Plot transition marker.

    Parameters
    ----------
    adata
        Annotated data matrix.
    top_genes
        Top genes users want to display in the plot.
    trajectory
        Name of a clade/branch user wants to plot transition markers.
    dpi
        The resolution of the plot.
    output
        The output folder of the plot.
    name
        The filename of the plot.
    Returns
    -------
    Anndata
    """

    if trajectory == None:
        raise ValueError("Please input the trajectory name!")
    if trajectory not in adata.uns:
        raise ValueError("Please input the right trajectory name!")

    pos = (
        adata.uns[trajectory][adata.uns[trajectory]["score"] >= 0]
        .sort_values("score", ascending=False)
        .reset_index(drop=True)[:top_genes]
    )
    neg = (
        adata.uns[trajectory][adata.uns[trajectory]["score"] < 0]
        .sort_values("score")
        .reset_index(drop=True)[:top_genes]
    )

    y = range(top_genes)
    x1 = list(neg["score"])[::-1]
    x2 = list(pos["score"])[::-1]

    pos = pos[::-1]
    neg = neg[::-1]

    if len(x1) < top_genes:
        for i in range(len(x1), top_genes):
            x1.append(0)
    if len(x2) < top_genes:
        for i in range(len(x2), top_genes):
            x2.append(0)

    fig, axes = plt.subplots(ncols=2, sharey=True)
    axes[0].barh(y, x1, align="center", color="#fb687a")
    axes[1].barh(y, x2, align="center", color="#31a2fb")
    fig.subplots_adjust(wspace=0)
    axes[0].spines["left"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["top"].set_visible(False)
    axes[1].spines["left"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["top"].set_visible(False)
    axes[0].get_yaxis().set_ticks([])
    axes[1].get_yaxis().set_ticks([])
    axes[0].tick_params(axis="both", which="both", length=0)
    axes[1].tick_params(axis="both", which="both", length=0)

    for i, x in enumerate([x1, x2]):
        if all(value == 0 for value in x):
            if i == 0:
                axes[0].get_xaxis().set_ticks([])
            if i == 1:
                axes[1].get_xaxis().set_ticks([])

    rects = axes[1].patches
    for i, rect in enumerate(rects):
        try:
            gene_name = list(pos["gene"])[i]
            p_value = "{:.2E}".format(Decimal(str(list(pos["p-value"])[i])))
        except:
            gene_name = ""
            p_value = ""
        alignment = {"horizontalalignment": "left", "verticalalignment": "center"}
        axes[1].text(
            rect.get_x() + rect.get_width() + 0.01,
            rect.get_y() + rect.get_height() / 2.0,
            gene_name,
            **alignment,
            size=6
        )
        axes[1].text(
            rect.get_x() + 0.01,
            rect.get_y() + rect.get_height() / 2.0,
            p_value,
            color="w",
            **alignment,
            size=6
        )

    rects = axes[0].patches
    for i, rect in enumerate(rects):
        try:
            gene_name = list(neg["gene"])[i]
            p_value = "{:.2E}".format(Decimal(str(list(neg["p-value"])[i])))
        except:
            gene_name = ""
            p_value = ""
        alignment = {"horizontalalignment": "right", "verticalalignment": "center"}
        axes[0].text(
            rect.get_x() + rect.get_width() - 0.01,
            rect.get_y() + rect.get_height() / 2.0,
            gene_name,
            **alignment,
            size=6
        )
        axes[0].text(
            rect.get_x() - 0.01,
            rect.get_y() + rect.get_height() / 2.0,
            p_value,
            color="w",
            **alignment,
            size=6
        )

    plt.figtext(0.5, 0.9, trajectory, ha="center", va="center")
    axes[0].set_xlabel("Spearman correlation coefficient")
    axes[0].xaxis.set_label_coords(1, -0.1)

    axes[0].grid(False)
    axes[1].grid(False)

    if name is None:
        name = trajectory

    if output is not None:
        fig.savefig(output + "/" + name, dpi=dpi, bbox_inches="tight", pad_inches=0)

    plt.show()
