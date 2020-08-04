from matplotlib import pyplot as plt
import numpy as np
from typing import Optional, Union
from anndata import AnnData


def QC_plot(
        adata: AnnData,
        library_id: str = None,
        name: str = None,
        data_alpha: float = 0.8,
        tissue_alpha: float = 1.0,
        cmap: str = "Spectral_r",
        spot_size: tuple = (5, 40),
        show_color_bar: bool = True,
        show_size_legend: bool = True,
        show_axis: bool = False,
        cropped: bool = True,
        margin: int = 100,
        dpi: int = 192,
        output: str = None,
) -> Optional[AnnData]:
    """\
        QC plot for sptial transcriptomics data.

        Parameters
        ----------
        adata
            Annotated data matrix.
        library_id
            Library id stored in AnnData.
        data_alpha
            Opacity of the spot.
        tissue_alpha
            Opacity of the tissue.
        cmap
            Color map to use.
        spot_size
            Size of the spot (min, max).
        show_color_bar
            Show color bar or not.
        show_axis
            Show axis or not.
        show_size_legend
            Show size legend or not.
        dpi
            Set dpi as the resolution for the plot.
        name
            Name of the output figure file.
        output
            Save the figure as file or not.
        copy
            Return a copy instead of writing to adata.
        Returns
        -------
        Nothing
        """

    imagecol = adata.obs["imagecol"]
    imagerow = adata.obs["imagerow"]
    from sklearn.preprocessing import MinMaxScaler
    reads_per_spot = adata.to_df().sum(axis=1)
    scaler = MinMaxScaler(feature_range=spot_size)
    reads_per_spot_size = scaler.fit_transform(
        reads_per_spot.to_numpy().reshape(-1, 1))
    genes_per_spot = adata.to_df().astype(bool).sum(axis=1)

    plt.rcParams['figure.dpi'] = dpi

    # Option for turning off showing figure
    plt.ioff()

    # Initialize matplotlib
    fig, a = plt.subplots()

    vmin = min(genes_per_spot)
    vmax = max(genes_per_spot)
    # Plot scatter plot based on pixel of spots
    plot = a.scatter(adata.obs["imagecol"], adata.obs["imagerow"], edgecolor="none", alpha=data_alpha,
                     s=reads_per_spot_size, marker="o",
                     vmin=vmin, vmax=vmax, cmap=plt.get_cmap(cmap), c=genes_per_spot)

    if show_color_bar:
        cb = plt.colorbar(plot, cax=fig.add_axes(
            [0.85, 0.5, 0.03, 0.35]), cmap=cmap)
        cb.ax.set_xlabel('Number of Genes', fontsize=10)
        cb.ax.xaxis.set_label_coords(0.98, 1.10)
        cb.outline.set_visible(False)

    if show_size_legend:
        size_min, size_max = spot_size
        markers = [size_min, size_min + 1 / 3 * (size_max - size_min),
                   size_min + 2 / 3 * (size_max - size_min), size_max]
        legend_markers = [plt.scatter([], [], s=i, c="grey") for i in markers]
        labels = [str(int(scaler.inverse_transform(np.array(i).reshape(1, 1))))
                  for i in markers]
        fig.legend(handles=legend_markers, labels=labels, loc='lower right', bbox_to_anchor=(0.5, 0.1),
                   scatterpoints=1, frameon=False, handletextpad=0.1, title="Number of Reads")

    if not show_axis:
        a.axis('off')
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    image = adata.uns["spatial"][library_id]["images"][adata.uns["spatial"]["use_quality"]]
    # Overlay the tissue image
    a.imshow(image, alpha=tissue_alpha, zorder=-1, )

    if cropped:
        a.set_xlim(imagecol.min() - margin,
                imagecol.max() + margin)

        a.set_ylim(imagerow.min() - margin,
                imagerow.max() + margin)
        
        a.set_ylim(a.get_ylim()[::-1])
        #plt.gca().invert_yaxis()

    # fig.tight_layout()
    if output is not None:
        fig.savefig(output + "/" + name + ".png", dpi=dpi,
                    bbox_inches='tight', pad_inches=0)

    plt.show()
