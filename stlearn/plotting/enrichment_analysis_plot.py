
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
from anndata import AnnData
#import cv2
import io
from matplotlib import cm


def enrichment_analysis_plot(
    adata: AnnData,
    method: str = "fa",
    factor: int = None,
    gene_sets: str = None,
    save_plot: str = None,
    cmap: str = "plasma",
    title: str = None,
    dpi: int = 192,
    store: bool = True,
    output: str = None,
    copy: bool = False,
) -> Optional[AnnData]:

    plt.rcParams['figure.dpi'] = dpi
    tmp = adata.uns["factor_sig"][method]["Factor_" +
                                          str(factor)][gene_sets]["result"]
    objects = tmp.Term[::-1]

    y_pos = np.arange(len(objects))
    performance = tmp['Odds Ratio'][::-1]
    fig, a = plt.subplots()

    y_pvalue = tmp["Adjusted P-value"][::-1]

    plot = plt.scatter(y_pvalue, y_pvalue, c=y_pvalue, cmap='plasma')

    colors = cm.plasma_r(y_pos / float(max(y_pos)))
    plt.clf()
    plt.colorbar(plot, orientation="horizontal", pad=0.2)
    plt.barh(range(len(y_pos)), performance, color=colors)

    #a.barh(y_pos, performance, align='center', alpha=0.5)
    plt.yticks(y_pos, objects)
    plt.xlabel('Odd ratio \n Adjusted P-value')
    if title is None:
        plt.title(gene_sets)
    else:
        plt.title(title)

    if output is not None:
        fig.savefig(output + "/enrich_plot_factor_" + str(factor) + "_" +
                    gene_sets + ".png", dpi=dpi, bbox_inches='tight', pad_inches=0)

    if store:

        fig_np = get_img_from_fig(fig, dpi)

        plt.close(fig)

        current_plot = {"img": fig_np}

        adata.uns["factor_sig"][method]["Factor_" +
                                        str(factor)][gene_sets].update(current_plot)

    print('The plot stored in adata.uns["factor_sig"]["' + method +
          '"]'+'["Factor_'+str(factor)+'"]["'+gene_sets+'"]["img"]')


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    from io import BytesIO

    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = np.asarray(Image.open(BytesIO(img_arr)))
    buf.close()
    #img = cv2.imdecode(img_arr, 1)
    #img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    return img
