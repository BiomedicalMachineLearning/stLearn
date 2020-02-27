from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import matplotlib
import numpy as np

from stlearn._compat import Literal
from typing import Optional, Union
from anndata import AnnData
import warnings

def subcluster_plot(
    adata: AnnData,
    name: str = None,
    use_label: str = "louvain",
    cluster: int = 0,
    data_alpha: float = 1.0,
    tissue_alpha: float = 1.0,
    cmap: str = "jet",
    title: str = None,
    spot_size: Union[float,int] = 5,
    show_axis: bool = False,
    dpi: int = 192,
    output: str = None,
    copy: bool = False,
) -> Optional[AnnData]:
    
    plt.rcParams['figure.dpi'] = dpi
    # Check condition
    #if len(adata.obs[adata.obs[use_label]==str(cluster)]["sub_cluster_labels"].unique()) < 2:
    #    raise ValueError("There is no subcluster from this cluster! Please choose other cluster")

    if str(cluster) not in adata.obs[use_label].unique():
        raise ValueError("This cluster is non-exist, please choose another cluster!")

    if use_label not in adata.obs.columns:
        raise ValueError("This label is non-exist, please choose another label!")
    plt.rcParams['figure.dpi'] = dpi

    colors = adata.obs[adata.obs[use_label]==str(cluster)]["sub_cluster_labels"]
    keys = list(np.sort(colors.unique()))
    vals = np.arange(len(keys))
    mapping = dict(zip(keys,vals))

    # Option for turning off showing figure
    #plt.ioff()
    colors = colors.replace(mapping)
    
    # Initialize matplotlib
    fig, a = plt.subplots()

    # Plot scatter plot based on pixel of spots
    plot = a.scatter(adata.obs[adata.obs[use_label]==str(cluster)]["imagecol"], adata.obs[adata.obs[use_label]==str(cluster)]["imagerow"], 
               edgecolor="none",s=spot_size,marker="o",cmap=plt.get_cmap(cmap),c=colors, alpha=data_alpha)


    if len(adata.obs[adata.obs[use_label]==str(cluster)]["sub_cluster_labels"].unique()) < 2:
        centroids = [centeroidpython(adata.obs[adata.obs[use_label]==str(cluster)][["imagecol","imagerow"]].values)]
        classes = np.array([adata.obs[adata.obs[use_label]==str(cluster)]["sub_cluster_labels"][0]])
        
    else:
        from sklearn.neighbors import NearestCentroid
        clf = NearestCentroid()
        clf.fit(adata.obs[adata.obs[use_label]==str(cluster)][["imagecol","imagerow"]].values, 
            adata.obs[adata.obs[use_label]==str(cluster)]["sub_cluster_labels"])

        centroids = clf.centroids_
        classes = clf.classes_

    norm = matplotlib.colors.Normalize(vmin=min(vals), vmax=max(vals))

    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    for i,label in enumerate(classes):
        a.text(centroids[i][0],centroids[i][1],label,color='white',fontsize = 5,zorder=3,
               bbox=dict(facecolor=matplotlib.colors.to_hex(m.to_rgba(mapping[label])),boxstyle='round',alpha=0.5))

    a.imshow(adata.uns["tissue_img"],alpha=tissue_alpha, zorder=-1,)
    


    if title is not None:
        a.set_title(title)
    if not show_axis:
        a.axis('off')

    plt.show()
    

    if output is not None:
        if name is None:
            print("The file name is not defined!")
            name = use_label
        fig.savefig(output + "/" + name + ".png", dpi=dpi,bbox_inches='tight',pad_inches=0)


    

    #plt.close(fig)

    #if "plots" not in adata.uns:
    #    adata.uns['plots'] = {}
    #adata.uns['plots'].update({"subcl_" + use_label + "_" + str(cluster): fig_np})

    #print("The plot stored in adata.uns['plots']['" + "subcl_" + use_label + "_" + str(cluster)+ "']")

    #import matplotlib.pyplot as plt
    #plt.rcParams['figure.dpi'] = dpi
    #plt.imshow(adata.uns['plots']["subcl_" + use_label + "_" + str(cluster)])
    #plt.axis('off')
    

def centeroidpython(data):
    x, y = zip(*data)
    l = len(x)
    return sum(x) / l, sum(y) / l