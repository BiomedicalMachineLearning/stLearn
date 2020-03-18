from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib
import numpy as np
import networkx as nx
from stlearn._compat import Literal
from typing import Optional, Union
from anndata import AnnData
import warnings

def local_plot(
    adata: AnnData,
    name: str = None,
    use_label: str = "louvain",
    cluster: int = 0,
    data_alpha: float = 1.0,
    arrow_alpha: float = 1.0,
    branch_alpha: float = 1.0,
    title: str = None,
    spot_size: Union[float,int] = 1,
    show_color_bar: bool = True,
    show_axis: bool = False,
    dpi: int = 180,
    output: str = None,
    copy: bool = False,
) -> Optional[AnnData]:


    ref_cluster = "local_cluster_" + str(cluster)
    ref_dpt = "cluster_" +str(cluster) + "_dpt"

    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['figure.figsize'] = 5, 5
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    centroids_ = []
    classes_ = []
    for i in adata.uns[ref_cluster].obs["sub_cluster_labels"].unique():
        classes_.append(i)
        centroids_.append(adata.uns["centroid_dict"][int(i)])

    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler()
    #centroids_scaled = scaler.fit_transform(centroids_)

    for i in range(0,len(centroids_)):
        if i == len(centroids_)-1:
            break
        j=0
        while j <= (len(centroids_)-2-i):
            j=j+1
            
            m = adata.uns[ref_dpt][classes_[i]]-adata.uns[ref_dpt][classes_[i+j]]
            y = calculate_y(np.abs(m))
            
            x = np.linspace(centroids_[i][0],centroids_[i+j][0], 1000)
            z = np.linspace(centroids_[i][1],centroids_[i+j][1], 1000)
            
            
            branch = ax.plot(x,y,z,zorder=10,c="#333333",linewidth=1,alpha=branch_alpha)
            
            if m <=0:
                xyz = ([x[500],x[520]],[y[500],y[520]],[z[500],z[520]])
            else:
                xyz = ([x[520],x[500]],[y[520],y[500]],[z[520],z[500]])
            
            arrow = Arrow3D(xyz[0],xyz[1],xyz[2],zorder=10,mutation_scale=5, 
                   lw=1, arrowstyle="simple", color="r",alpha=arrow_alpha,)
            ax.add_artist(arrow)
            
            ax.text(x[500], y[500]-0.15, z[500], np.round(np.abs(m),2), color='black',size=5,zorder=100)
        
            
    sc = ax.scatter(adata.uns[ref_cluster].obs["imagecol"],
               0,
               adata.uns[ref_cluster].obs["imagerow"],
               c=adata.uns[ref_cluster].obs["dpt_pseudotime"],
               s=spot_size,cmap = "viridis",zorder=0, alpha=data_alpha)

    _ = ax.scatter(adata.obs[adata.obs[use_label]!=cluster]["imagecol"],
               0,
               adata.obs[adata.obs[use_label]!=cluster]["imagerow"],
               c="grey",
               s=spot_size,zorder=0,alpha=0.1)

    if show_color_bar:
        cb = fig.colorbar(sc,cax = fig.add_axes([0.1, 0.3, 0.03, 0.5]))
        cb.outline.set_visible(False)
        
    ax.set_ylim([-1,0])
    ax.set_xlim([min(adata.obs["imagecol"])-10,max(adata.obs["imagecol"])+10])
    ax.set_zlim([min(adata.obs["imagerow"])-10,max(adata.obs["imagerow"])+10])

    if not show_axis:
    # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.get_zaxis().set_ticks([])

    ax.invert_zaxis()
    ax.patch.set_visible(False) 
    plt.show()
    plt.rcParams['figure.figsize'] = 6, 4
    
    fig.save

def calculate_y(m):
    import math
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    y = np.linspace(mu - 0.5*sigma, mu + 0.5*sigma, 1000)
    y = np.cos(np.absolute(y))
    y = -(m)*(y - np.min(y))/np.ptp(y)
    
    return y



class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)