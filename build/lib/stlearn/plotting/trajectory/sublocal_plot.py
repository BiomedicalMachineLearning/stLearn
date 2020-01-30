from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import matplotlib
import numpy as np
import networkx as nx
from stlearn._compat import Literal
from typing import Optional, Union
from anndata import AnnData
import warnings
from .utils import checkType

def sublocal_plot(
    adata: AnnData,
    name: str = None,
    subcluster: int = 0,
    route: str = "S0",
    data_alpha: float = 1.0,
    tissue_alpha: float = 1.0,
    title: str = None,
    spot_size: Union[float,int] = 6.5,
    show_color_bar: bool = True,
    show_axis: bool = False,
    dpi: int = 180,
    output: str = None,
    cmap: str = "cool",
    show_root: bool = False,
    copy: bool = False,
) -> Optional[AnnData]:
    
    plt.rcParams['figure.dpi'] = dpi

    comp1=0
    comp2=1
    key_graph='epg'
    epg = adata.uns["subcluster_"+str(subcluster)+"_pts"]['epg']
    flat_tree = adata.uns["subcluster_"+str(subcluster)+"_pts"]['flat_tree']
    dict_nodes_pos = nx.get_node_attributes(epg,'pos')

    n_route = len(flat_tree.edges())

    fig, a = plt.subplots()

    tmp = adata.uns["subcluster_"+str(subcluster)+"_adata"]

    current_pseudotime = tmp.obs[["node",route + "_pseudotime"]]
    color = []
    colors = np.array(current_pseudotime[route + "_pseudotime"])
    vmin = min(colors)
    vmax = max(colors)

    sc = a.scatter(tmp.obs["imagecol"], tmp.obs["imagerow"], edgecolor="none", alpha=0.8,s=6,marker="o",
           vmin=vmin, vmax=vmax,cmap=plt.get_cmap(cmap),c=colors)

    a.imshow(adata.uns["tissue_img"],alpha=1.0, zorder=-1,)

    for edge_i in flat_tree.edges():
        branch_i_nodes = flat_tree.edges[edge_i]['nodes']

        #if branch_i_nodes[0] != edge_i[0]:
        #        branch_i_nodes = branch_i_nodes[::-1]

        direction_arr = []
        for node in branch_i_nodes:
            tmpx = current_pseudotime[current_pseudotime["node"]==node]
            if len(tmpx)>0:
                direction_arr.append(tmpx.iloc[:,1][0])


        if  len(direction_arr) > 1:
                if not checkType(direction_arr):
                    branch_i_nodes = branch_i_nodes[::-1]


        branch_i_color = "#f4efd3"
        branch_i_pos = np.array([dict_nodes_pos[i] for i in branch_i_nodes])

        edgex = branch_i_pos[:,0]
        edgey = branch_i_pos[:,1]
        a.plot(edgex,edgey,c = branch_i_color,lw=2,zorder=1)
        for j in range(0,len(edgex)):
            a.arrow(edgex[j],edgey[j],edgex[j+1]-edgex[j],edgey[j+1]-edgey[j],color="red",length_includes_head=True,
                     head_width=5, head_length=5, linewidth=0,zorder=4)
            if j == len(edgex)-2:
                break
    if not show_axis:
        a.axis('off')

    

    if show_root:
        dict_node_state = nx.get_node_attributes(flat_tree,'label')
        for key,val in dict_node_state.items():
            a.text(dict_nodes_pos[key][0],dict_nodes_pos[key][1],val,fontsize=5,zorder=5)
    plt.show()