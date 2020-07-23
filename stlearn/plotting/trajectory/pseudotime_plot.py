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
#from .utils import get_img_from_fig, checkType

def pseudotime_plot(
    adata: AnnData,
    library_id: str = None,
    use_label: str = "louvain",
    use_pseudotime: str = "dpt_pseudotime",
    list_cluster: Union[str,list] = None,
    data_alpha: float = 1.0,
    tissue_alpha: float = 1.0,
    edge_alpha: float = 0.8,
    node_alpha: float = 1.0,
    spot_size: Union[float,int] = 6.5,
    node_size: float = 5,
    show_color_bar: bool = True,
    show_axis: bool = False,
    show_graph: bool = True,
    show_plot: bool = True,
    cropped: bool = True,
    margin: int = 100,
    output: str = None,
    name: str = None,
    dpi: int = 180,
    copy: bool = False,
) -> Optional[AnnData]:

    """\
    Global trajectory inference plot (Only DPT).

    Parameters
    ----------
    adata
        Annotated data matrix.
    library_id
        Library id stored in AnnData.
    use_label
        Use label result of clustering method.
    list_cluster
        Choose set of clusters that will display in the plot.
    data_alpha
        Opacity of the spot.
    tissue_alpha
        Opacity of the tissue.
    edge_alpha
        Opacity of edge in PAGA graph in the tissue.
    node_alpha
        Opacity of node in PAGA graph in the tissue.
    cmap
        Color map to use.
    spot_size
        Size of the spot.
    node_size
        Size of node in PAGA graph in the tissue.
    show_color_bar
        Show color bar or not.
    show_axis
        Show axis or not.
    show_graph
        Show PAGA graph or not.
    show_legend
        Show legend or not.
    show_plot
        Show plot or not
    dpi
        Set dpi as the resolution for the plot.
    output
        Save the figure as file or not.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Nothing
    """
    
    plt.rcParams['figure.dpi'] = dpi

    imagecol = adata.obs["imagecol"]
    imagerow = adata.obs["imagerow"]

    if list_cluster == "all":
        list_cluster = list(range(0,len(adata.obs[use_label].unique())))
    # Get query clusters
    command = []
    for i in list_cluster:
        command.append(use_label + ' == "' + str(i) + '"')
    tmp = adata.obs.query(" or ".join(command))

    G=adata.uns["global_graph"]

    labels = nx.get_edge_attributes(G,'weight')

    result = []
    query_node = get_node(list_cluster,adata.uns["split_node"])
    for edge in G.edges(query_node):
        if (edge[0] in query_node) and (edge[1] in query_node):
            result.append(edge)

    result2 = []
    for edge in result:
        try:
            result2.append(labels[edge]+0.5)
        except: 
            result2.append(labels[edge[::-1]]+0.5)

    fig, a = plt.subplots()
    centroid_dict = adata.uns["centroid_dict"]
    dpt = adata.obs[use_pseudotime]

    colors = adata.obs[use_label].astype(int)
    vmin = min(dpt)
    vmax = max(dpt)
    # Plot scatter plot based on pixel of spots
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scale = scaler.fit_transform(tmp[use_pseudotime].values.reshape(-1,1)).reshape(-1,1)

    plot = a.scatter(tmp["imagecol"], tmp["imagerow"], edgecolor="none", alpha=data_alpha,s=spot_size,marker="o",
               vmin=vmin, vmax=vmax,cmap=plt.get_cmap("viridis"),c=scale.reshape(1,-1)[0])

    n_clus = len(colors.unique())
    
    from scanpy.plotting import palettes
    cmaps = matplotlib.colors.LinearSegmentedColormap.from_list("",palettes.vega_20_scanpy)

    cmap = plt.get_cmap(cmaps)
    bounds=np.linspace(0, n_clus, n_clus+1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    


    norm = matplotlib.colors.Normalize(vmin=min(colors), vmax=max(colors))
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    if show_graph:
        nx.draw_networkx_edges(G, pos=centroid_dict,node_size=1,font_size=0,linewidths=2,edgelist=result,
                                width=result2,alpha=edge_alpha,edge_color='#333333',)

        for x,y in centroid_dict.items():

            if(x in get_node(list_cluster,adata.uns["split_node"])):
                a.text(y[0],y[1],get_cluster(str(x),adata.uns["split_node"]),color='white',fontsize = node_size,zorder=100,
                       bbox=dict(facecolor=cmap(int(get_cluster(str(x),adata.uns["split_node"]))/19),boxstyle='circle',alpha=node_alpha))
    if show_color_bar:
        cb = plt.colorbar(plot,cmap="viridis")
        cb.outline.set_visible(False)

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    image = adata.uns["spatial"][library_id]["images"][adata.uns["spatial"]["use_quality"]]
    
    a.imshow(image,alpha=tissue_alpha, zorder=-1,)

    if not show_axis:
        a.axis('off')

    if cropped:
        a.set_xlim(imagecol.min() - margin,
                imagecol.max() + margin)

        a.set_ylim(imagerow.min() - margin,
                imagerow.max() + margin)
        
        a.set_ylim(a.get_ylim()[::-1])
        #plt.gca().invert_yaxis()

    if output is not None:
        fig.savefig(output + "/" + name + ".png", dpi=dpi,
                    bbox_inches='tight', pad_inches=0)
    if show_plot == True:
        plt.show()

# get name of cluster by subcluster
def get_cluster(search,dictionary):
    for cl, sub in dictionary.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
        if search in sub:
            return(cl)
        
def get_node(node_list,split_node):
    result = np.array([])
    for node in node_list:
        result = np.append(result,np.array(split_node[node]).astype(int))
    return result.astype(int)