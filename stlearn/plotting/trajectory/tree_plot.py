from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import matplotlib
import numpy as np
import networkx as nx
import random
from stlearn._compat import Literal
from typing import Optional, Union
from anndata import AnnData
import warnings
import io
from copy import deepcopy


def tree_plot(
    adata: AnnData,
    figsize: Union[float,int] = (10,6),
    data_alpha: float = 1.0,
    use_label: str = "louvain",
    cmap: str = "vega_20_scanpy",
    spot_size: Union[float,int] = 50,
    piesize: float = 0.1,
    zoom: float = 0.4,
    dpi: int = 180,
    output: str = None,
    copy: bool = False,
) -> Optional[AnnData]:
    G = adata.uns["PTS_graph"]

    for node in G.nodes:
        if node == 9999:
            break
        tmp_img = _generate_image(adata,sub_cluster=node,zoom=zoom, spot_size=spot_size)

        G.nodes[node]['image'] = tmp_img

    plt.rcParams['figure.dpi'] = dpi
    pos = hierarchy_pos(G,9999) 
    fig=plt.figure(figsize=figsize)
    a= plt.subplot(111)

    a.axis('off')
    nx.draw_networkx_edges(G,pos,a=a,arrowstyle="-",with_labels=True,edge_color="#ADABAF")

    trans=a.transData.transform
    trans2=fig.transFigure.inverted().transform

    p2=piesize/2

    for n in G:
        if n == 9999:
            xx,yy=trans(pos[n]) # figure coordinates
            xa,ya=trans2((xx,yy)) # axes coordinates
            a = plt.axes([xa-p2,ya-p2, piesize, piesize])
            a.axis('off')
            a.text(0.5,1,"PSEUDO-ROOT",horizontalalignment='center',verticalalignment='center', transform=a.transAxes)
            break
            
        xx,yy=trans(pos[n]) # figure coordinates
        xa,ya=trans2((xx,yy)) # axes coordinates
        a = plt.axes([xa-p2,ya-p2, piesize, piesize])
        a.text(0.5,1.2,str(n),horizontalalignment='center',verticalalignment='center', transform=a.transAxes)
        #a.set_aspect('equal')
        a.axis('off')
        a.imshow(G.nodes[n]['image'])




def _generate_image(adata,sub_cluster,zoom = 0.5,spot_size=100,dpi=96,use_label="louvain",data_alpha=1):
    
    subset = adata.obs[adata.obs["sub_cluster_labels"]==str(sub_cluster)]
    base = subset[["imagecol","imagerow"]].values
    if len(base)<25:
        zoom = zoom*20
        spot_size = spot_size*3
    elif len(base)>2000:
        zoom = zoom/4
        spot_size = spot_size/5
    x = base[:,0]
    y = base[:,1]
    #plt.rcParams['figure.dpi'] = 300

    fig2, ax2 = plt.subplots()
    color = adata.uns["tmp_color"][int(subset[use_label][0])]

    ax2.imshow(adata.uns["tissue_img"],alpha=1)
    ax2.scatter(x,y,s=spot_size,alpha=data_alpha,edgecolor="none",c=color)
    ax2.axis('off')
    #ax2.get_xaxis().set_ticks([])
    #ax2.get_yaxis().set_ticks([])

    try:
        ptp_bound = np.array(base).ptp(axis=0)
    except:
        print(base)
        
    
    ax2.set_xlim(x.min() - ptp_bound[0]*zoom,
                x.max() + ptp_bound[0]*zoom)
    ax2.set_ylim(y.min() - ptp_bound[1]*zoom,
                y.max() + ptp_bound[1]*zoom)
    plt.gca().invert_yaxis()
    
    buf = io.BytesIO()
    fig2.savefig(buf, format='png', dpi = dpi,transparent=True,bbox_inches='tight',pad_inches=0)
    buf.seek(0)
    pil_img = deepcopy(Image.open(buf))
    plt.close(fig2)
    return pil_img






def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 

    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)