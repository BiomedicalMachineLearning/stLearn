import scipy.spatial as spatial
import numpy as np
import networkx as nx
from anndata import AnnData
from typing import Optional, Union

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid

from rpy2.robjects.packages import importr
from rpy2.robjects import r as R
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

from .extra import *


def initialize_graph(
    adata: AnnData,
    copy: bool = False,
) -> Optional[AnnData]:

    coor = adata.obs[["imagecol","imagerow"]]

    adata.obs["epg_cluster"] = adata.obs["sub_cluster_labels"]

    clf = NearestCentroid()
    clf.fit(coor, adata.obs["epg_cluster"])

    centroids = clf.centroids_
    adata.uns["pseudotimespace"] = {}

    adata.uns["pseudotimespace"].update({"epg_centroids": centroids})

    init_nodes_pos = centroids
    epg_nodes_pos = init_nodes_pos
    D=pairwise_distances(epg_nodes_pos)
    G=nx.from_numpy_matrix(D)
    mst=nx.minimum_spanning_tree(G)
    epg_edges = np.array(mst.edges())
    epg=nx.Graph()
    epg.add_nodes_from(range(epg_nodes_pos.shape[0]))
    epg.add_edges_from(epg_edges)
    dict_nodes_pos = {i:x for i,x in enumerate(epg_nodes_pos)}
    nx.set_node_attributes(epg,values=dict_nodes_pos,name='pos')

    dict_branches = extract_branches(epg)
    flat_tree = construct_flat_tree(dict_branches)

    nx.set_node_attributes(flat_tree,values={x:dict_nodes_pos[x] for x in flat_tree.nodes()},name='pos')

    adata.uns["pseudotimespace"].update({'epg': deepcopy(epg)})
    adata.uns["pseudotimespace"].update({'flat_tree': deepcopy(flat_tree)})
    adata.uns["pseudotimespace"].update({'seed_epg': deepcopy(epg)})
    adata.uns["pseudotimespace"].update({'seed_flat_tree': deepcopy(flat_tree)})

    project_cells_to_epg(adata)
    calculate_pseudotime(adata)

    return adata if copy else None


def pseudotimespace_epg(
    adata: AnnData,
    epg_n_nodes: int = 10,
    incr_n_nodes: int = 0,
    epg_lambda: int = 0.03,
    epg_mu: int = 0.01,
    epg_trimmingradius: str = 'Inf',
    epg_finalenergy: str = 'Penalized',
    epg_alpha: float = 0.02,
    epg_beta: float = 0.0,
    epg_n_processes: int = 1,
    nReps: int = 1,
    ProbPoint: int = 1,

    copy: bool = False,
) -> Optional[AnnData]:

    input_data = adata.uns["pseudotimespace"]["epg_centroids"]
    #input_data = adata.obs[["imagecol","imagerow"]].values
    epg = adata.uns["pseudotimespace"]['seed_epg']
    dict_nodes_pos = nx.get_node_attributes(epg,'pos')
    init_nodes_pos = np.array(list(dict_nodes_pos.values()))
    init_edges = np.array(list(epg.edges())) 
    R_init_edges = init_edges + 1

    if((init_nodes_pos.shape[0])<epg_n_nodes):
        print('epg_n_nodes is too small. It is corrected to the initial number of nodes plus incr_n_nodes')
        epg_n_nodes = init_nodes_pos.shape[0]+incr_n_nodes



    ElPiGraph = importr('ElPiGraph.R')
    pandas2ri.activate()
    print('Learning elastic principal graph...')
    import sys, os

    # Disable
    def blockPrint():
        sys.stdout = open(os.devnull, 'w')
    # Restore
    def enablePrint():
        sys.stdout = sys.__stdout__
        
    #blockPrint()
    epg_obj = ElPiGraph.computeElasticPrincipalTree(X=input_data,
                                                        NumNodes = epg_n_nodes, 
                                                        #InitNodePositions = init_nodes_pos,
                                                        #InitEdges=R_init_edges,
                                                        Lambda=epg_lambda, Mu=epg_mu,
                                                        TrimmingRadius= epg_trimmingradius,
                                                        FinalEnergy = epg_finalenergy,
                                                        alpha = epg_alpha,
                                                        beta = epg_beta,                                                    
                                                        Do_PCA=False,CenterData=False,
                                                        n_cores = 1,
                                                        nReps=nReps,
                                                        ProbPoint=ProbPoint,
                                                        drawAccuracyComplexity = False,
                                                        drawPCAView = False,
                                                        drawEnergy = False,
                                                        verbose = False,
                                                        ShowTimer=False,
                                                        ComputeMSEP=False)
    #enablePrint()

    epg_nodes_pos = np.array(epg_obj[0].rx2('NodePositions'))
    epg_edges = np.array((epg_obj[0].rx2('Edges')).rx2('Edges'),dtype=int)-1

    #store graph information and update adata
    epg=nx.Graph()
    epg.add_nodes_from(range(epg_nodes_pos.shape[0]))
    epg.add_edges_from(epg_edges)
    dict_nodes_pos = {i:x for i,x in enumerate(epg_nodes_pos)}
    nx.set_node_attributes(epg,values=dict_nodes_pos,name='pos')
    dict_branches = extract_branches(epg)
    flat_tree = construct_flat_tree(dict_branches)
    nx.set_node_attributes(flat_tree,values={x:dict_nodes_pos[x] for x in flat_tree.nodes()},name='pos')
    adata.uns["pseudotimespace"]['epg'] = deepcopy(epg)
    adata.uns["pseudotimespace"]['ori_epg'] = deepcopy(epg)
    adata.uns["pseudotimespace"]['epg_obj'] = deepcopy(epg_obj)    
    adata.uns["pseudotimespace"]['ori_epg_obj'] = deepcopy(epg_obj)
    adata.uns["pseudotimespace"]['flat_tree'] = deepcopy(flat_tree)
    project_cells_to_epg(adata)
    calculate_pseudotime(adata)
    print('Number of branches after learning elastic principal graph: ' + str(len(dict_branches)))

    return adata if copy else None
