import numpy as np
import pandas as pd
import networkx as nx
import pylab as plt
from copy import deepcopy
import itertools
from scipy.spatial import distance,cKDTree,KDTree
from scipy import interpolate
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import math
from scipy.linalg import eigh, svd, qr, solve
from decimal import *

def project_point_to_line_segment_matrix(XP,p):
    XP = np.array(XP,dtype=float)
    p = np.array(p,dtype=float)
    AA=XP[:-1,:]
    BB=XP[1:,:]
    AB = (BB-AA)
    AB_squared = (AB*AB).sum(1)
    Ap = (p-AA)
    t = (Ap*AB).sum(1)/AB_squared
    t[AB_squared == 0]=0
    Q = AA + AB*np.tile(t,(XP.shape[1],1)).T
    Q[t<=0,:]=AA[t<=0,:]
    Q[t>=1,:]=BB[t>=1,:]
    kdtree=cKDTree(Q)
    d,idx_q=kdtree.query(p)
    dist_p_to_q = np.sqrt(np.inner(p-Q[idx_q,:],p-Q[idx_q,:]))
    XP_p = np.row_stack((XP[:idx_q+1],Q[idx_q,:]))
    lam = np.sum(np.sqrt(np.square((XP_p[1:,:] - XP_p[:-1,:])).sum(1)))
    return list([Q[idx_q,:],idx_q,dist_p_to_q,lam])

def project_cells_to_epg(adata):
    input_data = adata.obs[["imagecol","imagerow"]].values
    epg = adata.uns["pseudotimespace"]['epg']
    dict_nodes_pos = nx.get_node_attributes(epg,'pos')
    nodes_pos = np.empty((0,input_data.shape[1]))
    nodes = np.empty((0,1),dtype=int)
    for key in dict_nodes_pos.keys():
        nodes_pos = np.vstack((nodes_pos,dict_nodes_pos[key]))
        nodes = np.append(nodes,key)    
    indices = pairwise_distances_argmin_min(input_data,nodes_pos,axis=1,metric='euclidean')[0]
    x_node = nodes[indices]
    adata.obs['node'] = x_node
    #update the projection info for each cell
    flat_tree = adata.uns["pseudotimespace"]['flat_tree']
    dict_branches_nodes = nx.get_edge_attributes(flat_tree,'nodes')
    dict_branches_id = nx.get_edge_attributes(flat_tree,'id')
    dict_node_state = nx.get_node_attributes(flat_tree,'label')
    list_x_br_id = list()
    list_x_br_id_alias = list()
    list_x_lam = list()
    list_x_dist = list()
    for ix,xp in enumerate(input_data): 
        list_br_id = [flat_tree.edges[br_key]['id'] for br_key,br_value in dict_branches_nodes.items() if x_node[ix] in br_value]
        dict_br_matrix = dict()
        for br_id in list_br_id:
            dict_br_matrix[br_id] = np.array([dict_nodes_pos[i] for i in flat_tree.edges[br_id]['nodes']])            
        dict_results = dict()
        list_dist_xp = list()
        for br_id in list_br_id:
            dict_results[br_id] = project_point_to_line_segment_matrix(dict_br_matrix[br_id],xp)
            list_dist_xp.append(dict_results[br_id][2])
        x_br_id = list_br_id[np.argmin(list_dist_xp)]
        x_br_id_alias = dict_node_state[x_br_id[0]],dict_node_state[x_br_id[1]]
        br_len = flat_tree.edges[x_br_id]['len']
        results = dict_results[x_br_id]
        x_dist = results[2]
        x_lam = results[3]
        if(x_lam>br_len):
            x_lam = br_len 
        list_x_br_id.append(x_br_id)
        list_x_br_id_alias.append(x_br_id_alias)
        list_x_lam.append(x_lam)
        list_x_dist.append(x_dist)
    adata.obs['branch_id'] = list_x_br_id
    adata.obs['branch_id_alias'] = list_x_br_id_alias
#     adata.uns['branch_id'] = list(set(adata.obs['branch_id'].tolist()))
    adata.obs['branch_lam'] = list_x_lam
    adata.obs['branch_dist'] = list_x_dist
    return None

def calculate_pseudotime(adata):
    flat_tree = adata.uns["pseudotimespace"]['flat_tree']
    dict_edge_len = nx.get_edge_attributes(flat_tree,'len')
    adata.obs = adata.obs[adata.obs.columns.drop(list(adata.obs.filter(regex='_pseudotime')))].copy()
    # dict_nodes_pseudotime = dict()
    for root_node in flat_tree.nodes():
        df_pseudotime = pd.Series(index=adata.obs.index)
        list_bfs_edges = list(nx.bfs_edges(flat_tree,source=root_node))
        dict_bfs_predecessors = dict(nx.bfs_predecessors(flat_tree,source=root_node))
        for edge in list_bfs_edges:
            list_pre_edges = list()
            pre_node = edge[0]
            while(pre_node in dict_bfs_predecessors.keys()):
                pre_edge = (dict_bfs_predecessors[pre_node],pre_node)
                list_pre_edges.append(pre_edge)
                pre_node = dict_bfs_predecessors[pre_node]
            len_pre_edges = sum([flat_tree.edges[x]['len'] for x in list_pre_edges]) 
            indices = adata.obs[(adata.obs['branch_id'] == edge) | (adata.obs['branch_id'] == (edge[1],edge[0]))].index
            if(edge==flat_tree.edges[edge]['id']):
                df_pseudotime[indices] = len_pre_edges + adata.obs.loc[indices,'branch_lam']
            else:
                df_pseudotime[indices] = len_pre_edges + (flat_tree.edges[edge]['len']-adata.obs.loc[indices,'branch_lam'])
        adata.obs[flat_tree.nodes[root_node]['label']+'_pseudotime'] = df_pseudotime
        # dict_nodes_pseudotime[root_node] = df_pseudotime
    # nx.set_node_attributes(flat_tree,values=dict_nodes_pseudotime,name='pseudotime')
    # adata.uns['flat_tree'] = flat_tree
    return None

def construct_flat_tree(dict_branches):
    flat_tree = nx.Graph()
    flat_tree.add_nodes_from(list(set(itertools.chain.from_iterable(dict_branches.keys()))))
    flat_tree.add_edges_from(dict_branches.keys())
    root = list(flat_tree.nodes())[0]
    edges = nx.bfs_edges(flat_tree, root)
    nodes = [root] + [v for u, v in edges]  
    dict_nodes_label = dict()
    for i,node in enumerate(nodes):
        dict_nodes_label[node] = 'S'+str(i) 
    nx.set_node_attributes(flat_tree,values=dict_nodes_label,name='label')
    dict_branches_color = dict()
    dict_branches_len = dict()
    dict_branches_nodes = dict()
    dict_branches_id = dict() #the direction of nodes for each edge
    for x in dict_branches.keys():

        dict_branches_len[x]=dict_branches[x]['len']
        dict_branches_nodes[x]=dict_branches[x]['nodes']
        dict_branches_id[x]=dict_branches[x]['id'] 
    nx.set_edge_attributes(flat_tree,values=dict_branches_nodes,name='nodes')
    nx.set_edge_attributes(flat_tree,values=dict_branches_id,name='id')
    nx.set_edge_attributes(flat_tree,values=dict_branches_len,name='len')    
    return flat_tree

def extract_branches(epg):
    #record the original degree(before removing nodes) for each node
    degrees_of_nodes = epg.degree()
    epg_copy = epg.copy()
    dict_branches = dict()
    clusters_to_merge=[]
    while epg_copy.order()>1: #the number of vertices
        leaves=[n for n,d in epg_copy.degree() if d==1]
        nodes_included=list(epg_copy.nodes())
        while leaves:
            leave=leaves.pop()
            nodes_included.remove(leave)
            nodes_to_merge=[leave]
            nodes_to_visit=list(epg_copy.nodes())
            dfs_from_leaf(epg_copy,leave,degrees_of_nodes,nodes_to_visit,nodes_to_merge)
            clusters_to_merge.append(nodes_to_merge)
            dict_branches[(nodes_to_merge[0],nodes_to_merge[-1])] = {}
            dict_branches[(nodes_to_merge[0],nodes_to_merge[-1])]['nodes'] = nodes_to_merge
            nodes_to_delete = nodes_to_merge[0:len(nodes_to_merge)-1]
            if epg_copy.degree()[nodes_to_merge[-1]] == 1: #avoid the single point
                nodes_to_delete = nodes_to_merge
                leaves = []
            epg_copy.remove_nodes_from(nodes_to_delete)
    dict_branches = add_branch_info(epg,dict_branches)
    # print('Number of branches: ' + str(len(clusters_to_merge)))
    return dict_branches

def dfs_from_leaf(epg_copy,node,degrees_of_nodes,nodes_to_visit,nodes_to_merge):
    nodes_to_visit.remove(node)
    for n2 in epg_copy.neighbors(node):
        if n2 in nodes_to_visit:
            if degrees_of_nodes[n2]==2:  #grow the branch
                if n2 not in nodes_to_merge:
                    nodes_to_merge.append(n2)
                dfs_from_leaf(epg_copy,n2,degrees_of_nodes,nodes_to_visit,nodes_to_merge)
            else:
                nodes_to_merge.append(n2)
                return

def add_branch_info(epg,dict_branches):
    dict_nodes_pos = nx.get_node_attributes(epg,'pos')
    if(dict_nodes_pos != {}):
        for i,(br_key,br_value) in enumerate(dict_branches.items()):
            nodes = br_value['nodes']
            dict_branches[br_key]['id'] = (nodes[0],nodes[-1]) #the direction of nodes for each branch
            br_nodes_pos = np.array([dict_nodes_pos[i] for i in nodes]) 
            dict_branches[br_key]['len'] = sum(np.sqrt(((br_nodes_pos[0:-1,:] - br_nodes_pos[1:,:])**2).sum(1)))
    return dict_branches

##############################

def bfs_edges_modified(tree, source, preference=None):
    visited, queue = [], [source]
    bfs_tree = nx.bfs_tree(tree,source=source)
    predecessors = dict(nx.bfs_predecessors(bfs_tree,source))
    edges = []
    while queue:
        vertex = queue.pop()
        if vertex not in visited:
            visited.append(vertex)
            if(vertex in predecessors.keys()):
                edges.append((predecessors[vertex],vertex))
            unvisited = set(tree[vertex]) - set(visited)
            if(preference != None):
                weights = list()
                for x in unvisited:
                    successors = dict(nx.bfs_successors(bfs_tree,source=x))
                    successors_nodes = list(itertools.chain.from_iterable(successors.values()))
                    weights.append(min([preference.index(s) if s in preference else len(preference) for s in successors_nodes+[x]]))
                unvisited = [x for _,x in sorted(zip(weights,unvisited),reverse=True,key=lambda x: x[0])]
            queue.extend(unvisited)
    return edges


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])