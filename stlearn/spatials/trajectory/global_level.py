from anndata import AnnData
from typing import Optional, Union
import numpy as np
import pandas as pd
import networkx as nx

def global_level(
    adata: AnnData,
    use_labels: str = "louvain",
    pseudo_root: int = 0,
    eps: float = 20,
    copy: bool = False,
) -> Optional[AnnData]:
	
	# Localize 
	from stlearn.spatials.clustering import localization 
	localization(adata,use_labels=use_labels,eps = eps)

	# Running paga
	from stlearn.external.scanpy.api.tl import paga
	paga(adata,groups=use_labels)

	# Get connection matrix
	cnt_matrix = adata.uns["paga"]["connectivities"].toarray()

	# Filter by threshold
	threshold = 0.01

	cnt_matrix[cnt_matrix<threshold] = 0.
	cnt_matrix = pd.DataFrame(cnt_matrix)

	# Mapping louvain label to subcluster
	split_node = {}
	for label in adata.obs[use_labels].unique():
	    split_node[int(label)] = list(adata.obs[adata.obs[use_labels]==label]["sub_cluster_labels"].unique())

	adata.uns["split_node"] = split_node
	# Replicate louvain label row to prepare for subcluster connection matrix construction
	replicate_list = np.array([])
	for i in range(0,len(cnt_matrix)):
	    replicate_list = np.concatenate([replicate_list,np.array([i]*len(split_node[i]))])

	# Connection matrix for subcluster
	cnt_matrix= cnt_matrix.loc[replicate_list.astype(int),replicate_list.astype(int)]


	# Replace column and index
	cnt_matrix.columns = replace_with_dict(cnt_matrix.columns,split_node)
	cnt_matrix.index = replace_with_dict(cnt_matrix.index,split_node)

	# Sort column and index
	cnt_matrix = cnt_matrix.loc[selection_sort(np.array(cnt_matrix.columns)),
				selection_sort(np.array(cnt_matrix.index))]

	# Create a connection graph of subclusters
	G = nx.from_numpy_matrix(cnt_matrix.values)

	adata.uns['global_graph'] = G

	# Create centroid dict for subclusters
	from sklearn.neighbors import NearestCentroid
	clf = NearestCentroid()
	clf.fit(adata.obs[["imagecol","imagerow"]].values, adata.obs["sub_cluster_labels"])
	centroid_dict = dict(zip(clf.classes_.astype(int),clf.centroids_))
	adata.uns["centroid_dict"] = centroid_dict

	# Choose pseudo-root for the global level 
	adata.uns["iroot"] = np.flatnonzero(adata.obs[use_labels]  == str(pseudo_root))[0]

	# Running diffusion pseudo-time
	from stlearn.external.scanpy.api.tl import dpt
	dpt(adata)

	return adata if copy else None


######## utils ########

def replace_with_dict(ar, dic):
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    out = np.zeros_like(ar)
    for key,val in zip(k,v):
        out[ar==key] = val
    return out


def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x
