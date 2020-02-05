from anndata import AnnData
from typing import Optional, Union
import numpy as np
from stlearn.em import run_pca,run_diffmap
from stlearn.pp import neighbors

def sublocal_level(
    adata: AnnData,
    subcluster: Union[int,list] = 0,
    use_data: str = "X_diffmap",
    alpha: float = 0.5,
    auto_tune: bool = True,
    nodes_per_log10_cells: int =20,
    n_neighbors: int = None,
    epg_n_nodes: int = 20,
    incr_n_nodes: int = 10,
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

		
	print("Start construct trajectory for subcluster " + str(subcluster))

	if type(subcluster) == list:
		query = ''
		for i in subcluster:
		     query = " | ".join([query,'sub_cluster_labels == "' + str(i) + '"'])

		tmp = adata.obs.query(query[3:])
	else:
		tmp = adata.obs[adata.obs["sub_cluster_labels"]==str(subcluster)]

	subcluster_data = adata[list(tmp.index)]
	
	from stlearn.preprocessing.graph import neighbors

	if n_neighbors is not None:
		auto_tune = False

	if auto_tune:
		def n_neighbors_cal(ncells,nodes_per_log10_cells=20):
		    import math
		    return round(nodes_per_log10_cells * math.log10(ncells))
		n_neighbors = n_neighbors_cal(len(tmp))
	#from stlearn.spatials.clustering import clustgeo
	neighbors(subcluster_data,n_neighbors=n_neighbors,use_rep='X_umap_disk')

	from stlearn.tools.clustering import louvain

	louvain(subcluster_data)

	#clustgeo(subcluster_data,use_data=use_data,n_clusters=cal_ncenter(len(tmp),
	#	nodes_per_log10_cells=nodes_per_log10_cells),alpha=alpha)

	from stlearn.spatials.clustering import localization

	subcluster_data.obs = subcluster_data.obs.drop("sub_cluster_labels",axis=1)
	localization(subcluster_data,use_labels="louvain",eps=10)

	from .pseudotimespace import initialize_graph, pseudotimespace_epg

	initialize_graph(subcluster_data)

	pseudotimespace_epg(subcluster_data, 
						epg_n_nodes = epg_n_nodes,
					    incr_n_nodes= incr_n_nodes,
					    epg_lambda= epg_lambda,
					    epg_mu=  epg_mu,
					    epg_trimmingradius= epg_trimmingradius,
					    epg_finalenergy= epg_finalenergy,
					    epg_alpha= epg_alpha,
					    epg_beta= epg_beta,
					    epg_n_processes= epg_n_processes,
					    nReps= nReps,
					    ProbPoint= ProbPoint,)
	adata.uns["subcluster_" + str(subcluster) + "_pts"] = subcluster_data.uns["pseudotimespace"]

	del subcluster_data.uns["paga"]
	adata.uns["subcluster_" + str(subcluster) + "_adata"] = subcluster_data

#def cal_ncenter(ncells,nodes_per_log10_cells=8):
#    import math
#    return round(nodes_per_log10_cells * math.log10(ncells))