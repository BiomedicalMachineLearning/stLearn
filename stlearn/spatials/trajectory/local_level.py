from anndata import AnnData
from typing import Optional, Union
import numpy as np
from stlearn.em import run_pca,run_diffmap
from stlearn.pp import neighbors

def local_level(
    adata: AnnData,
    use_label_lvl2: str = "louvain",
    use_data_lvl2: str = "X_diffmap",
    cluster: int = 9,
    copy: bool = False,
) -> Optional[AnnData]:
    
    print("Start construct trajectory for subcluster " + str(cluster))
    
    tmp=adata.obs[adata.obs[use_label_lvl2]==str(cluster)]
    cluster_data = adata[list(tmp.index)]


    #run_pca(cluster_data, svd_solver='arpack')
    #neighbors(cluster_data, n_neighbors=4, n_pcs=20)

    #run_diffmap(cluster_data)


    #neighbors(cluster_data, n_neighbors=10, use_rep=use_data_lvl2)

    #from stlearn.external.scanpy.api.tl import draw_graph

    #draw_graph(cluster_data)
    #from sklearn.metrics import pairwise_distances_argmin_min
    #x = [p[0] for p in cluster_data.obsm["X_draw_graph_fa"]]
    #y = [p[1] for p in cluster_data.obsm["X_draw_graph_fa"]]
    #centroid = (sum(x) / len(cluster_data.obsm["X_draw_graph_fa"]), sum(y) / len(cluster_data.obsm["X_draw_graph_fa"]))
    #closest, _ = pairwise_distances_argmin_min([centroid], cluster_data.obsm["X_draw_graph_fa"])
    #cluster_data.uns["cluster_" +str(cluster) +'_iroot'] = closest[0]

    #from stlearn.external.scanpy.api.tl import dpt
    #dpt(cluster_data)

    

    average_time = {}
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scale = scaler.fit_transform(cluster_data.obs['dpt_pseudotime'].values.reshape(-1,1)).reshape(-1,1)
    cluster_data.obs["dpt_pseudotime"] = scale 
    
    adata.uns["local_cluster_"+str(cluster)] = cluster_data
    
    for subcl in cluster_data.obs["sub_cluster_labels"].unique():
        average_time[subcl] = cluster_data.obs[cluster_data.obs["sub_cluster_labels"]==subcl]["dpt_pseudotime"].mean()

    adata.uns["cluster_" +str(cluster) + "_dpt"] = average_time

    return adata if copy else None


        


    