from anndata import AnnData
from typing import Optional, Union
import numpy as np
from stlearn.em import run_pca,run_diffmap
from stlearn.pp import neighbors
from scipy.spatial.distance import cdist

def local_level(
    adata: AnnData,
    use_labels: str = "louvain",
    cluster: int = 9,
    w: float = 0.5,
    copy: bool = False,
) -> Optional[AnnData]:
    
    print("Start construct trajectory for subcluster " + str(cluster))
    
    tmp=adata.obs[adata.obs[use_labels]==str(cluster)]
    cluster_data = adata[list(tmp.index)]

    list_cluster = cluster_data.obs["sub_cluster_labels"].unique()
    dpt = []
    sd = []
    centroid_dict = cluster_data.uns["centroid_dict"]
    for i in list_cluster:
        dpt.append(cluster_data.obs[cluster_data.obs["sub_cluster_labels"]==i]["dpt_pseudotime"].max())
        sd.append(centroid_dict[int(i)])
    dm = cdist(np.array(dpt).reshape(-1,1),np.array(dpt).reshape(-1,1),lambda u,v: np.abs(u-v))

    non_abs_dm = cdist(np.array(dpt).reshape(-1,1),np.array(dpt).reshape(-1,1),lambda u,v: u-v)
    adata.uns["nonabs_dpt_distance_matrix"] = non_abs_dm


    scale_dm = dm/np.max(dm)
    sdm = cdist(np.array(sd), np.array(sd), "euclidean")
    scale_sdm = sdm/np.max(sdm)

    stdm = scale_dm*w + scale_sdm*(1-w)
    adata.uns["ST_distance_matrix"] = stdm



    

    return adata if copy else None





    