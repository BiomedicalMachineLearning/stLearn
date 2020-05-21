from anndata import AnnData
import stlearn
import numpy as np

def PseudoTime(
    adata: AnnData,
    use_label: str ="louvain",
    root_cluster: int = 0,
    root_point: int = 0,
    eps: float = 100,
    radius: int = 50,
    use_sme: bool = False,
    copy: bool = False,

    ) -> AnnData:
	adata.uns["iroot"] = np.flatnonzero(adata.obs[use_label]  == str(root_cluster))[root_point]
	stlearn.spatial.trajectory.pseudotime(data,eps=eps,radius=radius,use_sme = use_sme)

	return adata if copy else None

def PseudoSpaceTime(
    adata: AnnData,
    use_label: str = "louvain",
    local_cluster: int = 0,
    global_cluster: list = [0,1],
    weight: float = 0.5,
    copy: bool = False,

    ) -> AnnData:

	slearn.spatial.trajectory.local_level(data,
		use_label=use_label,
		cluster=local_cluster,
		w=weight)
	slearn.spatial.trajectory.global_level(data,
		use_label=use_label,
		list_cluster=global_cluster,
		w=weight)

	return adata if copy else None