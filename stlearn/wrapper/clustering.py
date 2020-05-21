from anndata import AnnData
import stlearn
import os

def SMEclust(
    adata: AnnData,
    out_path: str = "../tiling",
    crop_size: float = 40,
    use_data: str = "X_pca",
    radius: float = 50,
    adjust_method: str = "mean",
    n_neighbors: int = 25,
    use_label: str = "louvain",
    n_clusters: int = 10,
    random_state: int = 0,
    copy: bool = False,

    ) -> AnnData:

	if not os.path.isdir(out_path):
		os.mkdir(out_path)

	stlearn.pp.tiling(adata,out_path=out_path,crop_size = crop_size)
	stlearn.pp.extract_feature(adata)
	stlearn.spatial.morphology.adjust(adata,
		use_data=use_data,
		radius=radius,
		method=adjust_method)

	stlearn.pp.neighbors(adata,
		n_neighbors=n_neighbors,
		use_rep=use_data + '_morphology',
		random_state=random_state)

	if use_label == "louvain":
		stlearn.tl.clustering.louvain(adata,
			random_state=random_state)
	elif use_label == "kmeans":
		stlearn.tl.clustering.kmeans(adata,
			use_data = use_data,
			n_clusters=n_clusters)
	else:
		raise ValueError("Wrong method!")

	return adata if copy else None
