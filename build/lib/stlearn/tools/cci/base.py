import numpy as np
import pandas as pd
import scipy.spatial as spatial
from anndata import AnnData
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def lr_cluster(
    adata: AnnData,
    use_data: str = "filtered_counts",
    lr_pairs: str = "lr_means",
    num_clusters: int = 3,
) -> AnnData:
    
    data = adata.obsm[use_data]

    # build Ligand-Receptor spatial expression matrix
    st_lr_exp = pd.DataFrame(index=data.index)

    for pair in adata.uns[lr_pairs]:
        lr_a = pair.split('_')[0]
        lr_b = pair.split('_')[1]
        try:
            st_lr_exp[pair] = data[lr_a] + data[lr_b]
        except:
            pass

    st_lr_norm = st_lr_exp.div(st_lr_exp.sum(axis=1) + 0.001, axis=0)

    print("Altogether " + str(st_lr_exp.shape) + " ligand receptor pairs.")

    # run PCA and k-means
    if st_lr_norm.shape[1] > 20:
        lpca = StandardScaler().fit_transform(st_lr_norm)
        pca = PCA(n_components=20)
        st_lr_pca = pca.fit_transform(lpca)
    else:
        st_lr_pca = st_lr_norm
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(st_lr_pca)

    # locate the highest Ligand-Receptor expressing cluster
    st_lr_cluster = []
    for n in range(num_clusters):
        pair_idx = [i for i in range(len(kmeans.labels_)) if kmeans.labels_[i]==n]
    st_lr_cluster.append(st_lr_norm.iloc[pair_idx, :].sum().sum() / len(pair_idx))
    print("The largest expressed cluster is: ", str(st_lr_cluster.index(max(st_lr_cluster))))

    adata.obs['lr_cluster'] = kmeans.labels_



def lr_scan(
    adata: AnnData,
    use_data: str = "filtered_counts",
    threshold: float = 0,
    distance: int = 10,
    num_clusters: int = 3,
) -> AnnData:

    data = adata.obsm[use_data]

    # build Ligand-Receptor percentage in neighbours
    st_lr_neighbour = pd.DataFrame(0, index=data.index, columns=adata.uns["lr_means"])

    coor = adata.obs[['imagecol', 'imagerow']]
    point_tree = spatial.cKDTree(coor)

    ligands = [item.split('_')[0] for item in adata.uns["lr_means"]]
    receptors = [item.split('_')[1] for item in adata.uns["lr_means"]]
    neighbours = []
    for spot in data.index:
        n_index = point_tree.query_ball_point(np.array([adata.obs['imagecol'].loc[spot], adata.obs['imagerow'].loc[spot]]), distance)
        neighbours.append([item for item in data.index[n_index] if item != spot])
    spot_ligands = data.loc[:, ligands]
    spot_receptors = data.loc[:, receptors]
    def nb_count(x):
        nbs = spot_receptors.loc[neighbours[data.index.tolist().index(x.name)], :]
        return (nbs > threshold).sum(axis=0) / nbs.shape[0]
    nb_receptors = spot_receptors.apply(nb_count, axis=1)
    st_lr_neighbour = pd.DataFrame((spot_ligands > threshold).values * nb_receptors.values, index=data.index, columns=adata.uns["lr_means"])
    adata.obsm['lr_neighbours'] = st_lr_neighbour
 
    # run PCA and k-means
    lpca = StandardScaler().fit_transform(st_lr_neighbour)
    pca = PCA(n_components=20)
    st_lr_pca = pca.fit_transform(lpca)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(st_lr_pca)

    # locate the highest Ligand-Receptor expressing cluster
    st_lr_cluster = []
    for n in range(num_clusters):
        pair_idx = [i for i in range(len(kmeans.labels_)) if kmeans.labels_[i]==n]
    st_lr_cluster.append(st_lr_neighbour.iloc[pair_idx, :].sum().sum() / len(pair_idx))
    print("The largest expressed LR cluster is: ", str(st_lr_cluster.index(max(st_lr_cluster))))

    adata.obs['lr_neighbour_cluster'] = kmeans.labels_
