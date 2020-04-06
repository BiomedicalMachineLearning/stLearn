import numpy as np
import pandas as pd
import scipy as sc
import scipy.spatial as spatial
from anndata import AnnData
from ...clustering.louvain import louvain
from ....preprocessing.graph import neighbors


# cluster spatial spots based on the proportion of known ligand-receptor co-expression among the neighbouring spots
def lr(
    adata: AnnData,
    use_data: str,
    threshold: float = 0,
    distance: int = 10,
) -> AnnData:

    data = adata.obsm[use_data]
    if not isinstance(data, pd.DataFrame):
        if sc.sparse.issparse(data):
            data = pd.DataFrame(data.toarray(), index=adata.obs_names, columns=adata.var_names)
        else:
            data = pd.DataFrame(data, index=adata.obs_names, columns=adata.var_names)

    lr_pairs = adata.uns['lr'].copy()
    lr_pairs += [item.split('_')[1]+'_'+item.split('_')[0] for item in lr_pairs]

    # get neighbour spots for each spot
    coor = adata.obs[['imagerow', 'imagecol']]
    point_tree = spatial.cKDTree(coor)
    neighbours = []
    for spot in adata.obs_names:
        n_index = point_tree.query_ball_point(np.array([adata.obs['imagerow'].loc[spot], adata.obs['imagecol'].loc[spot]]), distance)
        neighbours.append([item for item in data.index[n_index] if item != spot])
    
    # filter out those LR not existing in the dataset
    ligands = [item.split('_')[0] for item in lr_pairs]
    receptors = [item.split('_')[1] for item in lr_pairs]
    avail = [i for i, x in enumerate(ligands) if ligands[i] in data.columns and receptors[i] in data.columns]   
    spot_ligands = data.loc[:, [ligands[i] for i in avail]]
    spot_receptors = data.loc[:, [receptors[i] for i in avail]]
    print('Altogether ' + str(len(avail)) + ' valid L-R pairs')

    # count the co-expressed ligand-recptor pairs between neighbours
    def count_receptors(x):
        nbs = spot_receptors.loc[neighbours[data.index.tolist().index(x.name)], :]
        return (nbs > threshold).sum(axis=0) / nbs.shape[0]

    def count_ligands(x):
        nbs = spot_ligands.loc[neighbours[data.index.tolist().index(x.name)], :]
        return (nbs > threshold).sum(axis=0) / nbs.shape[0]

    nb_receptors = spot_receptors.apply(count_receptors, axis=1)   # proportion of neighbour spots which has receptor expression > threshold
    nb_ligands = spot_ligands.apply(count_ligands, axis=1)   # proportion of neighbour spots which has receptor expression > threshold
    # ligands on the spots
    st_lr_neighbour_ligands = pd.DataFrame((spot_ligands > threshold).values * nb_receptors.values, index=data.index, columns=[lr_pairs[i] for i in avail])
    # receptors on the spots
    st_lr_neighbour_receptors = pd.DataFrame((spot_receptors > threshold).values * nb_ligands.values, index=data.index, columns=[lr_pairs[i] for i in avail])
    adata.obsm['lr_neighbours'] = st_lr_neighbour_ligands + st_lr_neighbour_receptors
    print('L-R interactions with neighbours are counted and stored into adata[\'lr_neighbours\']')

    neighbors(adata,n_neighbors=25,use_rep='lr_neighbours')
    louvain(adata, key_added='lr_neighbours_louvain')
    
    # locate the highest Ligand-Receptor expressing cluster
    st_lr_cluster = []
    for n in range(max([int(i) for i in adata.obs['lr_neighbours_louvain']]) + 1):
        pair_idx = [i for i in range(len(adata.obs['lr_neighbours_louvain'])) if int(adata.obs['lr_neighbours_louvain'][i])==n]
        st_lr_cluster.append(adata.obsm['lr_neighbours'].iloc[pair_idx, :].sum().sum() / len(pair_idx))

    adata.uns['lr_neighbours_louvain_max'] = str(st_lr_cluster.index(max(st_lr_cluster)))
    print("Spatial distribution of LR co-expression is written to adata.obsm['lr_neighbours']")
    print("Result of LR-clustering is kept in adata.obs['lr_neighbours_louvain']")
    print("The largest expressed LR neighbouring cluster is: ", adata.uns['lr_neighbours_louvain_max'])

    return adata
