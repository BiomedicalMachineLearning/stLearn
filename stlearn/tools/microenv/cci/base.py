import numpy as np
import pandas as pd
import scipy as sc
import scipy.spatial as spatial
from anndata import AnnData

# cluster spatial spots based on the proportion of known ligand-receptor co-expression among the neighbouring spots
def lr(
    adata: AnnData,
    use_data: str,
    use_lr: str = 'cci_lr',
    distance: float = None,
) -> AnnData:
    """ cluster spatial spots based on the proportion of known ligand-receptor co-expression among the neighbouring spots
    Parameters
    ----------
    adata: AnnData          The data object to scan
    use_data: str           Data to be used in L-R scanning
    distance: int           Distance to determine the neighbours (default: nearest), distance=0 means within spot
    res: float              resolution for clustering (default: 1)
    
    Returns
    -------
    adata: AnnData          The data object including the lr_scan results
    """
    
    threshold = 0   # threshold when counting for ligand-receptor coexpression
    
    # automatically calculate distance if not given, won't overwrite distance=0 which is within-spot
    if not distance and distance != 0:
        scalefactors = next(iter(adata.uns['spatial'].values()))['scalefactors']
        distance = scalefactors['spot_diameter_fullres'] * scalefactors['tissue_' + adata.uns['spatial']['use_quality']+'_scalef'] * 2

    df = adata.obsm[use_data]
    if not isinstance(df, pd.DataFrame):
        if sc.sparse.issparse(df):
            df = pd.DataFrame(df.toarray(), index=adata.obs_names, columns=adata.var_names)
        else:
            df = pd.DataFrame(df, index=adata.obs_names, columns=adata.var_names)

    # expand the LR pairs list by swapping ligand-receptor positions
    lr_pairs = adata.uns['lr'].copy()
    lr_pairs += [item.split('_')[1]+'_'+item.split('_')[0] for item in lr_pairs]

    # get neighbour spots for each spot
    coor = adata.obs[['imagerow', 'imagecol']]
    point_tree = spatial.cKDTree(coor)
    neighbours = []
    for spot in adata.obs_names:
        if distance == 0:
            neighbours.append([spot])
        else:
            n_index = point_tree.query_ball_point(np.array([adata.obs['imagerow'].loc[spot], adata.obs['imagecol'].loc[spot]]), distance)
            neighbours.append([item for item in df.index[n_index]])
    
    # filter out those LR pairs not expressing in the dataset
    ligands = [item.split('_')[0] for item in lr_pairs]
    receptors = [item.split('_')[1] for item in lr_pairs]
    avail = [i for i, x in enumerate(ligands) if ligands[i] in df.columns and receptors[i] in df.columns]   
    spot_ligands = df.loc[:, [ligands[i] for i in avail]]
    spot_receptors = df.loc[:, [receptors[i] for i in avail]]
    print('Altogether ' + str(len(avail)) + ' valid L-R pairs')

    # function to count co-expressed ligand-recptor pairs between neighbours for each spot
    def count_receptors(x):
        nbs = spot_receptors.loc[neighbours[df.index.tolist().index(x.name)], :]
        if nbs.shape[0] > 0:
            return (nbs > threshold).sum() / nbs.shape[0]
        else:
            return 0
    
    # proportion of neighbour spots which has receptor expression > threshold
    nb_receptors = spot_receptors.apply(count_receptors, axis=1)

    # keep value of nb_receptors only if ligand is also expressed on the spots
    spot_lr = pd.DataFrame((spot_ligands > threshold).values * nb_receptors.values, index=df.index, columns=[lr_pairs[i] for i in avail]).sum(axis=1)
    adata.uns[use_lr] = spot_lr

    print("L-R interactions with neighbours are counted and stored into adata\.uns[\'" + use_lr + "\']")

    return adata
