import numpy as np
import pandas as pd
from anndata import AnnData
import scipy.spatial as spatial

def count(
    adata: AnnData,
    use_clustering: str = 'louvain',
    use_clusters: list = [],
    use_het: str = 'cci_het',
    distance: float = None,

) -> AnnData:
    """ Count the cell type densities
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count
    use_clustering:         The cell type results to use in counting  
    use_clusters:           Specify certain cluster(s) in counting
    use_het:                The stoarge place for result
    distance: int           Distance to determine the neighbours (default: nearest), distance=0 means within spot
    file: str               Result file from label transfer

    Returns
    -------
    adata: AnnData          With the counts of specified clusters in each window of the tissue stored as adata.uns['het']
    """

    # between spot
    if distance != 0:
        # automatically calculate distance if not given, won't overwrite distance=0 which is within-spot
        if not distance:
            # calculate default neighbour distance and number of windows
            scalefactors = next(iter(adata.uns['spatial'].values()))['scalefactors']
            distance = scalefactors['spot_diameter_fullres'] * scalefactors['tissue_' + adata.uns['spatial']['use_quality']+'_scalef'] * 2

        if len(use_clusters) == 0:
            use_clusters = list(set(adata.obs[use_clustering]))
        cluster = adata.obs[use_clustering]
        counts_ct = pd.DataFrame(0, adata.obs_names, ['CT'])

        # get neighbour spots for each spot
        coor = adata.obs[['imagerow', 'imagecol']]
        point_tree = spatial.cKDTree(coor)
        neighbours = []
        for spot in adata.obs_names:
            n_index = point_tree.query_ball_point(np.array([adata.obs['imagerow'].loc[spot], adata.obs['imagecol'].loc[spot]]), distance)
            neighbours = [item for item in adata.obs_names[n_index]]
            counts_ct.loc[spot] = len(set(cluster[neighbours]))
        counts_ct = counts_ct['CT']
    
    # within spot
    else:
        # count the cell types with prob > 0.2 in result of label transfer
        counts_ct = (adata.uns['label_transfer'] > 0.2).sum(axis=1)

    adata.uns[use_het] = counts_ct

    print("Counts for cluster (cell type) diversity stored into adata.uns[\'" + use_het + "\']")
    
    return adata
