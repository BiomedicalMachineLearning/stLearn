""" Performs LR analysis by grouping LR pairs which having hotspots across
    similar tissues.
"""

import scipy.spatial as spatial
from sklearn.cluster import DBSCAN
from anndata import AnnData
from tqdm import tqdm
import numpy as np
import pandas as pd

def get_hotspots(adata: AnnData, scores: np.ndarray, lrs: np.array,
                 eps: float, quantile=0.05):
    """Determines the hotspots for the inputted scores by progressively setting more stringent cutoffs & clustering in space, chooses point which maximises number of clusters.
        Parameters
        ----------
        adata: AnnData          The data object
        scores: np.ndarray      LR_pair*Spots containing the LR scores.
        lrs: np.array           The LR_pairs, in-line with the rows of scores.
        eps: float              The eps parameter used in DBScan to get the number of clusters.
        quantile: float         The quantiles to use for the cutoffs, if 0.05 then will take non-zero quantiles of 0.05, 0.1,..., 1 quantiles to cluster.

        Returns
        -------
        lr_hot_scores: np.ndarray, lr_cutoffs: np.array          First is the LR scores for just the hotspots, second is the cutoff used to get those LR_scores.
    """
    score_copy = scores.copy()
    quantiles = [quantile*i for i in range(int(1/quantile)+1)]
    coors = adata.obs[["imagerow", "imagecol"]].values

    # Values to return #
    lr_hot_scores = np.zeros(scores.shape)
    lr_cutoffs = []

    # Determining the cutoffs for hotspots #
    with tqdm(total=len(lrs),
              desc="Removing background lr scores...",
              bar_format="{l_bar}{bar}",
            ) as pbar:
        for i, lr_ in enumerate(lrs):
            lr_scores = score_copy[i,:]

            n_clusters = []
            cutoffs = np.quantile(lr_scores[lr_scores>0], quantiles)
            for cutoff in cutoffs:
                coor_ = coors[lr_scores>cutoff,:]
                clusters = DBSCAN(min_samples=2, eps=eps,
                                  metric='manhattan').fit_predict(coor_)
                n_clusters.append( len(np.unique(clusters)) )

            # Cutoff point where maximum number of clusters occurs #
            best_cutoff = cutoffs[np.argmax(n_clusters)]
            lr_cutoffs.append(best_cutoff)

            lr_scores[lr_scores < best_cutoff] = 0
            lr_hot_scores[i,:] = lr_scores

            pbar.update(1)

    return lr_hot_scores, np.array(cutoffs)





