""" Performs LR analysis by grouping LR pairs which having hotspots across
    similar tissues.
"""

from stlearn.pl import het_plot
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from anndata import AnnData
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


def get_hotspots(
    adata: AnnData,
    lr_scores: np.ndarray,
    lrs: np.array,
    eps: float,
    quantile=0.05,
    verbose=True,
    plot_diagnostics: bool = False,
    show_plot: bool = False,
):
    """Determines the hotspots for the inputted scores by progressively setting more stringent cutoffs & cluster in space, chooses point which maximises number of clusters.
    Parameters
    ----------
    adata: AnnData          The data object
    lr_scores: np.ndarray      LR_pair*Spots containing the LR scores.
    lrs: np.array           The LR_pairs, in-line with the rows of scores.
    eps: float              The eps parameter used in DBScan to get the number of clusters.
    quantile: float         The quantiles to use for the cutoffs, if 0.05 then will take non-zero quantiles of 0.05, 0.1,..., 1 quantiles to cluster.

    Returns
    -------
    lr_hot_scores: np.ndarray, lr_cutoffs: np.array          First is the LR scores for just the hotspots, second is the cutoff used to get those LR_scores.
    """
    coors = adata.obs[["imagerow", "imagecol"]].values
    lr_summary, lr_hot_scores = hotspot_core(
        lr_scores, lrs, coors, eps, quantile, plot_diagnostics, adata
    )

    if plot_diagnostics and show_plot:  # Showing the diagnostic plotting #
        plt.show()

    if verbose:
        print("Clustering LRs to help with ranking/interpretation...")
    # Clustering the LR pairs to obtain a set of clusters so to order within
    # each cluster
    clusterer = AgglomerativeClustering(
        affinity="euclidean", linkage="ward", distance_threshold=10, n_clusters=None
    )
    clusterer.fit(lr_hot_scores > 0)
    dist_cutoff = np.quantile(clusterer.distances_, 0.98)
    clusterer = AgglomerativeClustering(
        affinity="euclidean",
        linkage="ward",
        distance_threshold=dist_cutoff,
        n_clusters=None,
    )
    clusters = clusterer.fit_predict(lr_hot_scores > 0)
    cluster_set = np.unique(clusters)

    if verbose:
        print("Ranking LRs...")

    # Determining the ordering of the clusters so is useful to user #
    cluster_mean_spots = []
    for cluster in cluster_set:
        cluster_bool = clusters == cluster
        cluster_mean_spots.append(np.mean(lr_summary[cluster_bool, 2]))
    cluster_order = np.argsort(-np.array(cluster_mean_spots))

    # Determining order of lrs in cluster & also overall cluster scores #
    lr_order = []
    new_clusters = []
    cluster_scores = np.zeros((adata.shape[0], len(cluster_set)))
    for i, index in enumerate(cluster_order):
        cluster = cluster_set[index]
        cluster_indices = np.where(clusters == cluster)[0]
        lr_order_ = np.argsort(-lr_summary[cluster_indices, 2])
        lr_order.extend(cluster_indices[lr_order_])

        new_clusters += [i] * len(cluster_indices)

        cluster_scores[:, i] = lr_hot_scores[cluster_indices, :].mean(axis=0)

    if verbose:
        print("Saving results:")

    # Re-ordering the summary & the scores #
    lrs = lrs[lr_order]
    lr_summary = lr_summary[lr_order, :]
    lr_summary[:, 3] = new_clusters
    lr_summary = pd.DataFrame(
        lr_summary,
        index=lrs,
        columns=["spot_counts", "cutoff", "hotspot_counts", "lr_cluster"],
    )
    lr_scores = lr_scores[lr_order, :].transpose()
    lr_hot_scores = lr_hot_scores[lr_order, :].transpose()

    # Adding all this information to the AnnData #
    adata.uns["lr_summary"] = lr_summary
    adata.obsm["lr_scores"] = lr_scores
    adata.obsm["lr_hot_scores"] = lr_hot_scores
    adata.obsm["cluster_scores"] = cluster_scores

    if verbose:
        print(f"\tSummary values of lrs in adata.uns['lr_summary'].")
        print(
            f"\tMatrix of lr scores in same order as the summary in adata.obsm['lr_scores']."
        )
        print(f"\tMatrix of the hotspot scores in adata.obsm['lr_hot_scores'].")
        print(
            f"\tMatrix of the mean LR cluster scores in adata.obsm['cluster_scores']."
        )


def hotspot_core(
    lr_scores,
    lrs,
    coors,
    eps,
    quantile,
    plot_diagnostics=False,
    adata=None,
    verbose=True,
    max_score=False,
):
    """Made code for getting the hotspot information."""
    score_copy = lr_scores.copy()
    quantiles = [quantile * i for i in range(int(1 / quantile))]

    # Values to return #
    lr_hot_scores = np.zeros(score_copy.shape)
    # cols: spot_counts, cutoff, hotspot_counts, lr_cluster
    lr_summary = np.zeros((score_copy.shape[0], 4))

    ### Also creating grouping lr_pairs by quantiles to plot diagnostics ###
    if plot_diagnostics:
        lr_quantiles = [(i / 6) for i in range(1, 7)][::-1]
        lr_mean_scores = np.apply_along_axis(non_zero_mean, 1, score_copy)
        lr_quant_values = np.quantile(lr_mean_scores, lr_quantiles)
        quant_lrs = np.array(
            [lrs[lr_mean_scores == quant] for quant in lr_quant_values]
        )
        fig, axes = plt.subplots(6, 4, figsize=(20, 15))

    # Determining the cutoffs for hotspots #
    with tqdm(
        total=len(lrs),
        desc="Removing background lr scores...",
        bar_format="{l_bar}{bar}",
        disable=verbose == False,
    ) as pbar:
        for i, lr_ in enumerate(lrs):
            lr_score_ = score_copy[i, :]
            lr_summary[i, 0] = len(np.where(lr_score_ > 0)[0])

            cutoff_scores = []
            cutoffs = np.quantile(lr_score_[lr_score_ > 0], quantiles)
            for cutoff in cutoffs:
                spot_bool = lr_score_ >= cutoff
                if len(np.where(spot_bool)[0]) == 0:
                    cutoff_scores.append(0)
                    continue

                coor_ = coors[spot_bool, :]
                clusters = DBSCAN(
                    min_samples=2, eps=eps, metric="manhattan"
                ).fit_predict(coor_)
                score = len(np.unique(clusters)) * (np.mean(lr_score_[spot_bool])) ** 2
                cutoff_scores.append(score)

            # Cutoff point where maximum number of clusters occurs #
            best_cutoff = cutoffs[np.argmax(cutoff_scores)]
            if not max_score:
                lr_summary[i, 1] = best_cutoff
            else:
                lr_summary[i, 1] = cutoff_scores[np.argmax(cutoff_scores)]

            lr_score_[lr_score_ < best_cutoff] = 0
            lr_hot_scores[i, :] = lr_score_
            lr_summary[i, 2] = len(np.where(lr_score_ > 0)[0])

            # Adding the diagnostic plots #
            if plot_diagnostics and lr_ in quant_lrs and type(adata) != type(None):
                add_diagnostic_plots(
                    adata,
                    i,
                    lr_,
                    quant_lrs,
                    lr_quantiles,
                    lr_scores,
                    lr_hot_scores,
                    axes,
                    cutoffs,
                    cutoff_scores,
                    best_cutoff,
                )

            pbar.update(1)

    return lr_summary, lr_hot_scores


def non_zero_mean(vals):
    """Gives the non-zero mean of the values."""
    return vals[vals > 0].mean()


def add_diagnostic_plots(
    adata,
    i,
    lr_,
    quant_lrs,
    lr_quantiles,
    lr_scores,
    lr_hot_scores,
    axes,
    cutoffs,
    n_clusters,
    best_cutoff,
):
    """Adds diagnostic plots for the quantile LR pair to a figure to illustrate \
        how the cutoff is functioning.
    """
    q_i = np.where(quant_lrs == lr_)[0][0]

    # Scatter plot #
    axes[q_i][0].scatter(cutoffs, n_clusters)
    axes[q_i][0].set_title(f"n_clusts*mean_spot_score vs cutoff")
    axes[q_i][0].set_xlabel("cutoffs")
    axes[q_i][0].set_ylabel("n_clusts*mean_spot_score")

    # Distribution of scores with cutoff #
    scores_ = lr_scores[i, :]
    sb.distplot(
        scores_[scores_ > 0],
        ax=axes[q_i][1],
        hist=True,
        kde=False,
        color="red",
        norm_hist=True,
    )
    v_height = 0.5
    axes[q_i][1].vlines(best_cutoff, 0, v_height)
    axes[q_i][1].text(best_cutoff, v_height, str(round(best_cutoff, 2)))
    axes[q_i][1].set_title(f"Distrib {round(lr_quantiles[q_i], 2)}({lr_})")

    # Showing before & after filtering spots #
    adata.obsm["lr_scores"] = scores_
    het_plot(
        adata,
        use_het="lr_scores",
        ax=axes[q_i][2],
        show_color_bar=False,
    )
    axes[q_i][2].set_title("scores")

    adata.obsm["lr_scores"] = lr_hot_scores[i, :]
    het_plot(
        adata,
        use_het="lr_scores",
        ax=axes[q_i][3],
        show_color_bar=False,
    )
    axes[q_i][3].set_title("hotspot scores")
