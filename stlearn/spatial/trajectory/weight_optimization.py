import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData
from tqdm import tqdm

from .global_level import global_level
from .local_level import local_level
from .utils import lambda_dist, resistance_distance


def weight_optimizing_global(
    adata: AnnData,
    use_label: str = "louvain",
    list_clusters=None,
    step=0.01,
    k=10,
    use_rep="X_pca",
    n_dims=40,
):
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    # Determine effective k value based on available sub-clusters
    actual_k = k
    if use_label and list_clusters:
        if "sub_cluster_labels" not in adata.obs.columns:
            print(
                "Warning: 'sub_cluster_labels' column not found. Using provided "
                + "k value."
            )
        else:
            try:
                filtered_data = adata.obs[adata.obs[use_label].isin(list_clusters)]
                if len(filtered_data) == 0:
                    raise ValueError(
                        f"No cells found for clusters {list_clusters} "
                        + "in column '{use_label}'"
                    )

                # Minimum 1 cluster, use K or max available sub-clusters
                n_subclusters = len(filtered_data["sub_cluster_labels"].unique())
                actual_k = max(1, min(k, n_subclusters))

                if actual_k != k:
                    print(
                        f"Adjusted k from {k} to {actual_k} based on available "
                        + "sub-clusters ({n_subclusters})"
                    )

            except Exception as e:
                print(
                    f"Warning: Could not determine sub-cluster count: {e}. "
                    + "Using provided k value."
                )
                actual_k = k

    # Screening PTS graph
    print("Screening PTS global graph...")
    Gs = []
    j = 0
    total_iterations = int(1 / step + 1)
    with tqdm(
        total=total_iterations,
        desc="Screening",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for i in range(0, total_iterations):
            weight = round(i * step, 2)
            matrix = global_level(
                adata,
                use_label=use_label,
                list_clusters=list_clusters,
                use_rep=use_rep,
                n_dims=n_dims,
                w=weight,
                return_graph=True,
                verbose=False,
            )
            Gs.append(nx.to_scipy_sparse_array(matrix))
            j = j + step
            pbar.update(1)

    # Calculate the graph dissimilarity using Laplacian matrix
    print("Calculate the graph dissimilarity using Laplacian matrix...")
    result = []
    a1_list = []
    a2_list = []
    index = []
    w = 0
    with tqdm(
        total=int(1 / step - 1),
        desc="Calculating",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for i in range(1, int(1 / step)):
            w += step
            a1 = lambda_dist(Gs[i], Gs[0], k=actual_k)
            a2 = lambda_dist(Gs[i], Gs[-1], k=actual_k)
            a1_list.append(a1)
            a2_list.append(a2)
            index.append(w)
            result.append(np.absolute(1 - a1 / a2))
            pbar.update(1)

    screening_result = pd.DataFrame(
        {"w": index, "A1": a1_list, "A2": a2_list, "Dissmilarity_Score": result}
    )

    adata.uns["screening_result_global"] = screening_result

    normalised_result = normalize_data(result)

    try:
        optimized_ind = np.where(normalised_result == np.amin(normalised_result))[0][0]
        opt_w = round(index[optimized_ind], 2)
        print("The optimized weighting is:", str(opt_w))
        return opt_w
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        print("The optimized weighting is: 0.5")
        return 0.5


def weight_optimizing_local(
    adata: AnnData, use_label: str = "louvain", cluster=None, step=0.01
):
    # Screening PTS graph
    print("Screening PTS local graph...")
    Gs = []
    j = 0
    with tqdm(
        total=int(1 / step + 1),
        desc="Screening",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for i in range(0, int(1 / step + 1)):
            matrix = local_level(
                adata,
                use_label=use_label,
                cluster=cluster,
                w=round(j, 2),
                verbose=False,
                return_matrix=True,
            )
            Gs.append(matrix)
            j = j + step
            pbar.update(1)

    # Calculate the graph dissimilarity using Laplacian matrix
    print("Calculate the graph dissimilarity using Resistance distance...")
    result = []
    a1_list = []
    a2_list = []
    index = []
    w = 0

    with tqdm(
        total=int(1 / step - 1),
        desc="Calculating",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for i in range(1, int(1 / step)):
            w += step
            a1 = resistance_distance(Gs[i], Gs[0])
            a2 = resistance_distance(Gs[i], Gs[-1])
            a1_list.append(a1)
            a2_list.append(a2)
            index.append(w)
            result.append(np.absolute(1 - a1 / a2))
            pbar.update(1)

    screening_result = pd.DataFrame(
        {"w": index, "A1": a1_list, "A2": a2_list, "Dissmilarity_Score": result}
    )

    adata.uns["screening_result_local"] = screening_result

    normalised_result = normalize_data(result)

    optimized_ind = np.where(normalised_result == np.amin(normalised_result))[0][0]
    opt_w = round(index[optimized_ind], 2)
    print("The optimized weighting is:", str(opt_w))

    return opt_w


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
