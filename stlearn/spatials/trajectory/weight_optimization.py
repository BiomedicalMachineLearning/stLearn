import numpy as np
import pandas as pd
import networkx as nx
from .global_level import global_level
from .local_level import local_level
from .utils import lambda_dist, resistance_distance
from tqdm import tqdm


def weight_optimizing_global(
    adata,
    use_label=None,
    list_clusters=None,
    step=0.01,
    k=10,
    use_rep="X_pca",
    n_dims=40,
):
    # Screening PTS graph
    print("Screening PTS global graph...")
    Gs = []
    j = 0

    with tqdm(
        total=int(1 / step + 1),
        desc="Screening",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for i in range(0, int(1 / step + 1)):

            Gs.append(
                nx.to_scipy_sparse_array(
                    global_level(
                        adata,
                        use_label=use_label,
                        list_clusters=list_clusters,
                        use_rep=use_rep,
                        n_dims=n_dims,
                        w=round(j, 2),
                        return_graph=True,
                        verbose=False,
                    )
                )
            )

            j = j + step
            pbar.update(1)

    # Calculate the graph dissimilarity using Laplacian matrix
    print("Calculate the graph dissimilarity using Laplacian matrix...")
    result = []
    a1_list = []
    a2_list = []
    indx = []
    w = 0
    k = len(
        adata.obs[adata.obs[use_label].isin(list_clusters)][
            "sub_cluster_labels"
        ].unique()
    )
    with tqdm(
        total=int(1 / step - 1),
        desc="Calculating",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for i in range(1, int(1 / step)):
            w += step
            a1 = lambda_dist(Gs[i], Gs[0], k=k)
            a2 = lambda_dist(Gs[i], Gs[-1], k=k)
            a1_list.append(a1)
            a2_list.append(a2)
            indx.append(w)
            result.append(np.absolute(1 - a1 / a2))
            pbar.update(1)

    screening_result = pd.DataFrame(
        {"w": indx, "A1": a1_list, "A2": a2_list, "Dissmilarity_Score": result}
    )

    adata.uns["screening_result_global"] = screening_result

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    result = NormalizeData(result)

    try:
        optimized_ind = np.where(result == np.amin(result))[0][0]
        opt_w = round(indx[optimized_ind], 2)
        print("The optimized weighting is:", str(opt_w))
        return opt_w
    except:
        print("The optimized weighting is: 0.5")
        return 0.5


def weight_optimizing_local(adata, use_label=None, cluster=None, step=0.01):
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

            Gs.append(
                local_level(
                    adata,
                    use_label=use_label,
                    cluster=cluster,
                    w=round(j, 2),
                    verbose=False,
                    return_matrix=True,
                )
            )

            j = j + step
            pbar.update(1)

    # Calculate the graph dissimilarity using Laplacian matrix
    print("Calculate the graph dissimilarity using Resistance distance...")
    result = []
    a1_list = []
    a2_list = []
    indx = []
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
            indx.append(w)
            result.append(np.absolute(1 - a1 / a2))
            pbar.update(1)

    screening_result = pd.DataFrame(
        {"w": indx, "A1": a1_list, "A2": a2_list, "Dissmilarity_Score": result}
    )

    adata.uns["screening_result_local"] = screening_result

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    result = NormalizeData(result)

    optimized_ind = np.where(result == np.amin(result))[0][0]
    opt_w = round(indx[optimized_ind], 2)
    print("The optimized weighting is:", str(opt_w))

    return opt_w
