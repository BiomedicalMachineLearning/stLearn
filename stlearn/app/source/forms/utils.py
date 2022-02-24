# -*- coding: utf-8 -*-
"""Helper utilities and decorators."""
from flask import flash
import matplotlib.pyplot as plt


def flash_errors(form, category="warning"):
    """Flash all errors for a form."""
    for field, errors in form.errors.items():
        for error in errors:
            flash(getattr(form, field).label.text + " - " + error + ", category")


def savePlot(plot_name, dpi=100, tight_layout=True, folder="temp_plots/"):
    """Deals with the current matplotlib.pyplot."""

    if tight_layout:
        plt.tight_layout()

    plt.savefig(folder + plot_name, dpi=dpi, format=plot_name.split(".")[-1])


def get_all_paths(adata):

    import networkx as nx

    G = nx.from_numpy_matrix(adata.uns["paga"]["connectivities_tree"].toarray())
    mapping = {int(k): v for k, v in zip(G.nodes, adata.obs.clusters.cat.categories)}
    G = nx.relabel_nodes(G, mapping)

    all_paths = []
    for source in G.nodes:
        for target in G.nodes:
            paths = nx.all_simple_paths(G, source=source, target=target)
            for path in paths:
                all_paths.append(path)

    import numpy as np

    all_paths = list(map(lambda x: " - ".join(np.array(x).astype(str)), all_paths))

    return all_paths
