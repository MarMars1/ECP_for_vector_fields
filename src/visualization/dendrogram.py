"""
Dendrogram visualization.
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram


def save_dendrogram(
    matrix,
    labels,
    output_file: Path,
    method: str = "ward"
):
    """
    Save dendrogram figure.
    """

    # Identify type of matrix
    if matrix.shape[0] == matrix.shape[1]:
        if np.allclose(matrix, matrix.T, atol=1e-10) and np.all(np.diag(matrix) == 0):
            #print("Using distance matrix (squareform applied).")
            condensed_matrix = squareform(matrix)
        else:
            #print("Using feature matrix (pdist computed from square input).")
            condensed_matrix = pdist(matrix, metric='euclidean')
    else:
        #print("Using feature matrix (pdist computed).")
        condensed_matrix = pdist(matrix, metric='euclidean')

    # Hierarchical clustering
    Z = linkage(condensed_matrix, method=method)

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    dendrogram(
        Z,
        labels=labels,
        orientation="right",
        ax=ax
    )

    plt.tight_layout()

    fig.savefig(
        output_file,
        dpi=300
    )

    plt.close(fig)
