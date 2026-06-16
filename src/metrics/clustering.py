"""
Hierarchical clustering utilities.
"""

import numpy as np

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def linkage_matrix(
    matrix: np.ndarray,
    method: str = "ward"
):
    """
    Create linkage matrix.
    """

    if (
        matrix.shape[0] == matrix.shape[1]
        and np.allclose(matrix, matrix.T)
    ):
        condensed = squareform(matrix)
    else:
        condensed = pdist(matrix)

    return linkage(
        condensed,
        method=method
    )