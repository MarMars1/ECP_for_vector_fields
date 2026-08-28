"""
Distance metrics. 
"""

import numpy as np

from scipy.spatial.distance import (
    pdist,
    squareform
)


def build_distance_matrix(
    examples: list[np.ndarray],
    metric: str = "euclidean"
) -> np.ndarray:
    """
    Build pairwise distance matrix.

    Parameters
    ----------
    examples :
        List of vector fields.

    metric :
        scipy metric name.

    Returns
    -------
    ndarray
        Distance matrix.
    """

    flattened = np.asarray([
        ex.reshape(-1)
        for ex in examples
    ])

    return squareform(
        pdist(flattened, metric=metric)
    )
