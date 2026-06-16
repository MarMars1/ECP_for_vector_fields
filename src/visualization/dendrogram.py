"""
Dendrogram visualization.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from src.metrics.clustering import linkage_matrix


def save_dendrogram(
    matrix,
    labels,
    output_file: Path,
    method: str = "ward"
):
    """
    Save dendrogram figure.
    """

    Z = linkage_matrix(
        matrix,
        method=method
    )

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