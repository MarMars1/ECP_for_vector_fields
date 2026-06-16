"""
Phase portrait visualization.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_phase_portrait(
    X: np.ndarray,
    Y: np.ndarray,
    vector_field: np.ndarray,
    output_file: Path,
    density: float = 1.0,
):
    """
    Save single phase portrait.
    """

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.streamplot(
        X,
        Y,
        vector_field[:, :, 0],
        vector_field[:, :, 1],
        density=density,
        linewidth=0.7,
    )

    ax.contour(
        X,
        Y,
        vector_field[:, :, 0],
        levels=[0],
    )

    ax.contour(
        X,
        Y,
        vector_field[:, :, 1],
        levels=[0],
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_phase_portraits(
    X: np.ndarray,
    Y: np.ndarray,
    examples: list[np.ndarray],
    names: list[str],
    output_dir: Path,
):
    """
    Save all phase portraits.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for example, name in zip(examples, names):

        save_phase_portrait(
            X=X,
            Y=Y,
            vector_field=example,
            output_file=output_dir /
            f"{name}.jpg",
        )