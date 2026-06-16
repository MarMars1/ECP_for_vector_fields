"""
Generic ECP processing pipeline.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyEulerCurves as pyecc

from pyEulerCurves.distance_utils import (
    discretize_contributions,
    difference_ECP,
)

from src.config import (
    USE_DISCRETIZATION_2D,
    USE_DISCRETIZATION_3D,
    USE_DISCRETIZATION_4D,
)

from src.config import (
    build_dims,
    build_resolution
)

from src.visualization.dendrogram import save_dendrogram
from src.visualization.ecp_plot import save_ecp_plot, save_ecp_plot_2
from src.utils.io import save_distance_matrix


def compute_ecp_contributions(examples):
    

    transformer = pyecc.ECC_from_bitmap(
        multifiltration=True,
        workers=1,
        #workers=-1,
    )

    contributions = []

    for ex in examples:

        transformer.fit_transform(ex)

        contributions.append(
            transformer.contributions_list
        )

    return contributions

def compute_ecp_distance_matrix(
    contributions,
    dims,
    resolution,
    use_discretization=True,
    verbose=False,
):
    """
    Compute pairwise ECP distance matrix.
    """

    prepared = prepare_contributions(
        contributions,
        dims,
        resolution,
        use_discretization,
    )

    n = len(prepared)

    D = np.zeros(
        (n, n),
        dtype=np.float64,
    )

    for i in range(n):

        for j in range(i + 1, n):

            d = difference_ECP(
                prepared[i],
                prepared[j],
                dims=dims,
                verbose=verbose,
            )

            D[i, j] = d
            D[j, i] = d

    return D

def run_ecp_pipeline(
    examples,
    labels,
    output_dir,
    dendrogram_dir,
    ecp_plot_dir,
    output_name,
    method="ward",
    use_discretization=None,
    save_ecp_images=True,
):

    if save_ecp_images:
        ecp_plot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    n_dimensions = examples[0].shape[-1]

    if use_discretization is None:
        if n_dimensions == 2:
            use_discretization = USE_DISCRETIZATION_2D
        elif n_dimensions == 3:
            use_discretization = USE_DISCRETIZATION_3D
        elif n_dimensions == 4:
            use_discretization = USE_DISCRETIZATION_4D
        else:
            use_discretization = True


    dims = build_dims(n_dimensions)

    resolution = build_resolution(n_dimensions)

    print(f"[INFO] ECP dimensions={n_dimensions}")

    print(f"[INFO] Dims={dims}")

    print(f"[INFO] Resolution={resolution}")

    contributions = compute_ecp_contributions(examples)

    D = compute_ecp_distance_matrix(
        contributions,
        dims=dims,
        resolution=resolution,
        use_discretization=use_discretization,
    )

    dendrogram_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dendrogram_path = (
        dendrogram_dir /
        f"{output_name}.jpg"
    )

    save_dendrogram(
        matrix=D,
        labels=labels,
        output_file=dendrogram_path,
        method=method,
    )

    print(f"[SAVE] Dendrogram -> {dendrogram_path}")

    matrix_path = (
        output_dir /
        f"{output_name}.npy"
    )
    
    save_distance_matrix(
        D,
        matrix_path,
    )

    print(
        f"[SAVE] Distance mat. -> {matrix_path}"
    )

    if save_ecp_images:

        ecp_plot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for label, contribution in zip(
            labels,
            contributions,
        ):

            #save_ecp_plot(
            #    contributions=contribution,
            #    dims=dims,
            #    output_file=ecp_plot_dir / f"{label}_{output_name}.jpg",
            #)

            save_ecp_plot_2(
                contributions=contribution,
                output_file=ecp_plot_dir / f"{label}_{output_name}.jpg",
            )

    return D

def prepare_contributions(
    contributions,
    dims,
    resolution,
    use_discretization: bool,
):
    """
    Optionally discretize contributions.
    """

    if not use_discretization:
        return contributions

    return [
        discretize_contributions(
            c,
            dims=dims,
            resolution=resolution,
        )
        for c in contributions
    ]