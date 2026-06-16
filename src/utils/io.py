from pathlib import Path

import numpy as np


def save_npz(
    output_file,
    data,
):
    """
    Save dictionary to NPZ.
    """

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        output_file,
        **data,
    )


def save_distance_matrix(
    matrix,
    output_file,
):
    """
    Save distance matrix as CSV.
    """

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savetxt(
        output_file,
        matrix,
        delimiter=",",
    )


def save_contributions(
    contributions,
    output_file,
):
    """
    Save ECP contributions.
    """

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_file,
        contributions,
        allow_pickle=True,
    )