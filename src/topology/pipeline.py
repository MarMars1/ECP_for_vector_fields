"""
Generic ECP processing pipeline.

The ECP parameter space is inferred automatically from the
number of channels of the selected filtration.

Examples
--------
2D vector field:
    (Nx, Ny, 2)
    -> ECP dims has length 2

2D divergence filtration:
    (Nx, Ny, 3)
    -> ECP dims has length 3

3D Lorenz vector field:
    (Nx, Ny, Nz, 3)
    -> ECP dims has length 3

3D curl filtration:
    (Nx, Ny, Nz, 6)
    -> ECP dims has length 6
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pyEulerCurves as pyecc

from pyEulerCurves.distance_utils import (
    discretize_contributions,
    difference_ECP,
)

from src.config import (
    DEFAULT_RANGE,
    DEFAULT_RESOLUTION,
    USE_DISCRETIZATION_2D,
    USE_DISCRETIZATION_3D,
    USE_DISCRETIZATION_4D,
)
from src.visualization.dendrogram import save_dendrogram
from src.visualization.ecp_plot import save_ecp_plot_2
from src.utils.io import save_distance_matrix

# ECP PARAMETER HELPERS
def get_filtration_dimensions(
    examples: list[np.ndarray],
) -> int:
    """
    Determine the number of ECP filtration dimensions.

    IMPORTANT
    ---------
    This is NOT the spatial dimension.
    The number of ECP dimensions is determined by the last axis of the already constructed filtration.

    Examples
    --------
    (21, 21, 2) -> 2
    (21, 21, 3) -> 3
    (21, 21, 4) -> 4
    (11, 11, 11, 6) -> 6
    """

    if not examples:
        raise ValueError(
            "Cannot determine ECP dimensions from "
            "an empty list of examples."
        )

    first = np.asarray(
        examples[0]
    )

    if first.ndim < 2:
        raise ValueError(
            "ECP input must contain spatial dimensions "
            "and a filtration/channel dimension."
        )

    n_dimensions = first.shape[-1]

    if n_dimensions < 1:
        raise ValueError(
            "The filtration must contain at least "
            "one channel."
        )

    # Every example must have the same number of filtration dimensions.
    for index, example in enumerate(examples):
        example = np.asarray(example)
        if example.ndim != first.ndim:
            raise ValueError(
                "All examples must have the same number "
                f"of dimensions. Example 0 has ndim={first.ndim}, "
                f"example {index} has ndim={example.ndim}."
            )

        if example.shape[-1] != n_dimensions:
            raise ValueError(
                "All examples must have the same number "
                "of filtration channels. "
                f"Example 0 has {n_dimensions}, "
                f"example {index} has "
                f"{example.shape[-1]}."
            )
    return n_dimensions


def build_ecp_dims(
    examples: list[np.ndarray],
) -> tuple[tuple[float, float], ...]:
    """
    Build ECP parameter-space dimensions.

    If DEFAULT_RANGE is provided, the same range is used for every filtration dimension.
    If DEFAULT_RANGE is None, the range is determined independently for every filtration channel from the global minimum and maximum across all examples.
    Automatically computed ranges follow:
        lower = floor(global_min) - 1
        upper = ceil(global_max) + 1

    Examples
    --------
    With:
        DEFAULT_RANGE = (-150.0, 150.0)
    a 4-channel filtration produces:
        (
            (-150.0, 150.0),
            (-150.0, 150.0),
            (-150.0, 150.0),
            (-150.0, 150.0),
        )

    With:
        DEFAULT_RANGE = None
    the ranges are computed independently for every channel.
    """

    n_filtration_dimensions = get_filtration_dimensions(examples)

    # Explicit default range
    if DEFAULT_RANGE is not None:
        if len(DEFAULT_RANGE) != 2:
            raise ValueError(
                "DEFAULT_RANGE must contain exactly "
                "two values: (minimum, maximum)."
            )

        minimum, maximum = DEFAULT_RANGE

        if minimum >= maximum:
            raise ValueError(
                "DEFAULT_RANGE minimum must be smaller "
                "than maximum."
            )

        return tuple(
            (
                float(minimum),
                float(maximum),
            )
            for _ in range(
                n_filtration_dimensions
            )
        )

    # Automatic range from data
    first = np.asarray(
        examples[0]
    )

    # All axes except the final filtration/channel axis.
    spatial_axes = tuple(
        range(first.ndim - 1)
    )

    # Global minimum for every filtration channel.
    all_min = np.min(
        [
            np.min(
                np.asarray(example),
                axis=spatial_axes,
            )
            for example in examples
        ],
        axis=0,
    )

    # Global maximum for every filtration channel.
    all_max = np.max(
        [
            np.max(
                np.asarray(example),
                axis=spatial_axes,
            )
            for example in examples
        ],
        axis=0,
    )
    print("all_max: ", all_max)

    return tuple(
        (
            float(np.floor(all_min[i]) - 1),
            float(np.ceil(all_max[i]) + 1),
        )
        for i in range(
            n_filtration_dimensions
        )
    )

def build_ecp_resolution(
    examples: list[np.ndarray],
) -> tuple[int, ...]:
    """
    Build ECP resolution automatically.
    The number of resolution values equals the number of filtration dimensions.
    """
    n_filtration_dimensions = get_filtration_dimensions(examples)

    return tuple(
        DEFAULT_RESOLUTION
        for _ in range(
            n_filtration_dimensions
        )
    )


def get_default_discretization(
    examples: list[np.ndarray],
) -> bool:
    """
    Select default discretization mode from the spatial dimension of the original vector field.

    """
    if not examples:
        raise ValueError(
            "Cannot determine discretization from "
            "an empty list of examples."
        )

    field = np.asarray(
        examples[0]
    )

    # Last axis = vector/filtration channels.
    # All preceding axes are spatial axes.
    spatial_dimension = field.ndim - 1
    if spatial_dimension == 2:
        return USE_DISCRETIZATION_2D
    if spatial_dimension == 3:
        return USE_DISCRETIZATION_3D
    if spatial_dimension == 4:
        return USE_DISCRETIZATION_4D
    # For dimensions not explicitly configured, use discretization by default.
    return True


# ECP CONTRIBUTIONS
def compute_ecp_contributions(
    examples: list[np.ndarray],
):
    """
    Compute ECP contributions for all examples.
    """

    if not examples:
        raise ValueError(
            "Cannot compute ECP contributions "
            "for an empty dataset."
        )

    transformer = pyecc.ECC_from_bitmap(
        multifiltration=True,
        workers=1,
    )

    contributions = []
    for example in examples:
        transformer.fit_transform(example)
        contributions.append(transformer.contributions_list)
    return contributions


# CONTRIBUTION PREPARATION
def prepare_contributions(
    contributions,
    dims,
    resolution,
    use_discretization: bool,
):
    """
    Optionally discretize ECP contributions.
    """
    if not use_discretization:
        return contributions

    return [
        discretize_contributions(
            contribution,
            dims=dims,
            resolution=resolution,
        )
        for contribution in contributions
    ]

# ECP DISTANCE
def compute_ecp_distance_matrix(
    contributions,
    dims,
    resolution,
    use_discretization: bool = True,
    verbose: bool = False,
):
    """
    Compute pairwise ECP distance matrix.
    """

    prepared = prepare_contributions(
        contributions=contributions,
        dims=dims,
        resolution=resolution,
        use_discretization=use_discretization,
    )

    n = len(prepared)

    distance_matrix = np.zeros(
        (n, n),
        dtype=np.float64,
    )

    for i in range(n):
        for j in range(i + 1, n):
            distance = difference_ECP(
                prepared[i],
                prepared[j],
                dims=dims,
                verbose=verbose,
            )
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix


# MAIN ECP PIPELINE
def run_ecp_pipeline(
    examples,
    labels,
    output_dir: Path,
    dendrogram_dir: Path,
    ecp_plot_dir: Path,
    output_name: str,
    method: str = "ward",
    use_discretization: bool | None = None,
    save_ecp_images: bool = True,
):
    """
    Run the complete ECP pipeline.

    Parameters
    ----------
    examples:
        Already constructed filtration fields.

        IMPORTANT:
        These are NOT necessarily raw vector fields.
        They can be VECTOR, DIV, CURL, ANGLE, EIGS, etc.

    labels:
        Names corresponding to examples.

    output_dir:
        Main experiment output directory.

    dendrogram_dir:
        Directory for dendrogram images.

    ecp_plot_dir:
        Directory for ECP plots.

    output_name:
        Base name used for generated files.

    method:
        Hierarchical clustering method.

    use_discretization:
        If None, automatically selected from the spatial dimension.

    save_ecp_images:
        Whether to save ECP visualizations.

    Returns
    -------
    np.ndarray
        Pairwise ECP distance matrix.
    """

    if not examples:
        raise ValueError(
            "ECP pipeline received no examples."
        )

    if len(examples) != len(labels):
        raise ValueError(
            "Number of examples must match "
            "number of labels. "
            f"Got {len(examples)} examples and "
            f"{len(labels)} labels."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Determine ECP dimensions automatically
    filtration_dimensions = (
        get_filtration_dimensions(
            examples
        )
    )
    dims = build_ecp_dims(examples)
    resolution = build_ecp_resolution(examples)

    # Determine discretization
    if use_discretization is None:
        use_discretization = (
            get_default_discretization(
                examples
            )
        )

    print(
        "[INFO] Spatial dimension = "
        f"{examples[0].ndim - 1}"
    )

    print(
        "[INFO] Filtration dimensions = "
        f"{filtration_dimensions}"
    )

    print(
        "[INFO] Example shape = "
        f"{np.asarray(examples[0]).shape}"
    )

    print(
        "[INFO] ECP dims = "
        f"{dims}"
    )

    print(
        "[INFO] ECP resolution = "
        f"{resolution}"
    )

    print(
        "[INFO] Discretization = "
        f"{use_discretization}"
    )

    # Compute contributions
    contributions = compute_ecp_contributions(examples)

    # Compute ECP distance matrix
    distance_matrix = compute_ecp_distance_matrix(
        contributions=contributions,
        dims=dims,
        resolution=resolution,
        use_discretization=use_discretization,
    )

    # Save dendrogram
    dendrogram_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dendrogram_path = (
        dendrogram_dir
        / f"{output_name}.jpg"
    )

    save_dendrogram(
        matrix=distance_matrix,
        labels=labels,
        output_file=dendrogram_path,
        method=method,
    )

    print(
        "[SAVE] Dendrogram -> "
        f"{dendrogram_path}"
    )

    # Save distance matrix
    matrix_path = (
        output_dir
        / f"{output_name}.npy"
    )

    save_distance_matrix(
        distance_matrix,
        matrix_path,
    )

    print(
        "[SAVE] Distance mat. -> "
        f"{matrix_path}"
    )

    # Save ECP images
    if save_ecp_images:
        ecp_plot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for label, contribution in zip(
            labels,
            contributions,
        ):

            output_file = (
                ecp_plot_dir
                / f"{label}_{output_name}.jpg"
            )

            save_ecp_plot_2(
                contributions=contribution,
                output_file=output_file,
            )
    print(
        "[ECP] Pipeline finished."
    )

    return distance_matrix
