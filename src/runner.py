"""
Main experiment runner.

Responsible for:
- clean / noisy / full experiments,
- selected filtrations,
- ECP computation,
- classical distance metrics,
- saving results.

The runner is dimension-independent.
"""

from pathlib import Path
import numpy as np
from src.analysis.normalization import normalize_field
from src.metrics.distance import build_distance_matrix
from src.topology.filtration import build_all_filtrations
from src.topology.pipeline import run_ecp_pipeline
from src.utils.io import save_distance_matrix, save_npz
from src.utils.noise import add_noise
from src.visualization.dendrogram import save_dendrogram
from src.visualization.phase_portrait import save_phase_portraits

from src.config import (
    SYSTEM_NAME,
    ENABLED_FILTRATIONS,
    ENABLED_EXPERIMENT_MODES,
    ExperimentMode,
    NORMALIZATION_MODE,
    NOISE_FRACTION,
    SAVE_ECP_IMAGES,
    SAVE_PHASE_PORTRAITS,
)

'''# NOISE
def add_noise(
    examples: list[np.ndarray],
    noise_fraction: float,
) -> list[np.ndarray]:
    """
    Add Gaussian noise to vector fields.

    Noise standard deviation is calculated independently
    for each vector component.

    Works for arbitrary spatial dimension:

        2D field -> (Nx, Ny, 2)
        3D field -> (Nx, Ny, Nz, 3)
        etc.

    Parameters
    ----------
    examples:
        List of vector fields.

    noise_fraction:
        Noise level relative to the component-wise standard
        deviation.

    Returns
    -------
    list[np.ndarray]
        Noisy vector fields.
    """

    noisy = []

    for field in examples:

        field = np.asarray(field)

        # All axes except the final component axis
        spatial_axes = tuple(
            range(field.ndim - 1)
        )

        std = np.std(
            field,
            axis=spatial_axes,
            keepdims=True,
        )

        noise = (
            np.random.randn(*field.shape)
            * std
            * noise_fraction
        )

        noisy.append(
            field + noise
        )

    return noisy'''


# FIELD PREPARATION
def prepare_examples(
    examples: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Normalize vector fields before filtration.

    Normalization is controlled by NORMALIZATION_MODE.
    """

    return [
        normalize_field(
            field,
            NORMALIZATION_MODE,
        )
        for field in examples
    ]


# FILTRATIONS
def build_selected_filtrations(
    examples: list[np.ndarray],
    grids=None,
) -> dict[str, list[np.ndarray]]:
    """
    Build only filtrations selected in config.py.

    Gradients are computed using the actual spatial grids
    when grids are provided.
    """

    prepared_examples = prepare_examples(
        examples
    )

    return build_all_filtrations(
        prepared_examples,
        ENABLED_FILTRATIONS,
        grids=grids,
    )


# ECP
def run_all_ecp_modes(
    filtrations: dict[str, list[np.ndarray]],
    names: list[str],
    output_dir: Path,
    prefix: str,
) -> dict[str, np.ndarray]:
    """
    Run ECP pipeline for every selected filtration.
    """

    results = {}

    for filtration_name, fields in filtrations.items():
        print("")
        print("ECP PIPELINE")

        distance_matrix = run_ecp_pipeline(
            examples=fields,
            labels=names,
            output_dir=output_dir / "distance_matrices",
            dendrogram_dir=output_dir / "dendrograms",
            ecp_plot_dir=output_dir / "ecp_plots",
            output_name=f"{filtration_name}_{prefix}",
            save_ecp_images=SAVE_ECP_IMAGES,
        )

        results[filtration_name] = distance_matrix

    return results


# CLASSICAL METRICS
CLASSICAL_METRICS = {
    "L1": "cityblock",
    "L2": "euclidean",
    "Linf": "chebyshev",
}

def run_classical_metrics(
    examples: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Compute classical distance matrices.
    """

    return {
        name: build_distance_matrix(
            examples,
            metric,
        )
        for name, metric in CLASSICAL_METRICS.items()
    }


def save_classical_results(
    classical_results: dict[str, np.ndarray],
    labels: list[str],
    output_dir: Path,
    prefix: str,
) -> None:
    """
    Save classical distance matrices and dendrograms.
    """

    dendrogram_dir = (
        output_dir / "dendrograms"
    )

    matrix_dir = (
        output_dir / "distance_matrices"
    )

    dendrogram_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for metric_name, matrix in classical_results.items():

        if matrix.shape[0] != len(labels):

            raise ValueError(
                f"{metric_name}: distance matrix has "
                f"{matrix.shape[0]} rows, but there are "
                f"{len(labels)} labels."
            )

        dendrogram_path = (
            dendrogram_dir
            / f"{metric_name}_{prefix}.jpg"
        )

        matrix_path = (
            matrix_dir
            / f"{metric_name}_{prefix}.npy"
        )

        save_dendrogram(
            matrix=matrix,
            labels=labels,
            output_file=dendrogram_path,
        )

        save_distance_matrix(
            matrix,
            matrix_path,
        )

        print(
            f"[SAVE] Classical dendrogram -> "
            f"{dendrogram_path}"
        )

        print(
            f"[SAVE] Classical matrix -> "
            f"{matrix_path}"
        )


# DATASET
def run_dataset(
    examples: list[np.ndarray],
    names: list[str],
    output_dir: Path,
    prefix: str,
    grids=None,
) -> dict:
    """
    Run the complete analysis for one dataset.

    Dataset can be:
        clean
        noisy
        full
    """

    print()
    print("")
    print(f"RUNNING DATASET: {prefix.upper()}")

    # ECP
    filtrations = build_selected_filtrations(
        examples,
        grids=grids,
    )

    ecp_results = run_all_ecp_modes(
        filtrations=filtrations,
        names=names,
        output_dir=output_dir,
        prefix=prefix,
    )

    # Classical metrics
    classical_results = run_classical_metrics(
        examples
    )

    save_classical_results(
        classical_results=classical_results,
        labels=names,
        output_dir=output_dir,
        prefix=prefix,
    )

    return {
        "ecp": ecp_results,
        "classical": classical_results,
    }


# MAIN EXPERIMENT
def run_full_experiment(
    examples: list[np.ndarray],
    names: list[str],
    output_dir: Path,
    modes=None,
    grids=None,
) -> dict:
    """
    Run the complete experiment.

    Parameters
    ----------
    examples:
        Vector fields generated by the selected dynamical system.

    names:
        Names corresponding to examples.

    output_dir:
        Experiment output directory.

    modes:
        Experiment modes:
            CLEAN
            NOISY
            FULL

        If None, ENABLED_EXPERIMENT_MODES from config.py
        are used.

    grids:
        Optional coordinate grids.

        Example 2D:
            (X, Y)

        Example 3D:
            (X, Y, Z)

        This is used only for visualization.
    """

    if modes is None:
        modes = ENABLED_EXPERIMENT_MODES

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not examples:
        raise ValueError(
            "No examples were provided."
        )

    if len(examples) != len(names):
        raise ValueError(
            "Number of examples does not match "
            "number of names."
        )

    print()
    print("")
    print(f"EXPERIMENT ", SYSTEM_NAME)

    print(
        f"Examples:       {len(examples)}"
    )

    print(
        f"Dimensions:     {examples[0].shape[-1]}"
    )

    print(
        "Filtrations:    "
        + ", ".join(
            str(filtration)
            for filtration in ENABLED_FILTRATIONS
        )
    )

    print(
        "Normalization:  "
        f"{NORMALIZATION_MODE.value}"
    )

    print(
        "Modes:          "
        + ", ".join(
            mode.value
            for mode in modes
        )
    )

    # CLEAN
    clean_examples = []
    noisy_examples = []
    full_examples = []

    clean_names = []
    noisy_names = []
    full_names = []

    clean_results = {}
    noisy_results = {}
    full_results = {}

    if (
        ExperimentMode.CLEAN in modes
        or ExperimentMode.FULL in modes
    ):

        clean_examples = examples
        clean_names = names

        clean_results = run_dataset(
            examples=clean_examples,
            names=clean_names,
            output_dir=output_dir,
            prefix="clean",
            grids=grids,
        )

        if (
            SAVE_PHASE_PORTRAITS
            and grids is not None
            and len(grids) == 2
        ):

            X, Y = grids

            save_phase_portraits(
                X=X,
                Y=Y,
                examples=clean_examples,
                names=clean_names,
                output_dir=(
                    output_dir
                    / "phase_portraits"
                ),
            )

    # NOISY
    if (
        ExperimentMode.NOISY in modes
        or ExperimentMode.FULL in modes
    ):

        print()
        print(
            "Generating noisy dataset..."
        )

        noisy_examples = add_noise(
            examples,
            NOISE_FRACTION,
        )

        noisy_names = [
            f"{name}_noisy"
            for name in names
        ]

        noisy_results = run_dataset(
            examples=noisy_examples,
            names=noisy_names,
            output_dir=output_dir,
            prefix="noisy",
            grids=grids,
        )

        if (
            SAVE_PHASE_PORTRAITS
            and grids is not None
            and len(grids) == 2
        ):

            X, Y = grids

            save_phase_portraits(
                X=X,
                Y=Y,
                examples=noisy_examples,
                names=noisy_names,
                output_dir=(
                    output_dir
                    / "phase_portraits"
                ),
            )

    # FULL
    if ExperimentMode.FULL in modes:

        print()
        print(
            "Running FULL dataset..."
        )

        full_examples = (
            clean_examples
            + noisy_examples
        )

        full_names = (
            clean_names
            + noisy_names
        )

        full_results = run_dataset(
            examples=full_examples,
            names=full_names,
            output_dir=output_dir,
            prefix="full",
            grids=grids,
        )
        
    # SAVE NPZ
    npz_data = {}

    for filtration_name, matrix in (
        clean_results
        .get("ecp", {})
        .items()
    ):

        npz_data[
            f"clean_{filtration_name}"
        ] = matrix

    for filtration_name, matrix in (
        noisy_results
        .get("ecp", {})
        .items()
    ):

        npz_data[
            f"noisy_{filtration_name}"
        ] = matrix

    for filtration_name, matrix in (
        full_results
        .get("ecp", {})
        .items()
    ):

        npz_data[
            f"full_{filtration_name}"
        ] = matrix

    if npz_data:

        save_npz(
            output_dir / "all_results.npz",
            npz_data,
        )

    return {
        "clean": clean_results,
        "noisy": noisy_results,
        "full": full_results,
    }