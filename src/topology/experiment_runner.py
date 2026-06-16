"""
FULL experiment runner - exact reproduction of original pipeline:

✔ clean / noisy / full
✔ vector + div + curl + angle + eigs
✔ ECP + classical metrics
✔ dendrograms for every case
✔ full reproducibility
"""

from pathlib import Path
import numpy as np

from src.topology.pipeline import run_ecp_pipeline
from src.visualization.phase_portrait import save_phase_portraits
from src.metrics.distance import build_distance_matrix
from src.utils.io import save_npz
from src.config import NORMALIZATION_MODE, NOISE_FRACTION, ExperimentMode, ENABLED_FILTRATIONS, Filtration, SAVE_ECP_IMAGES, SAVE_PHASE_PORTRAITS
from src.analysis.field_features import compute_field
from src.analysis.normalization import (
    normalize_field,
)

from src.visualization.dendrogram import save_dendrogram
from src.utils.io import save_distance_matrix


def save_classical_results(
    classical_results: dict,
    labels_map: dict[str, list[str]],
    output_dir: Path,
    prefix: str,
):
    """
    Save dendrograms and distance matrices
    for classical metrics.

    Parameters
    ----------
    classical_results :
        Results returned by run_classical_metrics().

    labels_map :
        Example:

        {
            "clean": clean_names,
            "noisy": noisy_names,
            "full": full_names,
        }

    output_dir :
        Output directory.

    prefix :
        Filename prefix.
    """

    metric_dir_dendrograms = (
        output_dir /
        "dendrograms"
    )

    metric_dir_matrices = (
        output_dir /
        "distance_matrices"
    )

    
    metric_dir_dendrograms.mkdir(
        parents=True,
        exist_ok=True,
    )
    metric_dir_matrices.mkdir(
        parents=True,
        exist_ok=True,
    )

    for metric_name, datasets in classical_results.items():
        for dataset_name, matrix in datasets.items():
            if dataset_name not in labels_map:
                raise ValueError(
                    f"Missing labels for dataset "
                    f"'{dataset_name}'"
                )

            labels = labels_map[dataset_name]

            if matrix.shape[0] != len(labels):

                raise ValueError(
                    f"{metric_name}/{dataset_name}: "
                    f"matrix shape={matrix.shape}, "
                    f"labels={len(labels)}"
                )

            dendrogram_path = (
                metric_dir_dendrograms /
                f"{metric_name}_{dataset_name}.jpg"
            )

            save_dendrogram(
                matrix=matrix,
                labels=labels,
                output_file=dendrogram_path,
            )

            save_distance_matrix(
                matrix,
                metric_dir_matrices /
                f"{metric_name}_{dataset_name}.npy"
            )

            print(
                f"[SAVE] Classical dendrogram -> "
                f"{dendrogram_path}"
            )

# =========================================================
# NOISE
# =========================================================

def add_noise(examples, noise_fraction: float):
    """
    Add Gaussian noise proportional to std (same as original code).
    """
    noisy = []

    for ex in examples:
        std = ex.std(axis=(0, 1))
        noisy.append(
            ex + np.random.randn(*ex.shape) * std * noise_fraction
        )

    return noisy


# =========================================================
# FIELD TRANSFORMATIONS (THIS WAS MISSING IN YOUR RUNNER)
# =========================================================

def build_all_filtrations(examples):
    """
    Build all filtrations exactly like original pipeline.
    """

    vector = []
    div = []
    curl = []
    angle = []
    eigs = []

    for ex in examples:
        ex = normalize_field(ex, NORMALIZATION_MODE)

        field_div, field_curl, field_angle, field_eigs = compute_field(ex)
        
        vector.append(ex)
        div.append(field_div)
        curl.append(field_curl)
        angle.append(field_angle)
        eigs.append(field_eigs)


    result = {}

    if Filtration.VECTOR in ENABLED_FILTRATIONS:
        result["vector"] = vector

    if Filtration.DIV in ENABLED_FILTRATIONS:
        result["div"] = div

    if Filtration.CURL in ENABLED_FILTRATIONS:
        result["curl"] = curl

    if Filtration.ANGLE in ENABLED_FILTRATIONS:
        result["angle"] = angle

    if Filtration.EIGS in ENABLED_FILTRATIONS:
        result["eigs"] = eigs

    return result


# =========================================================
# ECP PIPELINE WRAPPER
# =========================================================

def run_all_ecp_modes(
    filtrations: dict,
    names,
    output_dir: Path,
    prefix: str,
):
    """
    Run ECP pipeline for ALL filtrations (this is what original code did manually).
    """

    results = {}

    for key, data in filtrations.items():

        print(f"\n========== ECP PIPELINE ==========")
        print(f"[ECP] Filtration: {key}")
        print(f"[ECP] Output dir: {output_dir / key}")

        dist = run_ecp_pipeline(
            examples=data,
            labels=names,
            output_dir=output_dir / "distance_matrices",
            #dendrogram_dir=output_dir / "dendrograms" / key,
            dendrogram_dir=output_dir / "dendrograms",
            #ecp_plot_dir=output_dir / "ecp_plots" / key,
            ecp_plot_dir=output_dir / "ecp_plots",
            output_name=f"{key}_{prefix}",
            save_ecp_images=SAVE_ECP_IMAGES,
        )

        results[key] = dist

    return results


# =========================================================
# CLASSICAL DISTANCES
# =========================================================

def run_classical_metrics(examples, noisy_examples, full_examples):

    metrics = {
        "L1": "cityblock",
        "L2": "euclidean",
        "Linf": "chebyshev",
    }

    results = {}

    for name, metric in metrics.items():

        results[name] = {
            "clean": build_distance_matrix(examples, metric),
            "noisy": build_distance_matrix(noisy_examples, metric),
            "full": build_distance_matrix(full_examples, metric),
        }

    return results


# =========================================================
# MAIN EXPERIMENT (FULL REPLICA)
# =========================================================

def run_full_experiment(
    X,
    Y,
    examples,
    names,
    output_dir: Path,
    modes,
):
    """
    Complete experiment runner.

    Modes
    -----
    CLEAN
        only clean dataset

    NOISY
        only noisy dataset

    FULL
        clean + noisy + merged
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_results = {}
    noisy_results = {}
    full_results = {}

    noisy_examples = []
    full_examples = []

    # =====================================================
    # CLEAN
    # =====================================================

    if (
        ExperimentMode.CLEAN in modes
        or ExperimentMode.FULL in modes
    ):

        print("Running CLEAN pipeline...")

        clean_filtrations = build_all_filtrations(
            examples
        )

        clean_results = run_all_ecp_modes(
            filtrations=clean_filtrations,
            names=names,
            output_dir=output_dir,
            prefix="clean",
        )
        if SAVE_PHASE_PORTRAITS:
            save_phase_portraits(
                X=X,
                Y=Y,
                examples=examples,
                names=names,
                output_dir=output_dir / "phase_portraits",
            )

    # =====================================================
    # NOISY
    # =====================================================
    if (
        ExperimentMode.NOISY in modes
        or ExperimentMode.FULL in modes
    ):

        print("Generating noisy dataset...")

        noisy_examples = add_noise(
            examples,
            NOISE_FRACTION,
        )

        noisy_names = [
            f"{name}_noisy"
            for name in names
        ]

        noisy_filtrations = build_all_filtrations(
            noisy_examples
        )

        noisy_results = run_all_ecp_modes(
            filtrations=noisy_filtrations,
            names=noisy_names,
            output_dir=output_dir,
            prefix="noisy",
        )

        if SAVE_PHASE_PORTRAITS:
            save_phase_portraits(
                X=X,
                Y=Y,
                examples=noisy_examples,
                names=noisy_names,
                output_dir=output_dir / "phase_portraits",
            )

    # =====================================================
    # FULL
    # =====================================================
    if ExperimentMode.FULL in modes:

        print("Running FULL dataset...")

        full_examples = (
            examples
            + noisy_examples
        )

        full_names = (
            names
            + [f"{n}_noisy" for n in names]
        )

        full_filtrations = build_all_filtrations(
            full_examples
        )

        full_results = run_all_ecp_modes(
            filtrations=full_filtrations,
            names=full_names,
            output_dir=output_dir,
            prefix="full",
        )

    # =====================================================
    # CLASSICAL METRICS
    # =====================================================

    classical = {}

    if ExperimentMode.FULL in modes:

        classical = run_classical_metrics(
            examples,
            noisy_examples,
            full_examples,
        )

    elif ExperimentMode.CLEAN in modes:

        classical = {
            metric: {
                "clean": build_distance_matrix(
                    examples,
                    metric_name,
                )
            }
            for metric, metric_name in {
                "L1": "cityblock",
                "L2": "euclidean",
                "Linf": "chebyshev",
            }.items()
        }

    elif ExperimentMode.NOISY in modes:

        classical = {
            metric: {
                "noisy": build_distance_matrix(
                    noisy_examples,
                    metric_name,
                )
            }
            for metric, metric_name in {
                "L1": "cityblock",
                "L2": "euclidean",
                "Linf": "chebyshev",
            }.items()
        }

    # ==========================================
    # SAVE CLASSICAL DENDROGRAMS
    # ==========================================

    if ExperimentMode.CLEAN in modes:

        save_classical_results(
            classical_results=classical,
            labels_map={
                "clean": names,
            },
            output_dir=output_dir,
            prefix="clean",
        )

    elif ExperimentMode.NOISY in modes:

        save_classical_results(
            classical_results=classical,
            labels_map={
                "noisy": [
                    f"{n}_noisy"
                    for n in names
                ]
            },
            output_dir=output_dir,
            prefix="noisy",
        )

    elif ExperimentMode.FULL in modes:

        save_classical_results(
            classical_results=classical,

            labels_map={

                "clean":
                    names,

                "noisy":
                    [
                        f"{n}_noisy"
                        for n in names
                    ],

                "full":
                    names +
                    [
                        f"{n}_noisy"
                        for n in names
                    ],
            },

            output_dir=output_dir,

            prefix="full",
        )

    # =====================================================
    # SAVE NPZ
    # =====================================================

    npz_data = {}

    for key, value in clean_results.items():
        npz_data[f"clean_{key}"] = value

    for key, value in noisy_results.items():
        npz_data[f"noisy_{key}"] = value

    for key, value in full_results.items():
        npz_data[f"full_{key}"] = value

    save_npz(
        output_dir / "all_results.npz",
        npz_data,
    )

    print("DONE ✔")

    return {
        "clean": clean_results,
        "noisy": noisy_results,
        "full": full_results,
        "classical": classical,
    }