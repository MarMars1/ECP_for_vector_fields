"""
Experiment folder utilities.
"""

from datetime import datetime
from pathlib import Path


def create_experiment_directory(
    root_dir: Path,
) -> Path:
    """
    Create timestamped experiment directory.

    Example
    -------
    output/2026-05-25
    """

    timestamp = datetime.now().strftime(
        "%Y-%m-%d"
    )
    #timestamp = datetime.now().strftime(
    #    "%Y-%m-%d_%H-%M-%S"
    #)
    experiment_dir = root_dir / timestamp

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return experiment_dir


def save_experiment_summary(
    output_dir: Path,
    names,
):
    """
    Save summary of experiment.
    """

    with open(
        output_dir /
        "experiment_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"Number of systems: {len(names)}\n"
        )

        f.write("\nSystems:\n")

        for name in names:
            f.write(
                f" - {name}\n"
            )
