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