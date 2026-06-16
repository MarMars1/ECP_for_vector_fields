"""
Save experiment configuration.
"""

import json
from pathlib import Path

from src import config


def save_config_snapshot(
    output_dir: Path,
    random_seed: int,
):
    """
    Save exact experiment settings.
    """

    data = {

        "random_seed": random_seed,

        "grid_points":
            config.GRID_POINTS,

        "x_min":
            config.X_MIN,

        "x_max":
            config.X_MAX,

        "y_min":
            config.Y_MIN,

        "y_max":
            config.Y_MAX,

        "noise_fraction":
            config.NOISE_FRACTION,

        "enabled_modes":
            [
                m.value
                for m in config.ENABLED_EXPERIMENT_MODES
            ],

        "enabled_filtrations":
            [
                f.value
                for f in config.ENABLED_FILTRATIONS
            ],

        "save_ecp_images":
            config.SAVE_ECP_IMAGES,

        "save_phase_portraits":
            config.SAVE_PHASE_PORTRAITS,

        "use_discretization_2d":
            config.USE_DISCRETIZATION_2D,

        "use_discretization_3d":
            config.USE_DISCRETIZATION_3D,

        "use_discretization_4d":
            config.USE_DISCRETIZATION_4D,
    }

    with open(
        output_dir / "config_snapshot.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
        )