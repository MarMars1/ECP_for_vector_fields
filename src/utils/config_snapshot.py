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
    Save exact experiment settings used for the experiment.
    """

    data = {
        "random_seed":
            random_seed,
        "grid_min":
            config.GRID_MIN,
        "grid_max":
            config.GRID_MAX,
        "grid_points":
            config.GRID_POINTS,
        "noise_fraction":
            config.NOISE_FRACTION,
        "enabled_modes":
            [
                mode.value if hasattr(mode, "value") else mode
                for mode in config.ENABLED_EXPERIMENT_MODES
            ],
        "enabled_filtrations":
            [
                filtration.value
                if hasattr(filtration, "value")
                else filtration
                for filtration in config.ENABLED_FILTRATIONS
            ],
        "normalization":
            (
                config.NORMALIZATION_MODE.value
                if hasattr(config.NORMALIZATION_MODE, "value")
                else config.NORMALIZATION_MODE
            ),
        "resolution":
            config.DEFAULT_RESOLUTION,
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
