"""
Application configuration.

All user-editable experiment settings are stored in experiment.toml.

This module only:
    1. loads experiment.toml,
    2. validates the configuration,
    3. exposes configuration values to the application.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import tomllib


# CONFIGURATION FILE
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_FILE = PROJECT_ROOT / "experiment.toml"


if not CONFIG_FILE.exists():
    raise FileNotFoundError(
        f"Configuration file not found: {CONFIG_FILE}"
    )


with open(CONFIG_FILE, "rb") as f:
    _CONFIG = tomllib.load(f)

class ExperimentMode(str, Enum):
    CLEAN = "clean"
    NOISY = "noisy"
    FULL = "full"

class NormalizationMode(str, Enum):
    NONE = "none"
    MINMAX_GLOBAL = "minmax_global"
    MINMAX_CHANNEL = "minmax_channel"
    ZSCORE_GLOBAL = "zscore_global"
    ZSCORE_CHANNEL = "zscore_channel"
    UNIT_VECTOR = "unit_vector"
    MAX_NORM_GLOBAL = "max_norm_global"


# SYSTEM
_system_config = _CONFIG["system"]

SYSTEM_NAME = str(
    _system_config["name"]
).upper()

SELECTED_EXAMPLES = list(
    _system_config.get(
        "examples",
        ["*"],
    )
)


# EXPERIMENT
_experiment_config = _CONFIG["experiment"]

ENABLED_EXPERIMENT_MODES = [
    ExperimentMode(mode)
    for mode in _experiment_config.get(
        "modes",
        ["clean"],
    )
]


ENABLED_FILTRATIONS = [
    str(filtration)
    for filtration in _experiment_config.get(
        "filtrations",
        ["vector"],
    )
]


NORMALIZATION_MODE = NormalizationMode(
    _experiment_config.get(
        "normalization",
        "none",
    )
)

# GRID
_grid_config = _CONFIG["grid"]


GRID_MIN = tuple(
    float(value)
    for value in _grid_config["minimum"]
)


GRID_MAX = tuple(
    float(value)
    for value in _grid_config["maximum"]
)


GRID_POINTS = tuple(
    int(value)
    for value in _grid_config["points"]
)

# GRID VALIDATION
if not (
    len(GRID_MIN)
    == len(GRID_MAX)
    == len(GRID_POINTS)
):
    raise ValueError(
        "grid.minimum, grid.maximum and grid.points "
        "must have the same number of dimensions."
    )


if any(
    points < 2
    for points in GRID_POINTS
):
    raise ValueError(
        "Every grid dimension must contain at least "
        "two points."
    )


if any(
    minimum >= maximum
    for minimum, maximum in zip(
        GRID_MIN,
        GRID_MAX,
    )
):
    raise ValueError(
        "Every grid minimum must be smaller than "
        "its corresponding maximum."
    )


# NOISE
_noise_config = _CONFIG.get(
    "noise",
    {},
)


NOISE_FRACTION = float(
    _noise_config.get(
        "fraction",
        0.01,
    )
)


# ECP / TOPOLOGY
_topology_config = _CONFIG.get(
    "topology",
    {},
)


_raw_default_range = _topology_config.get(
    "default_range",
    [-150.0, 150.0],
)

if _raw_default_range in (None, []):
    DEFAULT_RANGE = None

else:
    if len(_raw_default_range) != 2:
        raise ValueError(
            "topology.default_range must contain exactly "
            "two values: [minimum, maximum]."
        )

    DEFAULT_RANGE = (
        float(_raw_default_range[0]),
        float(_raw_default_range[1]),
    )

    if DEFAULT_RANGE[0] >= DEFAULT_RANGE[1]:
        raise ValueError(
            "topology.default_range minimum must be "
            "smaller than maximum."
        )


DEFAULT_RESOLUTION = int(
    _topology_config.get(
        "default_resolution",
        101,
    )
)

# DISCRETIZATION
USE_DISCRETIZATION = bool(
    _topology_config.get(
        "discretization",
        True,
    )
)

USE_DISCRETIZATION_2D = USE_DISCRETIZATION
USE_DISCRETIZATION_3D = USE_DISCRETIZATION
USE_DISCRETIZATION_4D = USE_DISCRETIZATION


# OUTPUT
_output_config = _CONFIG.get(
    "output",
    {},
)

_output_directory = _output_config.get(
    "directory",
    f"output_{SYSTEM_NAME}_{NORMALIZATION_MODE.value}",
)

OUTPUT_DIR = Path(
    _output_directory
)

SAVE_ECP_IMAGES = bool(
    _output_config.get(
        "save_ecp_images",
        True,
    )
)

SAVE_PHASE_PORTRAITS = bool(
    _output_config.get(
        "save_phase_portraits",
        True,
    )
)


# VALIDATION
_SUPPORTED_SYSTEMS = {
    "LINEAR",
    "HB",
    "FHN",
    "LORENZ",
}


if SYSTEM_NAME not in _SUPPORTED_SYSTEMS:
    raise ValueError(
        f"Unsupported system '{SYSTEM_NAME}'. "
        f"Available systems: "
        f"{sorted(_SUPPORTED_SYSTEMS)}"
    )


_SUPPORTED_FILTRATIONS = {
    "vector",
    "vector_div",
    "vector_curl",
    "vector_angle",
    "eigs",
}


invalid_filtrations = (
    set(ENABLED_FILTRATIONS)
    - _SUPPORTED_FILTRATIONS
)


if invalid_filtrations:
    raise ValueError(
        "Unsupported filtrations: "
        + ", ".join(
            sorted(invalid_filtrations)
        )
    )
