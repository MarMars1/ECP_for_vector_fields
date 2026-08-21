"""
Global configuration.
"""

from pathlib import Path
from enum import Enum

class ExperimentMode(Enum):
    CLEAN = "clean"
    NOISY = "noisy"
    FULL = "full"

ENABLED_EXPERIMENT_MODES = [
    ExperimentMode.CLEAN,
]

class Filtration(Enum):
    VECTOR = "vector"
    DIV = "div"
    CURL = "curl"
    ANGLE = "angle"
    EIGS = "eigs"

ENABLED_FILTRATIONS = [

    Filtration.VECTOR,
    Filtration.DIV,
    Filtration.CURL,
    Filtration.ANGLE,
    Filtration.EIGS,
]


class NormalizationMode(Enum):

    NONE = "none"

    # whole tensor min-max
    MINMAX_GLOBAL = "minmax_global"

    # each channel separately
    MINMAX_CHANNEL = "minmax_channel"

    # z-score whole tensor
    ZSCORE_GLOBAL = "zscore_global"

    # z-score per channel
    ZSCORE_CHANNEL = "zscore_channel"

    # normalize vector magnitude
    UNIT_VECTOR = "unit_vector"


NORMALIZATION_MODE = (
    NormalizationMode.NONE
)

OUTPUT_DIR = Path("output_NONE_NORM")

X_MIN = -2.0
X_MAX = 2.0

Y_MIN = -2.0
Y_MAX = 2.0

GRID_POINTS = 101

NOISE_FRACTION = 0.01

DEFAULT_RANGE = (-150.0, 150.0)

DEFAULT_RESOLUTION = 10201

USE_DISCRETIZATION_2D = True
USE_DISCRETIZATION_3D = True
USE_DISCRETIZATION_4D = True

SAVE_ECP_IMAGES = True
SAVE_PHASE_PORTRAITS = True

def build_dims(n_dimensions: int):

    return tuple(
        DEFAULT_RANGE
        for _ in range(n_dimensions)
    )


def build_resolution(n_dimensions: int):

    return tuple(
        DEFAULT_RESOLUTION
        for _ in range(n_dimensions)
    )
