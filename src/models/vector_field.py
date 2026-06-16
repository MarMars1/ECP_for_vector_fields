"""
Vector field computations.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearSystem:
    """
    Linear autonomous system:

        dx/dt = ax + by
        dy/dt = cx + dy
    """

    a: float
    b: float
    c: float
    d: float


def compute_vector_field(
    X: np.ndarray,
    Y: np.ndarray,
    system: LinearSystem,
    scale: float = 1.0
) -> np.ndarray:
    """
    Generate vector field.

    Parameters
    ----------
    X, Y :
        Meshgrid coordinates.

    system :
        Linear system parameters.

    scale :
        Scaling coefficient.

    Returns
    -------
    ndarray
        H×W×2 vector field.
    """

    u = scale * (
        system.a * X +
        system.b * Y
    )

    v = scale * (
        system.c * X +
        system.d * Y
    )

    return np.stack([u, v], axis=-1)