"""
Predefined linear autonomous systems used in experiments.

This module contains canonical examples of:
    - stable nodes
    - unstable nodes
    - saddles
    - stable focuses
    - unstable focuses
    - centers
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearSystem:
    """
    Linear autonomous system

        dx/dt = ax + by
        dy/dt = cx + dy
    """

    a: float
    b: float
    c: float
    d: float


EXAMPLES: dict[str, LinearSystem] = {

    # -------------------------
    # Stable nodes
    # -------------------------
    "stable node 1":
        LinearSystem(
            a=-1.5,
            b=0.0,
            c=0.0,
            d=-2.0
        ),

    "stable node 2":
        LinearSystem(
            a=-2.0,
            b=0.5,
            c=-0.5,
            d=-1.5
        ),

    # -------------------------
    # Unstable nodes
    # -------------------------
    "unstable node 1":
        LinearSystem(
            a=1.5,
            b=0.0,
            c=0.0,
            d=2.0
        ),

    "unstable node 2":
        LinearSystem(
            a=2.0,
            b=0.5,
            c=0.5,
            d=2.5
        ),

    # -------------------------
    # Saddles
    # -------------------------
    "saddle 1":
        LinearSystem(
            a=1.0,
            b=0.0,
            c=0.0,
            d=-1.0
        ),

    "saddle 2":
        LinearSystem(
            a=-1.0,
            b=0.0,
            c=0.0,
            d=1.0
        ),

    # -------------------------
    # Stable focuses
    # -------------------------
    "stable focus 1":
        LinearSystem(
            a=-1.0,
            b=-5.0,
            c=5.0,
            d=-1.0
        ),

    "stable focus 2":
        LinearSystem(
            a=-2.0,
            b=-4.0,
            c=4.0,
            d=-2.0
        ),

    # -------------------------
    # Unstable focuses
    # -------------------------
    "unstable focus 1":
        LinearSystem(
            a=1.0,
            b=5.0,
            c=-5.0,
            d=1.0
        ),

    "unstable focus 2":
        LinearSystem(
            a=2.0,
            b=4.0,
            c=-4.0,
            d=2.0
        ),

    # -------------------------
    # Centers
    # -------------------------
    "center 1":
        LinearSystem(
            a=0.0,
            b=-1.0,
            c=1.0,
            d=0.0
        ),

    "center 2":
        LinearSystem(
            a=0.0,
            b=1.0,
            c=-1.0,
            d=0.0
        ),
}


def generate_vector_field(
    X: np.ndarray,
    Y: np.ndarray,
    system: LinearSystem,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Generate vector field for a linear system.

    Parameters
    ----------
    X, Y :
        Meshgrid coordinates.

    system :
        Linear system parameters.

    scale :
        Multiplicative scaling factor.

    Returns
    -------
    np.ndarray
        Array of shape (H, W, 2).
    """

    u = scale * (
        system.a * X +
        system.b * Y
    )

    v = scale * (
        system.c * X +
        system.d * Y
    )

    return np.stack(
        [u, v],
        axis=-1
    )


def build_examples(
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Build all predefined vector fields.

    Returns
    -------
    examples :
        List of vector fields.

    names :
        List of example names.
    """

    examples = [
        generate_vector_field(
            X,
            Y,
            system
        )
        for system in EXAMPLES.values()
    ]

    names = list(EXAMPLES.keys())

    return examples, names
