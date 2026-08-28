"""
Lorenz system with rho parameter sweep.

System:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LorenzSystem:
    sigma: float
    rho: float
    beta: float

RHO_VALUES = np.arange(24.0, 28.1, 0.2)

EXAMPLES: dict[str, LorenzSystem] = {
    f"rho={rho:.1f}": LorenzSystem(
        sigma=10.0,
        rho=float(rho),
        beta=8.0 / 3.0,
    )
    for rho in RHO_VALUES
}


def generate_vector_field(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    system: LorenzSystem,
    scale: float = 1.0,
) -> np.ndarray:

    u = scale * system.sigma * (Y - X)

    v = scale * (
        X * (system.rho - Z)
        - Y
    )

    w = scale * (
        X * Y
        - system.beta * Z
    )

    return np.stack(
        [u, v, w],
        axis=-1,
    )


def build_examples(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    scale: float = 1.0,
) -> tuple[list[np.ndarray], list[str]]:

    examples = [
        generate_vector_field(
            X,
            Y,
            Z,
            system,
            scale=scale,
        )
        for system in EXAMPLES.values()
    ]
    names = list(EXAMPLES.keys())
    return examples, names