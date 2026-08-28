"""
Hopf bifurcation system with parameter sweep.

Model:
    dx/dt = beta * x - y - x * (x^2 + y^2)
    dy/dt = x + beta * y - y * (x^2 + y^2)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HopfSystem:
    beta: float

BETAS = np.round(np.arange(-1.0, 1.01, 0.1), 2)

EXAMPLES: dict[str, HopfSystem] = {
    f"beta={0.0 if beta == 0 else beta:.1f}": HopfSystem(
        beta=float(beta)
    )
    for beta in BETAS
}


def generate_vector_field(
    X: np.ndarray,
    Y: np.ndarray,
    system: HopfSystem,
    scale: float = 1.0,
) -> np.ndarray:

    r_squared = X**2 + Y**2

    u = (
        system.beta * X
        - Y
        - X * r_squared
    )

    v = (
        X
        + system.beta * Y
        - Y * r_squared
    )

    return scale * np.stack(
        [u, v],
        axis=-1,
    )


def build_examples(
    X: np.ndarray,
    Y: np.ndarray,
    scale: float = 1.0,
) -> tuple[list[np.ndarray], list[str]]:

    examples = [
        generate_vector_field(
            X,
            Y,
            system,
            scale=scale,
        )
        for system in EXAMPLES.values()
    ]
    names = list(EXAMPLES.keys())
    return examples, names