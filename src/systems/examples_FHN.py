"""
FitzHugh-Nagumo dynamical system examples.

Model:
    dx/dt = x - x^3 / 3 - y + R * I
    dy/dt = (x + a - b * y) / tau
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FitzHughNagumoSystem:
    a: float
    b: float
    R: float
    I: float
    tau: float

DEFAULT_A = 0.7
DEFAULT_R = 0.1
DEFAULT_TAU = 12.5

PARAMETER_PAIRS = np.array(
    [
        [0.8, -16],
        [0.8, 5],
        [2.0, 3.5],
        [2.0, 5.45],
        [0.8, 0],
        [0.5, 3],
        [1.5, 4],
        [2.0, 4.2],
        [0.8, 10],
    ],
    dtype=float,
)


EXAMPLES: dict[str, FitzHughNagumoSystem] = {
    f"b={b:.2f}, I={I:.2f}": FitzHughNagumoSystem(
        a=DEFAULT_A,
        b=float(b),
        R=DEFAULT_R,
        I=float(I),
        tau=DEFAULT_TAU,
    )
    for b, I in PARAMETER_PAIRS
}

def generate_vector_field(
    X: np.ndarray,
    Y: np.ndarray,
    system: FitzHughNagumoSystem,
    scale: float = 1.0,
) -> np.ndarray:

    dx_dt = (
        X
        - X**3 / 3.0
        - Y
        + system.R * system.I
    )

    dy_dt = (
        X
        + system.a
        - system.b * Y
    ) / system.tau

    return scale * np.stack(
        [dx_dt, dy_dt],
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