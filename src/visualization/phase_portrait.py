"""
Phase portrait visualization for 2D vector fields.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def prepare_2d_grid(
    X: np.ndarray,
    Y: np.ndarray,
    vector_field: np.ndarray,
):

    X = np.asarray(X)
    Y = np.asarray(Y)
    vector_field = np.asarray(vector_field)

    # Validate dimensions
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(
            "Phase portraits require 2D grids. "
            f"Got X.ndim={X.ndim}, Y.ndim={Y.ndim}."
        )

    if X.shape != Y.shape:
        raise ValueError(
            "X and Y must have the same shape. "
            f"Got X={X.shape}, Y={Y.shape}."
        )

    if vector_field.ndim != 3:
        raise ValueError(
            "Vector field must have shape (N1, N2, 2). "
            f"Got {vector_field.shape}."
        )

    if vector_field.shape[:2] != X.shape:
        raise ValueError(
            "Vector field spatial dimensions must match X and Y. "
            f"Grid={X.shape}, field={vector_field.shape}."
        )

    if vector_field.shape[-1] != 2:
        raise ValueError(
            "Phase portrait requires a 2D vector field "
            "with exactly 2 components. "
            f"Got {vector_field.shape[-1]} components."
        )

    xy_grid = (
        np.allclose(X, X[0, :][None, :])
        and
        np.allclose(Y, Y[:, 0][:, None])
    )

    ij_grid = (
        np.allclose(X, X[:, 0][:, None])
        and
        np.allclose(Y, Y[0, :][None, :])
    )
    if xy_grid:
        x = X[0, :]
        y = Y[:, 0]
        u = vector_field[..., 0]
        v = vector_field[..., 1]
        return x, y, u, v


    if ij_grid:
        x = X[:, 0]
        y = Y[0, :]
        u = vector_field[..., 0].T
        v = vector_field[..., 1].T

        return x, y, u, v

    raise ValueError(
        "X and Y do not represent a regular Cartesian meshgrid. "
        "Expected a grid created by np.meshgrid(..., indexing='xy') "
        "or np.meshgrid(..., indexing='ij')."
    )


def save_phase_portrait(
    X: np.ndarray,
    Y: np.ndarray,
    vector_field: np.ndarray,
    output_file: Path,
    density: float = 1.0,
):
    """
    Save a single 2D phase portrait.
    """

    x, y, u, v = prepare_2d_grid(
        X,
        Y,
        vector_field,
    )

    fig, ax = plt.subplots(
        figsize=(6, 6),
    )

    ax.streamplot(
        x,
        y,
        u,
        v,
        density=density,
        linewidth=0.7,
    )

    # Nullclines
    ax.contour(
        x,
        y,
        u,
        levels=[0],
    )

    ax.contour(
        x,
        y,
        v,
        levels=[0],
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_phase_portraits(
    X: np.ndarray,
    Y: np.ndarray,
    examples: list[np.ndarray],
    names: list[str],
    output_dir: Path,
):

    if len(examples) != len(names):
        raise ValueError(
            "Number of examples must match number of names. "
            f"Got examples={len(examples)}, "
            f"names={len(names)}."
        )

    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(
            "Phase portraits are supported only for 2D grids."
        )

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for example, name in zip(
        examples,
        names,
    ):
        example = np.asarray(example)
        # Lorenz and other higher-dimensional systems
        # are skipped because this function creates 2D phase portraits.
        if (
            example.ndim != 3
            or example.shape[-1] != 2
        ):
            continue

        safe_name = str(name).replace(
            "/",
            "_",
        )

        save_phase_portrait(
            X=X,
            Y=Y,
            vector_field=example,
            output_file=(
                output_dir /
                f"{safe_name}.jpg"
            ),
        )
