"""
Features derived from vector fields.
 
For a 2D vector field:

    field[..., 0] = u
    field[..., 1] = v

The following filtrations are defined:

    DIV:
        [u, v, du/dx + dv/dy]

    CURL:
        [u, v, dv/dx - du/dy]

    ANGLE:
        [u, v, atan2(v, u)]

    EIGS:
        [Re(lambda1), Im(lambda1),
         Re(lambda2), Im(lambda2)]

For 3D fields, spatial gradients are also supported.
"""

from __future__ import annotations

import numpy as np


def gradients_from_uv(
    u: np.ndarray,
    v: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    edge_order: int = 1,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute spatial derivatives.

    Returns
    -------
    du_dx
    du_dy
    dv_dx
    dv_dy
    """

    du_dy, du_dx = np.gradient(
        u,
        Y[:, 0],
        X[0, :],
        edge_order=edge_order,
    )

    dv_dy, dv_dx = np.gradient(
        v,
        Y[:, 0],
        X[0, :],
        edge_order=edge_order,
    )

    return (
        du_dx,
        du_dy,
        dv_dx,
        dv_dy,
    )


def eigs_2x2(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues of a 2x2 Jacobian analytically.

    """

    trace = a + d

    disc = np.lib.scimath.sqrt(
        ((a - d) * 0.5) ** 2
        + b * c
    )

    lam1 = trace * 0.5 + disc
    lam2 = trace * 0.5 - disc

    return lam1, lam2


def _clean_gradient(
    value: np.ndarray,
) -> np.ndarray:
    """
    Replace NaN/Inf.
    """

    return np.nan_to_num(
        value,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )


# 2D FIELD FEATURES
def compute_2d_field_features(
    field: np.ndarray,
    grids: tuple[np.ndarray, np.ndarray],
    eig_mode: str = "eigs",
    edge_order: int = 1,
) -> dict[str, np.ndarray]:
    """
    Compute 2D field filtrations.

    Parameters
    ----------
    field:
        Vector field with shape (Ny, Nx, 2).

    grids:
        Tuple (X, Y) generated using convention: X, Y = np.meshgrid(x, y)

    eig_mode:
        One of:
            "eigs"
            "real"
            "abs"
            "split"

        Default is "eigs".

    Returns
    -------
    dict
        Keys:
            vector
            div
            curl
            angle
            eigs
    """

    field = np.asarray(
        field,
        dtype=float,
    )

    if field.ndim != 3:
        raise ValueError(
            "2D vector field must have shape (Ny, Nx, 2)."
        )

    if field.shape[-1] != 2:
        raise ValueError(
            "2D vector field must have exactly 2 components."
        )

    X, Y = grids

    u = field[..., 0]
    v = field[..., 1]

    (
        du_dx,
        du_dy,
        dv_dx,
        dv_dy,
    ) = gradients_from_uv(
        u,
        v,
        X,
        Y,
        edge_order=edge_order,
    )

    du_dx = _clean_gradient(du_dx)
    du_dy = _clean_gradient(du_dy)
    dv_dx = _clean_gradient(dv_dx)
    dv_dy = _clean_gradient(dv_dy)

    # DIV / CURL / ANGLE

    div = du_dx + dv_dy
    curl = dv_dx - du_dy
    angle = np.arctan2(v, u)

    # EIGENVALUES
    lam1, lam2 = eigs_2x2(du_dx, du_dy, dv_dx, dv_dy)

    # EIG FILTRATION
    if eig_mode == "eigs":

        field_eigs = np.stack(
            [
                np.real(lam1),
                np.imag(lam1),
                np.real(lam2),
                np.imag(lam2),
            ],
            axis=-1,
        )

    elif eig_mode == "real":
        field_eigs = np.stack(
            [
                u,
                v,
                np.real(lam1),
                np.real(lam2),
            ],
            axis=-1,
        )

    elif eig_mode == "abs":
        field_eigs = np.stack(
            [
                u,
                v,
                np.abs(lam1),
                np.abs(lam2),
            ],
            axis=-1,
        )

    elif eig_mode == "split":
        field_eigs = np.stack(
            [
                u,
                v,
                np.real(lam1),
                np.imag(lam1),
                np.real(lam2),
                np.imag(lam2),
            ],
            axis=-1,
        )

    else:

        raise ValueError(
            f"Unknown eig_mode={eig_mode}"
        )

    # FILTRATIONS
    field_div = np.stack(
        [
            u,
            v,
            div,
        ],
        axis=-1,
    )

    field_curl = np.stack(
        [
            u,
            v,
            curl,
        ],
        axis=-1,
    )

    field_angle = np.stack(
        [
            u,
            v,
            angle,
        ],
        axis=-1,
    )

    return {
        "vector": field,
        "div": field_div,
        "curl": field_curl,
        "angle": field_angle,
        "eigs": field_eigs,
    }


# GENERIC SPATIAL GRADIENTS
def compute_spatial_gradients(
    field: np.ndarray,
    grids: tuple[np.ndarray, ...] | None = None,
    spacing: tuple[float, ...] | None = None,
    edge_order: int = 1,
) -> np.ndarray:

    field = np.asarray(
        field,
        dtype=float,
    )

    if field.ndim < 2:
        raise ValueError(
            "Vector field must have at least two dimensions."
        )

    n_dimensions = field.shape[-1]
    spatial_shape = field.shape[:-1]

    if len(spatial_shape) != n_dimensions:
        raise ValueError(
            "Number of vector components must equal "
            "the spatial dimension."
        )

    if grids is not None and spacing is not None:
        raise ValueError(
            "Provide either grids or spacing, not both."
        )

    # COORDINATES
    if grids is not None:
        if len(grids) != n_dimensions:
            raise ValueError(
                "Number of grids must match spatial dimension."
            )

        coordinates = []
        for axis, grid in enumerate(grids):
            grid = np.asarray(grid)
            if grid.shape != spatial_shape:
                raise ValueError(
                    f"Grid {axis} has shape {grid.shape}, "
                    f"expected {spatial_shape}."
                )
            selector = [0] * n_dimensions
            selector[axis] = slice(None)

            coordinates.append(
                grid[
                    tuple(selector)
                ]
            )
        coordinates = tuple(
            coordinates
        )

    elif spacing is not None:
        if len(spacing) != n_dimensions:
            raise ValueError(
                "Length of spacing must match "
                "spatial dimension."
            )
        coordinates = tuple(
            float(s)
            for s in spacing
        )
    else:
        coordinates = None

    # JACOBIAN
    jacobian = np.empty(
        spatial_shape + (
            n_dimensions,
            n_dimensions,
        ),
        dtype=float,
    )

    for component in range(
        n_dimensions
    ):
        component_field = field[
            ...,
            component
        ]
        if coordinates is None:
            gradients = np.gradient(
                component_field,
                edge_order=edge_order,
            )
        else:
            gradients = np.gradient(
                component_field,
                *coordinates,
                edge_order=edge_order,
            )

        for axis, gradient in enumerate(
            gradients
        ):
            jacobian[
                ...,
                component,
                axis,
            ] = _clean_gradient(
                gradient
            )
    return jacobian


# DIVERGENCE
def compute_divergence(
    jacobian: np.ndarray,
) -> np.ndarray:

    return np.trace(
        jacobian,
        axis1=-2,
        axis2=-1,
    )


# CURL
def compute_curl(
    jacobian: np.ndarray,
) -> np.ndarray:
    """
    Compute curl.
    2D:
        dv/dx - du/dy

    3D:
        standard vector curl.
    """

    n = jacobian.shape[-1]

    if n == 2:

        return (
            jacobian[..., 1, 0]
            - jacobian[..., 0, 1]
        )

    if n == 3:

        curl_x = (
            jacobian[..., 2, 1]
            - jacobian[..., 1, 2]
        )

        curl_y = (
            jacobian[..., 0, 2]
            - jacobian[..., 2, 0]
        )

        curl_z = (
            jacobian[..., 1, 0]
            - jacobian[..., 0, 1]
        )

        return np.stack(
            [
                curl_x,
                curl_y,
                curl_z,
            ],
            axis=-1,
        )

    raise ValueError(
        "Curl is implemented only for 2D and 3D."
    )


# ANGLE
def compute_angle(
    field: np.ndarray,
) -> np.ndarray:

    if field.shape[-1] != 2:
        raise ValueError(
            "ANGLE filtration is defined only for 2D."
        )

    return np.arctan2(
        field[..., 1],
        field[..., 0],
    )


# EIGENVALUES
def compute_eigenvalues(
    jacobian: np.ndarray,
) -> np.ndarray:
    """
    Compute eigenvalues.
    """

    n = jacobian.shape[-1]

    if n == 2:

        a = jacobian[..., 0, 0]
        b = jacobian[..., 0, 1]
        c = jacobian[..., 1, 0]
        d = jacobian[..., 1, 1]

        lam1, lam2 = eigs_2x2(
            a,
            b,
            c,
            d,
        )

        return np.stack(
            [
                lam1,
                lam2,
            ],
            axis=-1,
        )

    return np.linalg.eigvals(
        jacobian
    )


# MAIN FEATURE FUNCTION
def compute_field_features(
    field: np.ndarray,
    grids: tuple[np.ndarray, ...] | None = None,
    spacing: tuple[float, ...] | None = None,
    eig_mode: str = "eigs",
) -> dict[str, np.ndarray]:

    field = np.asarray(
        field,
        dtype=float,
    )

    if field.ndim < 2:
        raise ValueError(
            "Field must have at least one spatial "
            "dimension and one component dimension."
        )

    n_dimensions = field.shape[-1]

    # 2D
    if n_dimensions == 2:
        if grids is None:
            raise ValueError(
                "2D field features require grids=(X, Y)."
            )

        if spacing is not None:
            raise ValueError(
                "For 2D fields provide grids=(X, Y), "
                "not spacing."
            )

        return compute_2d_field_features(
            field,
            grids=grids,
            eig_mode=eig_mode,
            edge_order=1,
        )

    # 3D+
    jacobian = compute_spatial_gradients(
        field,
        grids=grids,
        spacing=spacing,
        edge_order=1,
    )

    features = {
        "vector": field,
    }

    # DIV
    divergence = compute_divergence(
        jacobian
    )

    features["div"] = np.concatenate(
        [
            field,
            divergence[..., None],
        ],
        axis=-1,
    )

    # CURL
    if n_dimensions == 3:

        curl = compute_curl(
            jacobian
        )

        features["curl"] = np.concatenate(
            [
                field,
                curl,
            ],
            axis=-1,
        )

    # EIGS
    eigenvalues = compute_eigenvalues(
        jacobian
    )

    features["eigs"] = np.concatenate(
        [
            np.real(eigenvalues),
            np.imag(eigenvalues),
        ],
        axis=-1,
    )

    return features
