"""
Field feature extraction
"""

import numpy as np

def gradients_from_uv(u, v, X, Y, edge_order=1):
    du_dy, du_dx = np.gradient(
        u,
        Y[:, 0],
        X[0, :],
        edge_order=edge_order
    )

    dv_dy, dv_dx = np.gradient(
        v,
        Y[:, 0],
        X[0, :],
        edge_order=edge_order
    )

    return du_dx, du_dy, dv_dx, dv_dy


def eigs_2x2(a, b, c, d):
    trace = a + d
    disc = np.lib.scimath.sqrt(((a - d) * 0.5) ** 2 + b * c)

    lam1 = trace * 0.5 + disc
    lam2 = trace * 0.5 - disc

    return lam1, lam2


def compute_field(ex, X, Y):
    """
    ex: (H, W, 2) vector field
    """

    u = ex[:, :, 0]
    v = ex[:, :, 1]

    du_dx, du_dy, dv_dx, dv_dy = gradients_from_uv(
        u, v, X, Y
    )

    du_dx = np.nan_to_num(du_dx)
    du_dy = np.nan_to_num(du_dy)
    dv_dx = np.nan_to_num(dv_dx)
    dv_dy = np.nan_to_num(dv_dy)

    div = du_dx + dv_dy
    curl = dv_dx - du_dy
    angle = np.arctan2(v, u)

    lam1, lam2 = eigs_2x2(du_dx, du_dy, dv_dx, dv_dy)

    field_div = np.stack([u, v, div], axis=-1)
    field_curl = np.stack([u, v, curl], axis=-1)
    field_angle = np.stack([u, v, angle], axis=-1)

    field_eigs = np.stack(
        [np.real(lam1), np.imag(lam1),
         np.real(lam2), np.imag(lam2)],
        axis=-1
    )

    return field_div, field_curl, field_angle, field_eigs 
