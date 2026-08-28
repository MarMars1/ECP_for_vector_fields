"""
Field normalization utilities.

The last axis of a vector field contains vector components.

Examples
--------
2D:
    (Nx, Ny, 2)

3D:
    (Nx, Ny, Nz, 3)

The normalization methods work for arbitrary spatial dimension.
"""

from __future__ import annotations

import numpy as np
from src.config import NormalizationMode

EPS = 1e-12


def normalize_field(
    field: np.ndarray,
    mode: NormalizationMode,
) -> np.ndarray:
    """
    Normalize a vector field according to the selected mode.

    Parameters
    ----------
    field:
        Vector field.

        Examples:
            2D -> (Nx, Ny, 2)
            3D -> (Nx, Ny, Nz, 3)

    mode:
        Normalization strategy.

    Returns
    -------
    np.ndarray
        Normalized field with the same shape as the input.

    """

    field = np.asarray(
        field,
        dtype=np.float64,
    )

    if field.ndim < 2:
        raise ValueError(
            "Vector field must have at least two dimensions: "
            "(spatial dimensions..., components)."
        )

    # NONE
    if mode == NormalizationMode.NONE:
        return field.copy()

    # MIN-MAX GLOBAL
    # One minimum and maximum for the entire field.
    if mode == NormalizationMode.MINMAX_GLOBAL:
        mn = np.min(field)
        mx = np.max(field)

        return (
            field - mn
        ) / (
            mx - mn + EPS
        )

    # MIN-MAX CHANNEL
    # Each vector component is normalized independently.
    # All spatial dimensions participate in calculating the minimum and maximum.
    if mode == NormalizationMode.MINMAX_CHANNEL:
        spatial_axes = tuple(
            range(field.ndim - 1)
        )

        mn = np.min(
            field,
            axis=spatial_axes,
            keepdims=True,
        )

        mx = np.max(
            field,
            axis=spatial_axes,
            keepdims=True,
        )

        return (
            field - mn
        ) / (
            mx - mn + EPS
        )

    # Z-SCORE GLOBAL
    # One mean and standard deviation for the entire field.
    if mode == NormalizationMode.ZSCORE_GLOBAL:
        mean = np.mean(field)
        std = np.std(field)

        return (
            field - mean
        ) / (
            std + EPS
        )

    # Z-SCORE CHANNEL
    # Each vector component is standardized independently.
    # All spatial dimensions participate in calculating mean and standard deviation.
    if mode == NormalizationMode.ZSCORE_CHANNEL:
        spatial_axes = tuple(
            range(field.ndim - 1)
        )

        mean = np.mean(
            field,
            axis=spatial_axes,
            keepdims=True,
        )

        std = np.std(
            field,
            axis=spatial_axes,
            keepdims=True,
        )

        return (
            field - mean
        ) / (
            std + EPS
        )

    # MAX-NORM GLOBAL
    # Divide the complete vector field by the maximum
    # vector magnitude occurring anywhere in the spatial domain.

    if mode == NormalizationMode.MAX_NORM_GLOBAL:
        magnitude = np.linalg.norm(
            field,
            axis=-1,
        )

        max_norm = np.max(
            magnitude
        )

        if max_norm <= EPS:
            return field.copy()

        return field / max_norm
    

    # UNIT VECTOR
    # Normalize the vector at every spatial point.
    # Works for arbitrary dimension:
    # 2D: (u, v) -> (u, v) / sqrt(u² + v²)
    # 3D: (u, v, w) -> (u, v, w) / sqrt(u² + v² + w²)

    if mode == NormalizationMode.UNIT_VECTOR:
        magnitude = np.sqrt(
            np.sum(
                field * field,
                axis=-1,
                keepdims=True,
            )
        )

        # Avoid division by zero.
        magnitude = np.maximum(
            magnitude,
            EPS,
        )

        return field / magnitude
