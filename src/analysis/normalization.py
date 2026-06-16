"""
Field normalization utilities.
"""

import numpy as np

from src.config import (
    NormalizationMode,
)


EPS = 1e-12


def normalize_field(
    field: np.ndarray,
    mode: NormalizationMode,
) -> np.ndarray:
    """
    Normalize field according to selected mode.

    Parameters
    ----------
    field :
        H×W×C tensor.

    mode :
        Normalization strategy.

    Returns
    -------
    ndarray
    """

    field = field.astype(
        np.float64,
        copy=False,
    )

    if mode == NormalizationMode.NONE:
        return field

    if mode == NormalizationMode.MINMAX_GLOBAL:

        mn = field.min()
        mx = field.max()

        return (
            field - mn
        ) / (
            mx - mn + EPS
        )

    if mode == NormalizationMode.MINMAX_CHANNEL:

        result = np.empty_like(
            field
        )

        for i in range(
            field.shape[-1]
        ):

            channel = field[:, :, i]

            mn = channel.min()
            mx = channel.max()

            result[:, :, i] = (
                channel - mn
            ) / (
                mx - mn + EPS
            )

        return result

    if mode == NormalizationMode.ZSCORE_GLOBAL:

        return (
            field - field.mean()
        ) / (
            field.std() + EPS
        )

    if mode == NormalizationMode.ZSCORE_CHANNEL:

        result = np.empty_like(
            field
        )

        for i in range(
            field.shape[-1]
        ):

            channel = field[:, :, i]

            result[:, :, i] = (
                channel
                - channel.mean()
            ) / (
                channel.std() + EPS
            )

        return result

    if mode == NormalizationMode.UNIT_VECTOR:

        if field.shape[-1] < 2:
            return field

        magnitude = np.linalg.norm(
            field[..., :2],
            axis=-1,
            keepdims=True,
        )

        magnitude = np.maximum(
            magnitude,
            EPS,
        )

        result = field.copy()

        result[..., :2] /= magnitude

        return result

    raise ValueError(
        f"Unknown normalization mode: {mode}"
    )