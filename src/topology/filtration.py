"""
Filtration builders.
"""

from enum import Enum

import numpy as np

from src.analysis.field_features import (
    compute_field_features,
)


class FiltrationType(str, Enum):
    VECTOR = "vector"
    VECTOR_DIV = "vector_div"
    VECTOR_CURL = "vector_curl"
    VECTOR_ANGLE = "vector_angle"
    VECTOR_EIGS = "vector_eigs"


def build_filtration(
    vector_field: np.ndarray,
    filtration_type: FiltrationType,
) -> np.ndarray:
    """
    Convert vector field into selected filtration.

    Parameters
    ----------
    vector_field :
        H×W×2 field.

    filtration_type :
        Filtration definition.

    Returns
    -------
    ndarray
    """

    if filtration_type == FiltrationType.VECTOR:
        return vector_field

    div_field, curl_field, angle_field, eig_field = (
        compute_field_features(vector_field)
    )

    if filtration_type == FiltrationType.VECTOR_DIV:
        return div_field

    if filtration_type == FiltrationType.VECTOR_CURL:
        return curl_field

    if filtration_type == FiltrationType.VECTOR_ANGLE:
        return angle_field

    if filtration_type == FiltrationType.VECTOR_EIGS:
        return eig_field

    raise ValueError(
        f"Unsupported filtration: {filtration_type}"
    )