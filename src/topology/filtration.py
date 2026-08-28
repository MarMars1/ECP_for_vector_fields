"""
Filtration builders for vector fields.
""" 

from enum import Enum
import numpy as np
from src.analysis.field_features import compute_field_features

class FiltrationType(str, Enum):
    VECTOR = "vector"
    VECTOR_DIV = "vector_div"
    VECTOR_CURL = "vector_curl"
    VECTOR_ANGLE = "vector_angle"
    EIGS = "eigs"


def get_available_filtrations(
    vector_field: np.ndarray,
) -> list[FiltrationType]:

    dimension = vector_field.shape[-1]

    available = [
        FiltrationType.VECTOR,
        FiltrationType.VECTOR_DIV,
        FiltrationType.EIGS,
    ]

    if dimension in (2, 3):
        available.append(
            FiltrationType.VECTOR_CURL
        )

    if dimension == 2:
        available.append(
            FiltrationType.VECTOR_ANGLE
        )

    return available


def build_filtration(
    vector_field: np.ndarray,
    filtration_type: FiltrationType,
    grids=None,
) -> np.ndarray:

    if filtration_type == FiltrationType.VECTOR:
        return vector_field

    features = compute_field_features(
        vector_field,
        grids=grids,
    )

    if filtration_type == FiltrationType.VECTOR_DIV:
        return features["div"]

    if filtration_type == FiltrationType.VECTOR_CURL:
        if "curl" not in features:
            raise ValueError(
                "VECTOR_CURL is not defined "
                "for this field dimension."
            )

        return features["curl"]

    if filtration_type == FiltrationType.VECTOR_ANGLE:
        if "angle" not in features:
            raise ValueError(
                "VECTOR_ANGLE is defined only "
                "for 2D vector fields."
            )
        return features["angle"]

    if filtration_type == FiltrationType.EIGS:
        return features["eigs"]

    raise ValueError(
        f"Unsupported filtration: {filtration_type}"
    )


def build_all_filtrations(
    examples: list[np.ndarray],
    enabled_filtrations=None,
    grids=None,
) -> dict[str, list[np.ndarray]]:
    if not examples:
        return {}

    if enabled_filtrations is None:
        filtration_types = get_available_filtrations(
            examples[0]
        )

    else:
        filtration_types = [
            (
                filtration
                if isinstance(
                    filtration,
                    FiltrationType,
                )
                else FiltrationType(
                    str(filtration)
                )
            )
            for filtration in enabled_filtrations
        ]

    result = {}

    for filtration_type in filtration_types:
        result[
            filtration_type.value
        ] = [
            build_filtration(
                example,
                filtration_type,
                grids=grids,
            )
            for example in examples
        ]

    return result
