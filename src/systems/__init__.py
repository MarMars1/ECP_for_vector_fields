"""
Dynamical systems and example builders.
"""

from src.systems.examples_linear import (
    LinearSystem,
    EXAMPLES as LINEAR_EXAMPLES,
    generate_vector_field as generate_linear_vector_field,
    build_examples as build_linear_examples,
)

from src.systems.examples_HB import (
    HopfSystem,
    EXAMPLES as HOPF_EXAMPLES,
    generate_vector_field as generate_hopf_vector_field,
    build_examples as build_hopf_examples,
)

from src.systems.examples_FHN import (
    FitzHughNagumoSystem,
    EXAMPLES as FHN_EXAMPLES,
    generate_vector_field as generate_fhn_vector_field,
    build_examples as build_fhn_examples,
)

from src.systems.examples_Lorenz import (
    LorenzSystem,
    EXAMPLES as LORENZ_EXAMPLES,
    generate_vector_field as generate_lorenz_vector_field,
    build_examples as build_lorenz_examples,
)


def build_system_examples(
    grids,
    selected_examples=None,
):
    """
    Build examples for the system selected in config.py.

    The system itself is selected through SYSTEM_NAME.
    """

    from src.config import SYSTEM_NAME

    if selected_examples is None:
        selected_examples = ["*"]

    if SYSTEM_NAME == "LINEAR":

        X, Y = grids

        examples, names = build_linear_examples(
            X,
            Y,
        )

    elif SYSTEM_NAME == "HB":

        X, Y = grids

        examples, names = build_hopf_examples(
            X,
            Y,
        )

    elif SYSTEM_NAME == "FHN":

        X, Y = grids

        examples, names = build_fhn_examples(
            X,
            Y,
        )

    elif SYSTEM_NAME == "LORENZ":

        X, Y, Z = grids

        examples, names = build_lorenz_examples(
            X,
            Y,
            Z,
        )

    else:

        raise ValueError(
            f"Unsupported system: {SYSTEM_NAME}"
        )

    # --------------------------------------------------------
    # Select requested examples
    # --------------------------------------------------------

    if "*" in selected_examples:
        return examples, names

    selected = set(
        selected_examples
    )

    filtered_examples = []
    filtered_names = []

    for example, name in zip(
        examples,
        names,
    ):

        if name in selected:

            filtered_examples.append(
                example
            )

            filtered_names.append(
                name
            )

    missing = (
        selected
        - set(filtered_names)
    )

    if missing:
        raise ValueError(
            "Requested examples were not found: "
            + ", ".join(sorted(missing))
        )

    return (
        filtered_examples,
        filtered_names,
    )


__all__ = [
    "LinearSystem",
    "HopfSystem",
    "FitzHughNagumoSystem",
    "LorenzSystem",

    "LINEAR_EXAMPLES",
    "HOPF_EXAMPLES",
    "FHN_EXAMPLES",
    "LORENZ_EXAMPLES",

    "build_linear_examples",
    "build_hopf_examples",
    "build_fhn_examples",
    "build_lorenz_examples",

    "build_system_examples",
]