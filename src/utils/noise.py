import numpy as np

def add_gaussian_noise(
    examples,
    noise_fraction=0.01
):
    noisy_examples = []

    for ex in examples:
        std = ex.std(axis=(0, 1))
        noisy = (
            ex
            + np.random.randn(*ex.shape)
            * std
            * noise_fraction
        )
        noisy_examples.append(noisy)
    return noisy_examples

def add_noise(
    examples: list[np.ndarray],
    noise_fraction: float,
) -> list[np.ndarray]:
    """
    Add Gaussian noise to vector fields.

    Noise standard deviation is calculated independently
    for each vector component.

    Works for arbitrary spatial dimension:

        2D field -> (Nx, Ny, 2)
        3D field -> (Nx, Ny, Nz, 3)
        etc.

    Parameters
    ----------
    examples:
        List of vector fields.

    noise_fraction:
        Noise level relative to the component-wise standard
        deviation.

    Returns
    -------
    list[np.ndarray]
        Noisy vector fields.
    """

    noisy = []

    for field in examples:

        field = np.asarray(field)

        # All axes except the final component axis
        spatial_axes = tuple(
            range(field.ndim - 1)
        )

        std = np.std(
            field,
            axis=spatial_axes,
            keepdims=True,
        )

        noise = (
            np.random.randn(*field.shape)
            * std
            * noise_fraction
        )

        noisy.append(
            field + noise
        )

    return noisy