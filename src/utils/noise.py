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