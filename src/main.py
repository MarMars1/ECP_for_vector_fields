
import numpy as np

from src.config import (
    GRID_MIN,
    GRID_MAX,
    GRID_POINTS,
    OUTPUT_DIR,
    SELECTED_EXAMPLES,
)

from src.systems import build_system_examples
from src.runner import run_full_experiment
from src.utils.experiment import create_experiment_directory
from src.utils.config_snapshot import save_config_snapshot

def build_grid():
    axes = [
        np.linspace(
            minimum,
            maximum,
            points,
        )
        for minimum, maximum, points in zip(
            GRID_MIN,
            GRID_MAX,
            GRID_POINTS,
        )
    ]

    return np.meshgrid(
        *axes,
    )


def main():
    random_seed = 42

    np.random.seed(
        random_seed
    )

    grids = build_grid()

    # Generate examples
    examples, names = build_system_examples(
        grids,
        selected_examples=SELECTED_EXAMPLES,
    )

    experiment_dir = create_experiment_directory(
        OUTPUT_DIR
    )

    save_config_snapshot(
        experiment_dir,
        random_seed,
    )


    # Run experiment
    run_full_experiment(
        examples=examples,
        names=names,
        output_dir=experiment_dir,
        grids=grids,
    )

    print()
    print("COMPUTATIONS FINISHED")
    print(f"Results saved in {experiment_dir}")


if __name__ == "__main__":
    main()
