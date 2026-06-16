from pathlib import Path

import numpy as np

from src.generators.examples import build_examples
from src.topology.experiment_runner import (
    run_full_experiment
)
from src.utils.experiment import (
    create_experiment_directory,
)

from src.utils.config_snapshot import (
    save_config_snapshot,
)

from src.config import (
    X_MIN,
    X_MAX,
    Y_MIN,
    Y_MAX,
    GRID_POINTS,
    OUTPUT_DIR,
    ENABLED_EXPERIMENT_MODES,
)

def main():

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    x = np.linspace(
        X_MIN,
        X_MAX,
        GRID_POINTS,
    )

    y = np.linspace(
        Y_MIN,
        Y_MAX,
        GRID_POINTS,
    )

    X, Y = np.meshgrid(
        x,
        y,
    )

    examples, names = build_examples(
        X,
        Y,
    )

    experiment_dir = create_experiment_directory(
        OUTPUT_DIR
    )

    save_config_snapshot(
        experiment_dir,
        RANDOM_SEED,
    )

    run_full_experiment(
        X=X,
        Y=Y,
        examples=examples,
        names=names,
        output_dir=experiment_dir,
        modes=ENABLED_EXPERIMENT_MODES,
    )

    print()
    print("================================")
    print("COMPUTATIONS FINISHED")
    print("Results saved in ./output")
    print("================================")


if __name__ == "__main__":
    main()