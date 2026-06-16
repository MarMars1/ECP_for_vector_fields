from pathlib import Path


def save_experiment_summary(
    output_dir: Path,
    names,
):
    """
    Save summary of experiment.
    """

    with open(
        output_dir /
        "experiment_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"Number of systems: {len(names)}\n"
        )

        f.write("\nSystems:\n")

        for name in names:
            f.write(
                f" - {name}\n"
            )