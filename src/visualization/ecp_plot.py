"""
ECP visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import cm
from matplotlib import colors


def ec_at_bifiltration(
    contributions,
    f1,
    f2,
):
    """
    Euler characteristic at bifiltration point.
    """

    return sum(
        c[1]
        for c in contributions
        if c[0][0] <= f1
        and c[0][1] <= f2
    )

def plot_ecp(
    contributions,
    dims,
    ax=None,
    colorbar=False,
    **kwargs,
):
    """
    Plot Euler Characteristic Profile.
    """

    if ax is None:
        ax = plt.gca()

    f1min, f1max, f2min, f2max = dims

    f1_values = (
        [f1min]
        + sorted(
            {
                c[0][0]
                for c in contributions
            }
        )
        + [f1max]
    )

    f2_values = (
        [f2min]
        + sorted(
            {
                c[0][1]
                for c in contributions
            }
        )
        + [f2max]
    )

    Z = np.zeros(
        (
            len(f2_values) - 1,
            len(f1_values) - 1,
        )
    )

    for i, f1 in enumerate(f1_values[:-1]):
        for j, f2 in enumerate(f2_values[:-1]):

            Z[j, i] = ec_at_bifiltration(
                contributions,
                f1,
                f2,
            )

    im = ax.pcolormesh(
        f1_values,
        f2_values,
        Z,
        **kwargs,
    )

    if colorbar:
        plt.colorbar(
            im,
            ax=ax,
        )

    return ax



#from src.topology.ecp import plot_ECP
def save_ecp_plot(
    contributions,
    dims,
    output_file: Path,
):

    #if len(dims) != 2:
    #    return

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    flat_dims = (
        dims[0][0],
        dims[0][1],
        dims[1][0],
        dims[1][1],
    )

    plot_ecp(
        contributions=contributions,
        dims=flat_dims,
        ax=ax,
        cmap="viridis",
    )

    plt.tight_layout()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_file,
        dpi=300,
    )

    #save_ecp_plot_2(
    #    contributions=contributions,
    #    output_file=output_file,
    #    xlim = (-5, 2),
    #    ylim = (-5, 2),
    #)
        

    plt.close(fig)

    print(f"[SAVE] ECP plot      -> {output_file}")


def save_ecp_plot_2(
    contributions,
    output_file: str | Path,
    levels: tuple[int, int] = (-10, 11),
):
    """
    Save Euler Characteristic Profile plot.

    This function reproduces the original plotting style
    used in the legacy implementation.

    Parameters
    ----------
    contributions
        ECP contribution list.

    output_file
        Output image path.
    """

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(7, 6),
    )

    levels = np.arange(-5, 6)

    cmap = cm.get_cmap(
        "gist_rainbow",
        len(levels),
    )

    boundaries = np.arange(
        levels[0] - 0.5,
        levels[-1] + 1.5,
        1,
    )

    norm = colors.BoundaryNorm(
        boundaries=boundaries,
        ncolors=len(levels),
    )

    plot_ecp(
        contributions=contributions,
        dims=(
            -1000000,
            1000000,
            -1000000,
            1000000,
        ),
        ax=ax,
        norm=norm,
        cmap=cmap,
        colorbar=False,
    )

    sm = cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    cbar = plt.colorbar(
        sm,
        ax=ax,
    )

    cbar.set_ticks(levels)

    cbar.set_ticklabels(
        [str(level) for level in levels]
    )

    xlim = (min(c[0][0] for c in contributions)-2, max(c[0][0] for c in contributions)+2)
    ylim = (min(c[0][1] for c in contributions)-2, max(c[0][1] for c in contributions)+2)
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    ax.set_xticks(
        range(
            int(xlim[0]),
            int(xlim[1]) + 1,
            1,
        )
    )

    ax.set_yticks(
        range(
            int(ylim[0]),
            int(ylim[1]) + 1,
            1,
        )
    )
    

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
    )

    fig.savefig(
        output_file,
        dpi=300,
        format="jpg",
        bbox_inches="tight",
    )

    plt.close(fig)