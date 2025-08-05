"""Plotting utilities."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors


def _plot_velocity_series(
    time_s: list[float],
    command_body: list[float],
    actual_body: list[float],
    error_body: list[float],
    axis: str,
    outdir: Path,
) -> None:
    """Save PNG with two stacked plots."""
    fig, (ax_top, ax_err) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7, 4),
        height_ratios=[3, 1],
    )

    ax_top.plot(time_s, command_body, label=f"command v{axis}")
    ax_top.plot(time_s, actual_body, label=f"actual  v{axis}")
    ax_top.set_ylabel(f"v{axis}  [m·s⁻¹]")
    ax_top.legend(loc="upper right")

    ax_err.plot(time_s, error_body, label="error", linewidth=1)
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("err")
    ax_err.legend(loc="upper right")

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"velocity_{axis}.png", dpi=150)
    plt.close(fig)




def _plot_xy_trajectory(
    ref_x: list[float],
    ref_y: list[float],
    act_x: list[float],
    act_y: list[float],
    outdir: Path,
) -> None:
    """
    Save a top-down plot comparing reference vs. actual XY trajectories.
    Reference path: green → blue, actual path: yellow → red (early → late).
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    def add_gradient_line(
        x: list[float],
        y: list[float],
        start_col: str,
        end_col: str,
        label: str,
        lw: float = 2.0,
    ) -> None:
        """Plot a LineCollection that fades from *start_col* → *end_col*."""
        pts = np.column_stack([x, y])
        segs = np.concatenate([pts[:-1, None], pts[1:, None]], axis=1)

        cmap = colors.LinearSegmentedColormap.from_list(
            f"{label}_cmap",
            [start_col, end_col],
        )

        lc = LineCollection(
            segs,
            cmap=cmap,
            norm=colors.Normalize(0, len(x) - 1),
            linewidth=lw,
        )
        lc.set_array(np.arange(len(x)))
        ax.add_collection(lc)

        # add dummy handle for legend
        mid_colour = cmap(0.75)
        ax.plot([], [], color=mid_colour, lw=lw, label=label)

    # reference: green → blue
    add_gradient_line(ref_x, ref_y, "#00b050", "#0070ff", "reference", lw=2.5)

    # actual: yellow → red
    add_gradient_line(act_x, act_y, "#ffd700", "#ff0000", "actual", lw=1.5)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")
    ax.set_title("XY trajectory (light→dark = time)")
    ax.legend(loc="best")

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "traj_xy.png", dpi=150)
    plt.close(fig)
