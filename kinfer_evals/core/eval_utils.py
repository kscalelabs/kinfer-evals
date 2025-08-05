"""Shared utilities for running evals."""

import asyncio
from pathlib import Path
from typing import Callable

import numpy as np
from kinfer.rust_bindings import PyModelRunner
from kinfer_sim.provider import ModelProvider
from kinfer_sim.server import find_mjcf, get_model_metadata
from kinfer_sim.simulator import MujocoSimulator
from kscale import K
from kscale.web.gen.api import RobotURDFMetadataOutput
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors

from kinfer_evals.core.eval_types import CommandFactory


def cmd(vx: float = 0.0, yaw: float = 0.0) -> list[float]:
    """Return a 6-D ExpandedControlVector (vx, vy, yaw, h, roll, pitch)."""
    return [vx, 0.0, yaw]


def ramp(start: float, end: float, duration_s: float, freq_hz: float) -> list[float]:
    """Evenly-spaced values from start→end over `duration_s`, at control rate."""
    n = max(1, int(round(duration_s * freq_hz)))
    step = (end - start) / n
    return [start + i * step for i in range(1, n + 1)]


def get_yaw_from_quaternion(quat: np.ndarray) -> float:
    """Extract yaw angle from quaternion data."""
    return float(
        np.arctan2(
            2 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
        )
    )


def default_sim(
    mjcf: Path,
    meta: RobotURDFMetadataOutput,
    *,
    dt: float = 1e-4,
    render: bool = True,
) -> MujocoSimulator:
    return MujocoSimulator(
        model_path=mjcf,
        model_metadata=meta,
        dt=dt,
        render_mode="window" if render else "offscreen",
        start_height=1.1,
    )


async def load_sim_and_runner(
    kinfer: Path,
    robot: str,
    cmd_factory: CommandFactory,
    *,
    make_sim: Callable[..., MujocoSimulator] = default_sim,
) -> tuple[MujocoSimulator, PyModelRunner, ModelProvider]:
    """Shared download + construction logic."""
    async with K() as api:
        model_dir, meta = await asyncio.gather(
            api.download_and_extract_urdf(robot, cache=True),
            get_model_metadata(api, robot),
        )

    mjcf = find_mjcf(model_dir)
    sim = make_sim(mjcf, meta)
    provider = ModelProvider(sim, keyboard_state=cmd_factory())
    runner = PyModelRunner(str(kinfer), provider)
    return sim, runner, provider


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
