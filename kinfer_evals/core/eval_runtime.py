"""Common utilities for evals."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
from kinfer.rust_bindings import PyModelRunner
from kinfer_sim.provider import InputState, ModelProvider
from kinfer_sim.server import find_mjcf, get_model_metadata
from kinfer_sim.simulator import MujocoSimulator
from kscale import K
from kscale.web.gen.api import RobotURDFMetadataOutput
from matplotlib import pyplot as plt

from kinfer_evals.core.types import RunArgs
from kinfer_evals.reference_state import ReferenceStateTracker

logger = logging.getLogger(__name__)


class CommandFactory(Protocol):
    """Anything that returns an InputState-compatible object."""

    def __call__(self) -> InputState: ...


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


async def run_episode(
    sim: MujocoSimulator,
    runner: PyModelRunner,
    seconds: float,
    outdir: Path,
    provider: ModelProvider | None = None,
) -> list[dict]:
    """Physics → inference → actuation loop + reference-error logging & plots."""
    tracker = ReferenceStateTracker()

    # metrics we collect every control tick
    time_s: list[float] = []

    command_vx_body: list[float] = []
    command_vy_body: list[float] = []

    actual_vx_body: list[float] = []
    actual_vy_body: list[float] = []

    error_vx_body: list[float] = []
    error_vy_body: list[float] = []

    quat0 = sim._data.sensor("imu_site_quat").data
    yaw0 = get_yaw_from_quaternion(quat0)
    tracker.reset(tuple(sim._data.qpos[:2]), heading_rad=yaw0)

    carry, log, t0 = runner.init(), [], time.time()
    dt_ctrl = 1.0 / sim._control_frequency

    try:
        while time.time() - t0 < seconds:
            # Step physics
            for _ in range(sim.sim_decimation):
                await sim.step()

            # Advance command index if we're using a PrecomputedInputState
            if provider and hasattr(provider.keyboard_state, "step"):
                provider.keyboard_state.step()

            # Get commands
            cmd_vx_body = cmd_vy_body = 0.0
            if provider is not None:
                cmd_vx_body, cmd_vy_body = provider.keyboard_state.value[:2]

            # Inference
            out, carry = runner.step(carry)
            runner.take_action(out)

            # Update reference state
            quat = sim._data.sensor("imu_site_quat").data
            yaw = get_yaw_from_quaternion(quat)
            tracker.step((cmd_vx_body, cmd_vy_body), dt_ctrl, heading_rad=yaw)

            # ----- convert simulator's world-frame base velocity → body frame -----
            vx_world = float(sim._data.qvel[0])
            vy_world = float(sim._data.qvel[1])

            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            vx_act_body = cos_yaw * vx_world + sin_yaw * vy_world
            vy_act_body = -sin_yaw * vx_world + cos_yaw * vy_world

            # ----- store metrics in body frame ------------------------------------
            time_s.append(len(time_s) * dt_ctrl)

            command_vx_body.append(cmd_vx_body)
            command_vy_body.append(cmd_vy_body)

            actual_vx_body.append(vx_act_body)
            actual_vy_body.append(vy_act_body)

            error_vx_body.append(vx_act_body - cmd_vx_body)
            error_vy_body.append(vy_act_body - cmd_vy_body)

            # keep logging the sim state
            log.append(sim.get_state().as_dict())
            await asyncio.sleep(0)

    finally:
        await sim.close()

    # Produce plots
    _plot_velocity_series(time_s, command_vx_body, actual_vx_body, error_vx_body, "x", outdir)
    _plot_velocity_series(time_s, command_vy_body, actual_vy_body, error_vy_body, "y", outdir)

    # -------- summary on stdout --------------------------------------------
    mae_vx = float(np.mean(np.abs(error_vx_body)))
    mae_vy = float(np.mean(np.abs(error_vy_body)))
    rmse_vx = float(np.sqrt(np.mean(np.square(error_vx_body))))
    rmse_vy = float(np.sqrt(np.mean(np.square(error_vy_body))))

    logger.info(
        "\n=== velocity-tracking summary ===\n"
        "Mean-abs error  vx: %.4f m/s   vy: %.4f m/s\n"
        "RMSE            vx: %.4f m/s   vy: %.4f m/s\n"
        "samples: %d\n"
        "==================================\n",
        mae_vx,
        mae_vy,
        rmse_vx,
        rmse_vy,
        len(error_vx_body),
    )

    return log


def save_json(log: Sequence[dict], out: Path, fname: str = "log.json") -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / fname).write_text(json.dumps(log, indent=2))


async def run_factory(
    args: RunArgs,
    cmd_factory: Callable[[], InputState],
    make_commands: Callable[[float], list[list[float]]] | None = None,
) -> None:
    """
    Spin up the sim/runner, optionally precompute a command list, and
    execute `run_episode`.  Used by every eval.

    Args:
        args: common parsed arguments (kinfer, robot, out, seconds, …).
        cmd_factory: 0-arg callable that returns a *stateful* InputState.
        make_commands: if provided, we run the sim once to learn control
            frequency, build the command list *before* the episode starts,
            and replace the provider’s keyboard_state with a PrecomputedInputState.
    """
    sim, runner, provider = await load_sim_and_runner(
        args.kinfer,
        args.robot,
        cmd_factory=cmd_factory,
    )

    # Pre-compute commands if requested
    if make_commands is not None:
        freq = sim._control_frequency
        commands = make_commands(freq)
        provider.keyboard_state = PrecomputedInputState(commands)      # type: ignore[attr-defined]
        args.seconds = len(commands) / freq

    outdir = args.out / time.strftime("%Y%m%d-%H%M%S")
    log = await run_episode(sim, runner, args.seconds, outdir, provider)
    save_json(log, outdir, f"{args.name}_log.json")


class PrecomputedInputState(InputState):
    """InputState that walks through a pre-computed command list."""

    def __init__(self, commands: list[list[float]]) -> None:
        self._cmds = commands
        self._idx = 0
        self.value = self._cmds[0]

    async def update(self, _key: str) -> None:  # not used here
        pass

    def step(self) -> None:  # advance one tick
        if self._idx + 1 < len(self._cmds):
            self._idx += 1
            self.value = self._cmds[self._idx]


def cmd(vx: float = 0.0, yaw: float = 0.0) -> list[float]:
    """Return a 6-D ExpandedControlVector (vx, vy, yaw, h, roll, pitch)."""
    return [vx, 0.0, yaw]


def ramp(start: float, end: float, duration_s: float, freq_hz: float) -> list[float]:
    """Evenly-spaced values from start→end over `duration_s`, at control rate."""
    n = max(1, int(round(duration_s * freq_hz)))
    step = (end - start) / n
    return [start + i * step for i in range(1, n + 1)]
