"""Common utilities for evals."""

import asyncio
import json
import logging
import time
from pathlib import Path

# Import CommandMaker type (avoiding circular import)
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
from kinfer.rust_bindings import PyModelRunner
from kinfer_sim.provider import ModelProvider
from kinfer_sim.simulator import MujocoSimulator

from kinfer_evals.core.eval_types import PrecomputedInputState, RunArgs
from kinfer_evals.core.eval_utils import _plot_velocity_series, get_yaw_from_quaternion, load_sim_and_runner
from kinfer_evals.reference_state import ReferenceStateTracker

if TYPE_CHECKING:
    from kinfer_evals.evals import CommandMaker

logger = logging.getLogger(__name__)


async def run_episode(
    sim: MujocoSimulator,
    runner: PyModelRunner,
    seconds: float,
    outdir: Path,
    provider: ModelProvider | None = None,
) -> list[Mapping[str, object]]:
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


def save_json(log: Sequence[Mapping[str, object]], out: Path, fname: str = "log.json") -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / fname).write_text(json.dumps(log, indent=2))


async def run_eval(
    make_cmds: "CommandMaker",
    eval_name: str,
    args: RunArgs,
) -> None:
    """Common driver used by every eval.

    • spin up sim/runner with a dummy keyboard state
    • build the full command list upfront
    • wrap it in PrecomputedInputState
    • run the episode & save artefacts
    """
    sim, runner, provider = await load_sim_and_runner(
        args.kinfer,
        args.robot,
        cmd_factory=lambda: PrecomputedInputState([[0.0, 0.0, 0.0]]),
    )

    freq = sim._control_frequency
    commands = make_cmds(freq, args.seconds)
    provider.keyboard_state = PrecomputedInputState(commands)
    args.seconds = len(commands) / freq

    outdir = args.out / eval_name
    log = await run_episode(sim, runner, args.seconds, outdir, provider)
    save_json(log, outdir, f"{eval_name}_log.json")
