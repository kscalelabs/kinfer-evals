"""Common utilities for evals."""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
from kinfer.rust_bindings import PyModelRunner
from kinfer_sim.provider import ModelProvider
from kinfer_sim.simulator import MujocoSimulator
from tabulate import tabulate

from kinfer_evals.core.eval_types import PrecomputedInputState, RunArgs
from kinfer_evals.core.eval_utils import get_yaw_from_quaternion, load_sim_and_runner
from kinfer_evals.core.plots import (
    _plot_xy_trajectory,
    plot_accel,
    plot_heading,
    plot_omega,
    plot_velocity,
)
from kinfer_evals.reference_state import ReferenceStateTracker

from kinfer_evals.core.notion import push_summary

if TYPE_CHECKING:
    from kinfer_evals.evals import CommandMaker

logger = logging.getLogger(__name__)


async def run_episode(
    sim: MujocoSimulator,
    runner: PyModelRunner,
    seconds: float,
    outdir: Path,
    provider: ModelProvider | None = None,
    run_info: dict | None = None,
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

    # yaw / ω
    yaw_ref: list[float] = []
    yaw_act: list[float] = []
    cmd_omega: list[float] = []

    # XY traces
    ref_x: list[float] = []
    ref_y: list[float] = []
    act_x: list[float] = []
    act_y: list[float] = []

    quat0 = sim._data.sensor("imu_site_quat").data
    yaw0 = get_yaw_from_quaternion(quat0)
    tracker.reset(tuple(sim._data.qpos[:2]), yaw=yaw0)

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
            omega_cmd = provider.keyboard_state.value[2] if provider else 0.0  # commanded angular vel (rad/s)
            tracker.step((cmd_vx_body, cmd_vy_body), omega_cmd, dt_ctrl)

            # Convert velocity from world frame to body frame
            vx_world = float(sim._data.qvel[0])
            vy_world = float(sim._data.qvel[1])

            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            vx_act_body = cos_yaw * vx_world + sin_yaw * vy_world
            vy_act_body = -sin_yaw * vx_world + cos_yaw * vy_world

            # Store metrics in body frame
            time_s.append(len(time_s) * dt_ctrl)

            command_vx_body.append(cmd_vx_body)
            command_vy_body.append(cmd_vy_body)

            actual_vx_body.append(vx_act_body)
            actual_vy_body.append(vy_act_body)

            error_vx_body.append(vx_act_body - cmd_vx_body)
            error_vy_body.append(vy_act_body - cmd_vy_body)

            yaw_ref.append(tracker.yaw)
            yaw_act.append(yaw)
            cmd_omega.append(omega_cmd)

            ref_x.append(tracker.pos_x)
            ref_y.append(tracker.pos_y)
            act_x.append(float(sim._data.qpos[0]))
            act_y.append(float(sim._data.qpos[1]))

            # keep logging the sim state
            log.append(sim.get_state().as_dict())
            await asyncio.sleep(0)

    finally:
        await sim.close()

    # Track acceleration
    dt = dt_ctrl
    command_ax_body = np.diff(command_vx_body) / dt
    command_ay_body = np.diff(command_vy_body) / dt
    actual_ax_body = np.diff(actual_vx_body) / dt
    actual_ay_body = np.diff(actual_vy_body) / dt

    # time stamps shortened by one sample
    time_acc = time_s[1:]

    err_ax = actual_ax_body - command_ax_body
    err_ay = actual_ay_body - command_ay_body

    # total (magnitude) acceleration
    cmd_am = np.sqrt(command_ax_body**2 + command_ay_body**2)
    act_am = np.sqrt(actual_ax_body**2 + actual_ay_body**2)
    err_am = act_am - cmd_am

    # unwrap yaw → avoid π jumps
    yaw_ref_u = np.unwrap(yaw_ref)
    yaw_act_u = np.unwrap(yaw_act)
    yaw_err = yaw_act_u - yaw_ref_u

    act_omega = np.diff(yaw_act_u) / dt
    err_omega = act_omega - np.asarray(cmd_omega[:-1])
    time_omega = time_s[1:]

    # Produce plots
    assert run_info is not None
    run_meta = {
        "kinfer": run_info["kinfer_file"] if run_info else "",
        "robot": run_info["robot"] if run_info else "",
        "eval_name": run_info["eval_name"] if run_info else "",
        "timestamp": run_info["timestamp"] if run_info else "",
        "outdir": run_info["output_directory"] if run_info else "",
    }

    plot_velocity(time_s, command_vx_body, actual_vx_body, error_vx_body, "x", outdir, run_meta)
    plot_velocity(time_s, command_vy_body, actual_vy_body, error_vy_body, "y", outdir, run_meta)

    _plot_xy_trajectory(ref_x, ref_y, act_x, act_y, outdir, run_meta)

    cmd_ax_l, cmd_ay_l = command_ax_body.tolist(), command_ay_body.tolist()
    act_ax_l, act_ay_l = actual_ax_body.tolist(), actual_ay_body.tolist()
    err_ax_l, err_ay_l = err_ax.tolist(), err_ay.tolist()
    cmd_am_l, act_am_l, err_am_l = cmd_am.tolist(), act_am.tolist(), err_am.tolist()

    plot_accel(time_acc, cmd_ax_l, act_ax_l, err_ax_l, "x", outdir, run_meta)
    plot_accel(time_acc, cmd_ay_l, act_ay_l, err_ay_l, "y", outdir, run_meta)
    plot_accel(time_acc, cmd_am_l, act_am_l, err_am_l, "mag", outdir, run_meta)

    # heading & ω plots
    yaw_ref_l, yaw_act_l, yaw_err_l = yaw_ref_u.tolist(), yaw_act_u.tolist(), yaw_err.tolist()
    act_omega_l, err_omega_l = act_omega.tolist(), err_omega.tolist()

    plot_heading(time_s, yaw_ref_l, yaw_act_l, yaw_err_l, outdir, run_meta)
    plot_omega(time_omega, cmd_omega[:-1], act_omega_l, err_omega_l, outdir, run_meta)

    # Velocity errors
    mae_vx = float(np.mean(np.abs(error_vx_body)))
    mae_vy = float(np.mean(np.abs(error_vy_body)))
    rmse_vx = float(np.sqrt(np.mean(np.square(error_vx_body))))
    rmse_vy = float(np.sqrt(np.mean(np.square(error_vy_body))))

    # Acceleration errors
    mae_ax = float(np.mean(np.abs(err_ax)))
    mae_ay = float(np.mean(np.abs(err_ay)))
    mae_am = float(np.mean(np.abs(err_am)))
    rmse_ax = float(np.sqrt(np.mean(np.square(err_ax))))
    rmse_ay = float(np.sqrt(np.mean(np.square(err_ay))))
    rmse_am = float(np.sqrt(np.mean(np.square(err_am))))

    # Heading / ω errors
    mae_yaw = float(np.mean(np.abs(yaw_err)))
    rmse_yaw = float(np.sqrt(np.mean(np.square(yaw_err))))
    mae_om = float(np.mean(np.abs(err_omega)))
    rmse_om = float(np.sqrt(np.mean(np.square(err_omega))))

    # 1) velocity & acceleration (vector-axes table)
    vel_acc_table: list[list[object]] = [
        ["metric", "x-axis", "y-axis", "magnitude"],
        ["mean abs velocity error  [m/s]", f"{mae_vx:.4f}", f"{mae_vy:.4f}", "—"],
        ["root mean square velocity error  [m/s]", f"{rmse_vx:.4f}", f"{rmse_vy:.4f}", "—"],
        ["mean abs acceleration error  [m/s²]", f"{mae_ax:.4f}", f"{mae_ay:.4f}", f"{mae_am:.4f}"],
        ["root mean square acceleration error [m/s²]", f"{rmse_ax:.4f}", f"{rmse_ay:.4f}", f"{rmse_am:.4f}"],
        ["samples (velocity)", len(error_vx_body), len(error_vy_body), "—"],
    ]

    # 2) heading & angular-velocity (single-value table)
    heading_table: list[list[object]] = [
        ["metric", "value"],
        ["mean abs heading error  [rad]", f"{mae_yaw:.4f}"],
        ["root mean square heading error  [rad]", f"{rmse_yaw:.4f}"],
        ["mean abs ω error  [rad/s]", f"{mae_om:.4f}"],
        ["root mean square ω error  [rad/s]", f"{rmse_om:.4f}"],
        ["samples (ω)", len(err_omega)],
    ]

    logger.info(
        "\n%s\n\n%s\n",
        tabulate(vel_acc_table, headers="firstrow", tablefmt="github"),
        tabulate(heading_table, headers="firstrow", tablefmt="github"),
    )

    summary: dict[str, object] = {
        # velocity
        "mae_vel_x": mae_vx,
        "mae_vel_y": mae_vy,
        "rmse_vel_x": rmse_vx,
        "rmse_vel_y": rmse_vy,
        "vel_samples": len(error_vx_body),
        # acceleration
        "mae_accel_x": mae_ax,
        "mae_accel_y": mae_ay,
        "mae_accel_mag": mae_am,
        "rmse_accel_x": rmse_ax,
        "rmse_accel_y": rmse_ay,
        "rmse_accel_mag": rmse_am,
        # heading (yaw)
        "mae_heading": mae_yaw,
        "rmse_heading": rmse_yaw,
        # angular velocity
        "mae_omega": mae_om,
        "rmse_omega": rmse_om,
        "omega_samples": len(err_omega),
    }

    # merge with run-level metadata and save **one** file
    combined: dict[str, object] = {**run_info, **summary}
    (outdir / "run_summary.json").write_text(json.dumps(combined, indent=2))
    logger.info("Saved combined summary to %s", outdir / "run_summary.json")

    try:
        pngs = sorted(outdir.glob("*.png"))
        url = push_summary(combined, pngs)
        logger.info("Logged run to Notion: %s", url)
    except Exception as exc:
        logger.warning("Failed to push results to Notion: %s", exc)

    return log


def save_json(log: Sequence[Mapping[str, object]], out: Path, fname: str = "log.json") -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / fname).write_text(json.dumps(log, indent=2))
    logger.info("Saved log to %s", out / fname)


def build_run_info(args: RunArgs, timestamp: str, outdir: Path, duration_seconds: float) -> dict:
    """Save metadata about this run for tracking purposes."""
    run_info = {
        "timestamp": timestamp,
        "eval_name": args.eval_name,
        "kinfer_file": str(args.kinfer.absolute()),
        "robot": args.robot,
        "duration_seconds": duration_seconds,
        "output_directory": str(outdir.absolute()),
    }

    return run_info


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
    commands = make_cmds(freq)
    provider.keyboard_state = PrecomputedInputState(commands)
    duration_seconds = len(commands) / freq

    # Create timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = args.out / eval_name / timestamp

    # Save & keep run metadata
    run_info = build_run_info(args, timestamp, outdir, duration_seconds)

    log = await run_episode(sim, runner, duration_seconds, outdir, provider, run_info)
    save_json(log, outdir, f"{eval_name}_log.json")
