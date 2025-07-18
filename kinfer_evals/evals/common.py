"""Common utilities for evals."""

import asyncio
import json
import time
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from kinfer.rust_bindings import PyModelRunner
from kinfer_evals.reference_state import ReferenceStateTracker
from kinfer_sim.provider import InputState, ModelProvider
from kinfer_sim.server import find_mjcf, get_model_metadata
from kinfer_sim.simulator import MujocoSimulator
from kscale import K
from kscale.web.gen.api import RobotURDFMetadataOutput


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


async def run_episode(
    sim: MujocoSimulator,
    runner: PyModelRunner,
    seconds: float,
    provider: ModelProvider | None = None,
) -> list[dict]:
    """Physics → inference → action loop with reference-error tracking."""
    tracker = ReferenceStateTracker()
    cumulative_pos_err = 0.0
    cumulative_vel_err = 0.0
    samples = 0

    # grab initial world-frame heading
    quat = sim._data.sensor("imu_site_quat").data
    yaw0 = get_yaw_from_quaternion(quat)
    tracker.reset(tuple(sim._data.qpos[:2]), heading_rad=yaw0)

    carry, log, t0 = runner.init(), [], time.time()
    dt_ctrl = 1.0 / sim._control_frequency

    try:
        while time.time() - t0 < seconds:

            # Step physics
            for _ in range(sim.sim_decimation):
                await sim.step()

            # Step command bookkeeping
            if provider and hasattr(provider.keyboard_state, "step"):
                provider.keyboard_state.step()

            # Get commands
            vx_cmd_b, vy_cmd_b = 0.0, 0.0
            if provider is not None:
                cmd_vec = provider.keyboard_state.value
                vx_cmd_b, vy_cmd_b = cmd_vec[0], cmd_vec[1]

            # Inference (get action)
            out, carry = runner.step(carry)
            runner.take_action(out)

            # Update reference state
            quat = sim._data.sensor("imu_site_quat").data
            yaw = get_yaw_from_quaternion(quat)
            tracker.step((vx_cmd_b, vy_cmd_b), dt_ctrl, heading_rad=yaw)

            # Calculate error metrics
            # position error
            act_xy = sim._data.qpos[:2]
            err_pos = float(np.linalg.norm(act_xy - tracker.pos))

            # velocity error (world frame)
            c, s = np.cos(yaw), np.sin(yaw)
            cmd_vx_w = c * vx_cmd_b - s * vy_cmd_b
            cmd_vy_w = s * vx_cmd_b + c * vy_cmd_b
            act_vx_w, act_vy_w = sim._data.qvel[0], sim._data.qvel[1]
            err_vel = float(np.hypot(act_vx_w - cmd_vx_w, act_vy_w - cmd_vy_w))

            cumulative_pos_err += err_pos
            cumulative_vel_err += err_vel
            samples += 1

            # Log
            log.append(sim.get_state().as_dict())
            await asyncio.sleep(0)
    finally:
        await sim.close()

    if samples:
        print(
            f"[eval] ⌀ position error {cumulative_pos_err/samples:.4f} m   "
            f"⌀ velocity error {cumulative_vel_err/samples:.4f} m/s   "
            f"({samples} samples)"
        )

    return log


def save_json(log: Sequence[dict], out: Path, fname: str = "log.json") -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / fname).write_text(json.dumps(log, indent=2))


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


# quick helper to build the six-dim command vector (vx, vy, yaw, h, roll, pitch)
def cmd(vx: float = 0.0, yaw: float = 0.0) -> list[float]:
    """Return a 6-D ExpandedControlVector (vx, vy, yaw, h, roll, pitch)."""
    return [vx, 0.0, yaw, 0.0, 0.0, 0.0]


def ramp(start: float, end: float, duration_s: float, freq_hz: float) -> list[float]:
    """Evenly-spaced values from start→end over `duration_s`, at control rate."""
    n = max(1, int(round(duration_s * freq_hz)))
    step = (end - start) / n
    return [start + i * step for i in range(1, n + 1)]
