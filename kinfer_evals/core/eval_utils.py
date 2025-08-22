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

from kinfer_evals.core.eval_types import CommandFactory


def cmd(vx: float = 0.0, yaw: float = 0.0) -> list[float]:
    """Return a 16-D control vector matching ControlVectorInputState.

    Indices:
    0: vx [m/s]
    1: vy [m/s]
    2: yaw rate [rad/s]
    3: base height offset [m]
    4: base roll [rad]
    5: base pitch [rad]
    6-10: right arm (shoulder pitch, shoulder roll, elbow pitch, elbow roll, wrist pitch) [rad]
    11-15: left arm (shoulder pitch, shoulder roll, elbow pitch, elbow roll, wrist pitch) [rad]

    Only the first `model_num_commands` are consumed by the model; the rest
    are ignored if the model expects fewer.
    """
    vec = [0.0] * 16
    vec[0] = float(vx)
    vec[2] = float(yaw)
    return vec


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
    render: bool = False,
    free_camera: bool = False,
) -> MujocoSimulator:
    return MujocoSimulator(
        model_path=mjcf,
        model_metadata=meta,
        dt=dt,
        render_mode="window" if render else "offscreen",
        start_height=1.1,
        free_camera=free_camera,
    )


async def load_sim_and_runner(
    kinfer: Path,
    robot: str,
    cmd_factory: CommandFactory,
    *,
    make_sim: Callable[..., MujocoSimulator] = default_sim,
    **sim_kwargs: object,
) -> tuple[MujocoSimulator, PyModelRunner, ModelProvider]:
    """Shared download + construction logic."""
    async with K() as api:
        model_dir, meta = await asyncio.gather(
            api.download_and_extract_urdf(robot, cache=True),
            get_model_metadata(api, robot),
        )

    mjcf = find_mjcf(model_dir)
    sim = make_sim(mjcf, meta, **sim_kwargs)
    provider = ModelProvider(sim, keyboard_state=cmd_factory())
    runner = PyModelRunner(str(kinfer), provider)
    return sim, runner, provider
