"""Common utilities for evals."""

import asyncio
import json
import time
from pathlib import Path
from typing import Callable, Protocol, Sequence

from kinfer.rust_bindings import PyModelRunner
from kinfer_sim.provider import InputState, ModelProvider
from kinfer_sim.server import find_mjcf, get_model_metadata
from kinfer_sim.simulator import MujocoSimulator
from kscale import K
from kscale.web.gen.api import RobotURDFMetadataOutput


class CommandFactory(Protocol):
    """Anything that returns an InputState-compatible object."""

    def __call__(self) -> InputState: ...


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
) -> tuple[MujocoSimulator, PyModelRunner]:
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
    return sim, runner


async def run_episode(
    sim: MujocoSimulator,
    runner: PyModelRunner,
    seconds: float,
) -> list[dict]:
    """Physics → inference → action loop."""
    carry, log, t0 = runner.init(), [], time.time()
    try:
        while time.time() - t0 < seconds:
            for _ in range(sim.sim_decimation):
                await sim.step()
            out, carry = runner.step(carry)
            runner.take_action(out)
            log.append(sim.get_state().as_dict())
            await asyncio.sleep(0)
    finally:
        await sim.close()
    return log


def save_json(log: Sequence[dict], out: Path, fname: str = "log.json") -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / fname).write_text(json.dumps(log, indent=2))
