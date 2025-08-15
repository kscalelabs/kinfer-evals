"""EpisodeRunner + sinks (HDF5, video)."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from kinfer.rust_bindings import PyModelRunner
from kinfer_sim.provider import ModelProvider
from kinfer_sim.simulator import MujocoSimulator
from kmv.app.viewer import DefaultMujocoViewer
from kmv.utils.logging import VideoWriter

from kinfer_evals.core.eval_types import RunInfo
from kinfer_evals.core.io_h5 import EpisodeWriter


@dataclass
class StepSnapshot:
    step_idx: int
    time_s: float
    cmd_vx: float
    cmd_vy: float
    cmd_omega: float
    action: np.ndarray
    inputs: dict[str, np.ndarray]
    sim: MujocoSimulator


class StepSink(Protocol):
    def on_step(self, snap: StepSnapshot) -> None: ...
    def close(self) -> None: ...


class H5Sink:
    def __init__(self, path: Path, sim: MujocoSimulator, *, run_info: RunInfo) -> None:
        self._writer = EpisodeWriter(
            path,
            sim._model,
            control_rate_hz=sim._control_frequency,
            run_info=run_info,
        )

    def on_step(self, snap: StepSnapshot) -> None:
        self._writer.append(
            snap.sim._data,
            t=snap.time_s,
            cmd_vel=(snap.cmd_vx, snap.cmd_vy, snap.cmd_omega),
            action=snap.action,
            inputs=snap.inputs,
        )

    def close(self) -> None:
        self._writer.close()


class VideoSink:
    """Writes a 30 FPS mp4 using GLFW viewer frames, if available."""

    def __init__(self, path: Path, sim: MujocoSimulator) -> None:
        if not isinstance(sim._viewer, DefaultMujocoViewer):
            raise RuntimeError("VideoSink requires DefaultMujocoViewer (GLFW). Run without --render.")
        fps_target = 30
        self._decim = max(1, int(round(sim._control_frequency / fps_target)))
        self._vw = VideoWriter(path, fps=fps_target)

    def on_step(self, snap: StepSnapshot) -> None:
        if snap.step_idx % self._decim == 0:
            self._vw.append(snap.sim.read_pixels())

    def close(self) -> None:
        self._vw.close()


class EpisodeRollout:
    """Runs the physics+policy loop and fans out to sinks."""

    def __init__(
        self,
        sim: MujocoSimulator,
        runner: PyModelRunner,
        provider: ModelProvider | None,
        sinks: list[StepSink],
    ) -> None:
        self._sim = sim
        self._runner = runner
        self._provider = provider
        self._sinks = sinks

    async def run(self, seconds: float) -> None:
        dt_ctrl = 1.0 / self._sim._control_frequency
        n_ctrl_steps = int(round(seconds * self._sim._control_frequency))

        carry = self._runner.init()
        step_idx = 0

        try:
            while step_idx < n_ctrl_steps:
                for _ in range(self._sim.sim_decimation):
                    await self._sim.step()

                # Advance command index if available
                if self._provider and hasattr(self._provider.keyboard_state, "step"):
                    self._provider.keyboard_state.step()

                # Read commands
                cmd_vx_body = cmd_vy_body = cmd_omega = 0.0
                if self._provider is not None:
                    val = getattr(self._provider.keyboard_state, "value", [0.0, 0.0, 0.0])
                    if len(val) >= 2:
                        cmd_vx_body, cmd_vy_body = float(val[0]), float(val[1])
                    if len(val) >= 3:
                        cmd_omega = float(val[2])

                # Inference
                out, carry = self._runner.step(carry)
                self._runner.take_action(out)

                arrays_copy = {k: v.copy() for k, v in (self._provider.arrays.items() if self._provider else [])}

                snap = StepSnapshot(
                    step_idx=step_idx,
                    time_s=step_idx * dt_ctrl,
                    cmd_vx=cmd_vx_body,
                    cmd_vy=cmd_vy_body,
                    cmd_omega=cmd_omega,
                    action=out,
                    inputs=arrays_copy,
                    sim=self._sim,
                )
                for sink in self._sinks:
                    sink.on_step(snap)

                await asyncio.sleep(0)
                step_idx += 1
        finally:
            for sink in self._sinks:
                sink.close()
            await self._sim.close()
