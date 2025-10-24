"""EpisodeRunner + sinks (HDF5, video)."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from kinfer.rust_bindings import PyModelRunner
from kinfer_sim.provider import ModelProvider
from kinfer_sim.simulator import MujocoSimulator
from kmv.app.viewer import DefaultMujocoViewer
from kmv.utils.logging import VideoWriter

from kinfer_evals.core.eval_types import CommandProvider, RunInfo
from kinfer_evals.core.io_h5 import EpisodeWriter

logger = logging.getLogger(__name__)


@dataclass
class StepSnapshot:
    step_idx: int
    time_s: float
    command_frame: dict[str, float]
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
            command_frame=snap.command_frame,
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
        command_provider: CommandProvider | None,
        provider: ModelProvider | None,
        sinks: list[StepSink],
    ) -> None:
        self._sim = sim
        self._runner = runner
        self._command_provider = command_provider
        self._provider = provider
        self._sinks = sinks

    async def run(self, seconds: float | None) -> None:
        dt_ctrl = 1.0 / self._sim._control_frequency
        
        # If seconds is None, run until motion completes
        if seconds is not None:
            n_ctrl_steps = int(round(seconds * self._sim._control_frequency))
        else:
            n_ctrl_steps = None  # Run indefinitely until motion completes

        step_idx = 0
        logger.info("Starting rollout, n_ctrl_steps=%s", n_ctrl_steps)

        try:
            while n_ctrl_steps is None or step_idx < n_ctrl_steps:
                # Check if motion is complete before doing anything
                if n_ctrl_steps is None and self._command_provider is not None:
                    if self._command_provider.current_frame is None:
                        logger.info("Motion completed at step %d", step_idx)
                        break
                
                if step_idx % 50 == 0:
                    logger.info("Step %d, current_frame is None: %s", step_idx, 
                               self._command_provider.current_frame is None if self._command_provider else "no provider")
                
                # Get current command frame
                command_frame = {}
                if self._command_provider is not None and self._command_provider.current_frame is not None:
                    command_frame = self._command_provider.current_frame.copy()
                
                # Inference and action in one call (new PyModelRunner API)
                self._runner.step_and_take_action()

                # Advance to next frame after inference
                if self._command_provider and hasattr(self._command_provider, "step"):
                    self._command_provider.step()

                # Run simulation for one control step
                for _ in range(self._sim.sim_decimation):
                    await self._sim.step()

                # Get arrays from the ModelProvider
                arrays_copy: dict[str, np.ndarray] = {}
                action_copy: np.ndarray = np.array([])
                if self._provider is not None and hasattr(self._provider, "arrays"):
                    arrays_copy = {k: v.copy() for k, v in self._provider.arrays.items()}
                    # Get action from arrays if available
                    action_copy = arrays_copy.get("action", np.array([]))

                snap = StepSnapshot(
                    step_idx=step_idx,
                    time_s=step_idx * dt_ctrl,
                    command_frame=command_frame,
                    action=action_copy,
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
