"""Project-wide typed arrays and data containers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Sequence, TypedDict

import numpy as np
import numpy.typing as npt
from kmotions.motions import Motion

F32 = np.float32
I16 = np.int16

Array1 = npt.NDArray[F32]  # shape (T,)
Array2 = npt.NDArray[F32]  # shape (T, N)
Int1 = npt.NDArray[I16]  # shape (T,)


@dataclass
class RunArgs:
    eval_name: str
    kinfer: Path
    robot: str
    out: Path
    render: bool = False
    # Optional: use a local, pre-downloaded URDF/MJCF directory instead of fetching
    local_model_dir: Path | None = None
    # Optional: passthrough for command type (currently unused by evals but accepted)
    command_type: str | None = None


class CommandFactory(Protocol):
    """Factory that creates a Motion."""

    def __call__(self, dt: float) -> Motion: ...


class RunInfo(TypedDict):
    kinfer_file: str
    robot: str
    eval_name: str
    timestamp: str
    outdir: str


@dataclass(frozen=True)
class EpisodeData:
    time: Array1  # (T,)
    qpos: Array2  # (T, nq)
    qvel: Array2  # (T, nv)
    actuator_force: Array2  # (T, nu)
    action_target: Optional[Array2]  # (T, nu) or None
    commands: dict[str, Array1]  # {cmd_name -> (T,)} command values over time
    contact_wrench: list[np.ndarray]  # list[(6*ncon_t,)] float32 per step
    contact_body: list[np.ndarray]  # list[(2*ncon_t,)] int16 per step
    contact_count: Int1  # (T,)
    contact_force_mag: Array1  # (T,)
    body_names: list[str]
    dt: float
    # Policy inputs captured during the run: {name -> (T, N)}
    inputs: dict[str, Array2] = field(default_factory=dict)


class CommandProvider:
    """Command provider that uses kmotions Motion objects.

    Compatible with kinfer-sim's ModelProvider interface which expects:
    - get_cmd(command_names: Sequence[str]) -> list[float]
    """

    def __init__(self, motion: Motion, command_names: Sequence[str]) -> None:
        """Initialize with a kmotions Motion object.

        Args:
            motion: kmotions Motion object that generates motion frames
            command_names: List of command names from kinfer metadata
        """
        self._motion = motion
        self._command_names = command_names
        self._current_frame = motion.get_next_motion_frame()
        if self._current_frame is None:
            raise ValueError("Motion object produced no frames")

    def get_cmd(self, command_names: Sequence[str]) -> list[float]:
        """Get current command vector, extracting named commands from motion frame.

        Args:
            command_names: Sequence of command names to extract.

        Returns:
            List of floats corresponding to the requested command names, using .get() with default 0.0.
        """
        if self._current_frame is None:
            return [0.0] * len(command_names)
        # Extract values from the motion frame dict using .get() with default 0.0
        return [self._current_frame.get(name, 0.0) for name in command_names]

    def step(self) -> None:
        """Advance to the next motion frame."""
        next_frame = self._motion.get_next_motion_frame()
        # Always update current_frame, even if None (motion complete)
        self._current_frame = next_frame

    @property
    def current_frame(self) -> dict[str, float] | None:
        """Get the current motion frame."""
        return self._current_frame
