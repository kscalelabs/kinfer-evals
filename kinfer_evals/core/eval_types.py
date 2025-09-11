"""Project-wide typed arrays and data containers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, TypedDict

import numpy as np
import numpy.typing as npt
from kinfer_sim.provider import InputState

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
    """Anything that returns an InputState-compatible object."""

    def __call__(self) -> InputState: ...


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
    cmd_vel: Array2  # (T, 3)  [vx, vy, omega]
    contact_wrench: list[np.ndarray]  # list[(6*ncon_t,)] float32 per step
    contact_body: list[np.ndarray]  # list[(2*ncon_t,)] int16 per step
    contact_count: Int1  # (T,)
    contact_force_mag: Array1  # (T,)
    body_names: list[str]
    dt: float
    # Policy inputs captured during the run: {name -> (T, N)}
    inputs: dict[str, Array2] = field(default_factory=dict)


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
