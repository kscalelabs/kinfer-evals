"""Types shared across the eval package."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from kinfer_sim.provider import InputState


@dataclass
class RunArgs:
    eval_name: str
    kinfer: Path
    robot: str
    out: Path
    render: bool = False


class CommandFactory(Protocol):
    """Anything that returns an InputState-compatible object."""

    def __call__(self) -> InputState: ...


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
