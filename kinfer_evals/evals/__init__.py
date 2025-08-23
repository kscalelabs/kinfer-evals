"""Ultra-light registry mapping eval-name → make_commands function."""

from typing import Callable, Dict, List

CommandMaker = Callable[[float], List[List[float]]]  # (freq only)
REGISTRY: Dict[str, CommandMaker] = {}


def register(name: str, fn: CommandMaker) -> None:
    if name in REGISTRY:
        raise ValueError(f"Duplicate eval '{name}'")
    REGISTRY[name] = fn


# keep explicit imports so registration happens on package import
from . import (
    stand_still,  # noqa: E402,F401
    walk_forward,  # noqa: E402,F401
    walk_forward_right,  # noqa: E402,F401
    walk_1ms,  # noqa: E402,F401
)
