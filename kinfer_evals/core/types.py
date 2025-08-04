from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunArgs:
    name: str
    kinfer: Path
    robot: str
    out: Path
    seconds: float