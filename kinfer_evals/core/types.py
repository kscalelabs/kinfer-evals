from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunArgs:
    eval_name: str
    kinfer: Path
    robot: str
    out: Path
    seconds: float