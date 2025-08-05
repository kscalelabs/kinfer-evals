"""Stand-still (all-zero command)."""

from kinfer_evals.core.eval_utils import cmd
from kinfer_evals.evals import register

DEFAULT_LEN_S = 5.0  # keep it short; tweak whenever


def make_commands(freq: float) -> list[list[float]]:
    return [cmd()] * int(round(freq * DEFAULT_LEN_S))


register("stand_still", make_commands)
