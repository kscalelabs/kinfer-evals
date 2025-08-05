"""Stand-still (all-zero command)"""

from kinfer_evals.core.eval_utils import cmd
from kinfer_evals.evals import register


def make_commands(freq: float, seconds: float):
    return [cmd()] * int(round(freq * seconds))


register("stand_still", make_commands)
