"""Walk forward evaluation."""

from kinfer_evals.core.eval_utils import cmd
from kinfer_evals.evals import register


def make_commands(freq: float) -> list[list[float]]:
    """1 s stand → 0.5 s ramp-up → 5 s walk → 0.5 s ramp-down → 1 s stand."""

    def s(t: float) -> int:
        return int(round(t * freq))

    seq = []

    # 1) stand 1 s
    seq += [cmd(0.0)] * s(1.0)

    # 2) ramp-up 0→0.5 m/s in 0.1 m/s increments (0.1 s each)
    for v in (0.1, 0.2, 0.3, 0.4, 0.5):
        seq += [cmd(v)] * s(0.1)

    # 3) cruise 5 s @ 0.5 m/s
    seq += [cmd(0.5)] * s(5.0)

    # 4) ramp-down 0.5→0 in −0.1 m/s steps (0.1 s each)
    for v in (0.4, 0.3, 0.2, 0.1, 0.0):
        seq += [cmd(v)] * s(0.1)

    # 5) stand 1 s
    seq += [cmd(0.0)] * s(1.0)

    return seq


register("walk_forward", make_commands)
