"""Walk forward and right evaluation.

stand 1 s
ramp → walk fwd 0.5 s
walk fwd 3 s
ramp → stop    0.5 s
ramp → turn R  0.5 s (negative yaw command)
keep turn      0.5 s
ramp → stop    0.5 s
ramp → walk fwd 0.5 s
walk fwd 3 s
ramp → stop    0.5 s
stand 3 s
"""

from kinfer_evals.core.eval_utils import cmd, ramp
from kinfer_evals.evals import register


def make_commands(freq: float, _seconds: float):
    """Walk-around command script with 1 s yaw transition + 1 s hold."""
    seq = []

    def s(t: float) -> int:
        return int(round(t * freq))  # samples helper

    vx = 0.5  # forward speed (m/s)
    yaw = -1.5  # turn-right heading (rad)

    # 1) stand 1 s
    seq += [cmd()] * s(1.0)

    # 2) ramp fwd 0 → VX in 0.5 s
    seq += [cmd(v) for v in ramp(0.0, vx, 0.5, freq)]

    # 3) walk fwd 3 s
    seq += [cmd(vx)] * s(3.0)

    # 4) ramp down to 0 in 0.5 s
    seq += [cmd(v) for v in ramp(vx, 0.0, 0.5, freq)]

    # 5) yaw 0 → YAW smoothly over 1 s
    seq += [cmd(yaw=y) for y in ramp(0.0, yaw, 1.0, freq)]

    # 6) hold heading for 1 s
    seq += [cmd(yaw=yaw)] * s(1.0)

    # 7) ramp fwd again 0 → VX in 0.5 s (keep heading)
    seq += [cmd(v, yaw=yaw) for v in ramp(0.0, vx, 0.5, freq)]

    # 8) walk fwd 3 s at new heading
    seq += [cmd(vx, yaw=yaw)] * s(3.0)

    # 9) ramp down to 0 in 0.5 s (still at YAW)
    seq += [cmd(v, yaw=yaw) for v in ramp(vx, 0.0, 0.5, freq)]

    # 10) stand 3 s at final heading
    seq += [cmd(yaw=yaw)] * s(3.0)

    return seq


register("walk_forward_right", make_commands)
