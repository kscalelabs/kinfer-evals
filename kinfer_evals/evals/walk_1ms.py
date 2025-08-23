"""Walk 1 m/s with right/left 90° turns (instant vx changes).

sequence:
- stand still 5 s
- walk forward at 1.0 m/s for 5 s
- turn right 90°
- walk forward 5 s
- turn left 90°
- walk forward 5 s
- stand still 5 s
"""


from kinfer_evals.core.eval_utils import cmd
from kinfer_evals.evals import register
import math


def make_commands(freq: float) -> list[list[float]]:
    """Return the full command list for this manoeuvre."""
    seq = []

    def s(t: float) -> int:
        return int(round(t * freq))  # samples helper

    vx = 1.0  # forward speed  [m s⁻¹]
    omega_mag = 1.5  # angular speed magnitude for turns  [rad s⁻¹]
    t90 = (math.pi / 2.0) / omega_mag  # duration to rotate 90° at omega_mag

    # 1) stand still 5 s
    seq += [cmd()] * s(5.0)

    # 2) walk forward 5 s (instant vx change)
    seq += [cmd(vx)] * s(5.0)

    # 3) turn right 90° (in-place)
    seq += [cmd(yaw=-omega_mag)] * s(t90)

    # 4) walk forward 5 s
    seq += [cmd(vx)] * s(5.0)

    # 5) turn left 90° (in-place)
    seq += [cmd(yaw=omega_mag)] * s(t90)

    # 6) walk forward 5 s
    seq += [cmd(vx)] * s(5.0)

    # 7) stand still 5 s
    seq += [cmd()] * s(5.0)

    return seq


register("walk_1ms", make_commands)
