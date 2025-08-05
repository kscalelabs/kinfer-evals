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


def make_commands(freq: float) -> list[list[float]]:
    """Return the full command list for this manoeuvre."""
    seq = []

    def s(t: float) -> int:
        return int(round(t * freq))  # samples helper

    vx = 0.5  # forward speed  [m s⁻¹]
    omega = -1.5  # right-turn angular velocity  [rad s⁻¹]  (≈ 86 ° s⁻¹)

    # 1) stand 1 s
    seq += [cmd()] * s(1.0)

    # 2) ramp fwd 0 → VX in 0.5 s
    seq += [cmd(v) for v in ramp(0.0, vx, 0.5, freq)]

    # 3) walk fwd 3 s
    seq += [cmd(vx)] * s(3.0)

    # 4) ramp down to 0 in 0.5 s
    seq += [cmd(v) for v in ramp(vx, 0.0, 0.5, freq)]

    # 5) spin-up: ω 0 → omega in 0.5 s  (in-place turn)
    seq += [cmd(yaw=w) for w in ramp(0.0, omega, 0.5, freq)]

    # 6) keep turning for 0.5 s
    seq += [cmd(yaw=omega)] * s(0.5)

    # 7) spin-down ω omega → 0 in 0.5 s
    seq += [cmd(yaw=w) for w in ramp(omega, 0.0, 0.5, freq)]

    # 8) ramp fwd again 0 → VX in 0.5 s (now facing right)
    seq += [cmd(v) for v in ramp(0.0, vx, 0.5, freq)]

    # 9) walk fwd 3 s
    seq += [cmd(vx)] * s(3.0)

    # 10) ramp down to 0 in 0.5 s
    seq += [cmd(v) for v in ramp(vx, 0.0, 0.5, freq)]

    # 11) stand 3 s
    seq += [cmd()] * s(3.0)

    return seq


register("walk_forward_right", make_commands)
