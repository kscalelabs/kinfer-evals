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

import argparse
import asyncio
import time
from pathlib import Path

import colorlogging

from kinfer_evals.evals.common import PrecomputedInputState, cmd, load_sim_and_runner, ramp, run_episode, save_json


def make_commands(freq_hz: float) -> list[list[float]]:
    """Walk-around command script with 1 s yaw transition + 1 s hold."""
    seq: list[list[float]] = []

    def s(t: float) -> int:
        return int(round(t * freq_hz))  # samples helper

    vx = 0.5  # forward speed (m/s)
    yaw = -1.5  # turn-right heading (rad)

    # 1) stand 1 s
    seq += [cmd()] * s(1.0)

    # 2) ramp fwd 0 → VX in 0.5 s
    seq += [cmd(v) for v in ramp(0.0, vx, 0.5, freq_hz)]

    # 3) walk fwd 3 s
    seq += [cmd(vx)] * s(3.0)

    # 4) ramp down to 0 in 0.5 s
    seq += [cmd(v) for v in ramp(vx, 0.0, 0.5, freq_hz)]

    # 5) yaw 0 → YAW smoothly over 1 s
    seq += [cmd(yaw=y) for y in ramp(0.0, yaw, 1.0, freq_hz)]

    # 6) hold heading for 1 s
    seq += [cmd(yaw=yaw)] * s(1.0)

    # 7) ramp fwd again 0 → VX in 0.5 s (keep heading)
    seq += [cmd(v, yaw=yaw) for v in ramp(0.0, vx, 0.5, freq_hz)]

    # 8) walk fwd 3 s at new heading
    seq += [cmd(vx, yaw=yaw)] * s(3.0)

    # 9) ramp down to 0 in 0.5 s (still at YAW)
    seq += [cmd(v, yaw=yaw) for v in ramp(vx, 0.0, 0.5, freq_hz)]

    # 10) stand 3 s at final heading
    seq += [cmd(yaw=yaw)] * s(3.0)

    return seq


async def _run(args: argparse.Namespace) -> None:
    # Boot sim once to learn control frequency.
    sim, runner, provider = await load_sim_and_runner(
        args.kinfer,
        args.robot,
        cmd_factory=lambda: PrecomputedInputState([cmd()]),  # placeholder
    )

    freq = sim._control_frequency
    commands = make_commands(freq)

    # Swap in the real pre-computed state.
    provider.keyboard_state = PrecomputedInputState(commands)  # type: ignore[attr-defined]

    seconds = len(commands) / freq
    outdir = args.out / time.strftime("%Y%m%d-%H%M%S")
    log = await run_episode(sim, runner, seconds, outdir, provider)

    save_json(log, outdir, "walk_around_log.json")
    save_json(commands, outdir, "commands.json")


if __name__ == "__main__":
    colorlogging.configure()

    p = argparse.ArgumentParser()
    p.add_argument("kinfer", type=Path)
    p.add_argument("robot")
    p.add_argument("--out", type=Path, default=Path("runs/walk_around"))
    asyncio.run(_run(p.parse_args()))
