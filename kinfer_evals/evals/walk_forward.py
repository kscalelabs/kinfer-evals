"""Walk forward evaluation."""

import argparse
import asyncio
import time
from pathlib import Path

import colorlogging

from kinfer_evals.core.eval_runtime import (
    PrecomputedInputState,
    cmd,
    run_factory,
)
from kinfer_evals.core.types import RunArgs


def make_commands(freq_hz: float) -> list[list[float]]:
    """1 s stand → 0.5 s ramp-up → 5 s walk → 0.5 s ramp-down → 1 s stand."""

    def steps(s: float) -> int:
        return int(round(s * freq_hz))

    seq: list[list[float]] = []

    # 1) stand 1 s
    seq += [cmd(0.0)] * steps(1.0)

    # 2) ramp-up 0→0.5 m/s in 0.1 m/s increments (0.1 s each)
    for v in (0.1, 0.2, 0.3, 0.4, 0.5):
        seq += [cmd(v)] * steps(0.1)

    # 3) cruise 5 s @ 0.5 m/s
    seq += [cmd(0.5)] * steps(5.0)

    # 4) ramp-down 0.5→0 in −0.1 m/s steps (0.1 s each)
    for v in (0.4, 0.3, 0.2, 0.1, 0.0):
        seq += [cmd(v)] * steps(0.1)

    # 5) stand 1 s
    seq += [cmd(0.0)] * steps(1.0)

    return seq


async def _main(args: argparse.Namespace) -> None:
    await run_factory(
        RunArgs(
            name="walk_forward",
            kinfer=args.kinfer,
            robot=args.robot,
            out=args.out,
            seconds=0.0,          # overwritten inside run_factory
        ),
        cmd_factory=lambda: PrecomputedInputState([cmd(0.0)]),
        make_commands=make_commands,
    )


# python -m kinfer_evals.evals.walk_forward tests/assets/kinfer_files/walk_jun22.kinfer kbot-headless
if __name__ == "__main__":
    colorlogging.configure()

    p = argparse.ArgumentParser()
    p.add_argument("kinfer", type=Path)
    p.add_argument("robot")
    p.add_argument("--out", type=Path, default=Path("runs/walk_forward"))
    asyncio.run(_main(p.parse_args()))
