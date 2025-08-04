"""Stand still evaluation."""

import argparse
import asyncio
import time
from pathlib import Path

import colorlogging
from kinfer_sim.provider import ControlVectorInputState, InputState

from kinfer_evals.core.eval_runtime import run_factory
from kinfer_evals.core.types import RunArgs


def cmd_factory() -> InputState:
    return ControlVectorInputState()  # all zeros → stand still


async def _main(args: argparse.Namespace) -> None:
    await run_factory(
        RunArgs(
            name="stand_still",
            kinfer=args.kinfer,
            robot=args.robot,
            out=args.out,
            seconds=args.seconds,
        ),
        cmd_factory=cmd_factory,
    )


# python -m kinfer_evals.evals.stand_still tests/assets/kinfer_files/walk_jun22.kinfer kbot-headless
if __name__ == "__main__":
    colorlogging.configure()

    p = argparse.ArgumentParser()
    p.add_argument("kinfer", type=Path)
    p.add_argument("robot")
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--out", type=Path, default=Path("runs/stand_still"))
    asyncio.run(_main(p.parse_args()))
