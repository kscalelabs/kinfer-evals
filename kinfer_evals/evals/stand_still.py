"""Stand still evaluation."""

import argparse
import asyncio
from pathlib import Path

from kinfer_sim.provider import ExpandedControlVectorInputState, InputState

from kinfer_evals.evals.common import load_sim_and_runner, run_episode, save_json


def cmd_factory() -> InputState:
    return ExpandedControlVectorInputState()  # all zeros → stand still


async def _main(args: argparse.Namespace) -> None:
    sim, runner, provider = await load_sim_and_runner(args.kinfer, args.robot, cmd_factory)
    log = await run_episode(sim, runner, args.seconds, provider)
    save_json(log, args.out)


# python -m kinfer_evals.evals.stand_still tests/assets/kinfer_files/walk_jun22.kinfer kbot-headless
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("kinfer", type=Path)
    p.add_argument("robot")
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--out", type=Path, default=Path("runs/stand_still"))
    asyncio.run(_main(p.parse_args()))
