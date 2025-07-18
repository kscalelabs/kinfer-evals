"""Walk forward evaluation."""

import argparse
import asyncio
import time
from pathlib import Path

from kinfer_evals.evals.common import (  # shared helpers
    PrecomputedInputState,
    cmd,  # just added above
    load_sim_and_runner,
    run_episode,
    save_json,
)


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
    # We must know the control frequency before building the list,
    # so we spin up the simulator first (then build, then replace its state).
    sim, runner, provider = await load_sim_and_runner(
        args.kinfer,
        args.robot,
        cmd_factory=lambda: PrecomputedInputState([cmd(0.0)]),  # placeholder
    )

    freq = sim._control_frequency  # e.g. 1 000 Hz
    commands = make_commands(freq)

    # swap in the real pre-computed state (keep provider alive)
    provider.keyboard_state = PrecomputedInputState(commands)

    # episode length = command_count / freq seconds
    seconds = len(commands) / freq
    outdir = args.out / time.strftime("%Y%m%d-%H%M%S")
    log = await run_episode(sim, runner, seconds, outdir, provider)

    # also dump the command list for inspection
    save_json(log, outdir, "walk_log.json")
    save_json(commands, outdir, "commands.json")


# python -m kinfer_evals.evals.walk_forward tests/assets/kinfer_files/walk_jun22.kinfer kbot-headless
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("kinfer", type=Path)
    p.add_argument("robot")
    p.add_argument("--out", type=Path, default=Path("runs/walk_forward"))
    asyncio.run(_main(p.parse_args()))
