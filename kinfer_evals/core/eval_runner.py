"""Entry point for running evals.

CLI driver:  kinfer-eval  <policy>  <robot>  <eval-name>  [--seconds] [--out]
"""

import argparse
import asyncio
from pathlib import Path

import colorlogging

from kinfer_evals.core.eval_engine import run_eval
from kinfer_evals.core.eval_types import RunArgs
from kinfer_evals.evals import REGISTRY


def main() -> None:
    colorlogging.configure()

    parser = argparse.ArgumentParser(prog="kinfer-eval")
    parser.add_argument("kinfer", type=Path)
    parser.add_argument("robot")
    parser.add_argument("eval", choices=sorted(REGISTRY.keys()))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--author", type=str, default="")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the eval in a window using the K-Scale Mujoco viewer",
    )

    ns = parser.parse_args()
    make = REGISTRY[ns.eval]  # the registered function
    args = RunArgs(eval_name=ns.eval, kinfer=ns.kinfer, robot=ns.robot, out=ns.out, author=ns.author, render=ns.render)
    asyncio.run(run_eval(make, ns.eval, args))


if __name__ == "__main__":
    main()
