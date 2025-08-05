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

    ns = parser.parse_args()
    make = REGISTRY[ns.eval]  # the registered function
    args = RunArgs(ns.eval, ns.kinfer, ns.robot, ns.out)
    asyncio.run(run_eval(make, ns.eval, args))


if __name__ == "__main__":
    main()
