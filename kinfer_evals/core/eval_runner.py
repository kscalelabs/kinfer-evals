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

_p = argparse.ArgumentParser(prog="kinfer-eval")
_p.add_argument("kinfer", type=Path)
_p.add_argument("robot")
_p.add_argument("eval", choices=sorted(REGISTRY.keys()))
_p.add_argument("--out", type=Path, default=Path("runs"))


def main() -> None:
    colorlogging.configure()

    ns = _p.parse_args()
    make = REGISTRY[ns.eval]  # the registered function
    args = RunArgs(ns.eval, ns.kinfer, ns.robot, ns.out)
    asyncio.run(run_eval(make, ns.eval, args))


if __name__ == "__main__":
    main()
