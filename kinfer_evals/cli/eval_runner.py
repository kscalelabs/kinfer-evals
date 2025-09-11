"""Entry point for running evals.

CLI driver:  kinfer-eval  <policy>  <robot>  <eval-name>  [--out] [--render]
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
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the eval in a window using the K-Scale Mujoco viewer",
    )
    parser.add_argument(
        "--local-model-dir",
        type=Path,
        default=None,
        help="Path to a local robot URDF/MJCF directory to use instead of downloading",
    )
    parser.add_argument(
        "--command-type",
        type=str,
        default=None,
        help="Passthrough command type (e.g., 'unified'); reserved for future evals",
    )

    ns = parser.parse_args()
    make = REGISTRY[ns.eval]
    args = RunArgs(
        ns.eval,
        ns.kinfer,
        ns.robot,
        ns.out,
        ns.render,
        local_model_dir=ns.local_model_dir,
        command_type=ns.command_type,
    )
    url = asyncio.run(run_eval(make, ns.eval, args))
    if url:
        print(url)


if __name__ == "__main__":
    main()
