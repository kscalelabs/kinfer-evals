"""Entry point for running evals.

CLI driver:  kinfer-eval  <policy>  <robot>  <eval-name>  [--out] [--render]
"""

import argparse
import asyncio
from pathlib import Path

import colorlogging
from kmotions.motions import MOTIONS

from kinfer_evals.core.eval_engine import run_eval
from kinfer_evals.core.eval_types import RunArgs


def main() -> None:
    colorlogging.configure()

    parser = argparse.ArgumentParser(prog="kinfer-eval")
    parser.add_argument("kinfer", type=Path)
    parser.add_argument("robot")
    parser.add_argument("motion", choices=sorted(MOTIONS.keys()), help="Motion name from kmotions")
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
    print(f"[kinfer-eval] Parsed arguments:")
    print(f"  kinfer: {ns.kinfer}")
    print(f"  robot: {ns.robot}")
    print(f"  motion: {ns.motion}")
    print(f"  out: {ns.out}")
    print(f"  render: {ns.render}")
    print(f"  local_model_dir: {ns.local_model_dir}")
    
    args = RunArgs(
        ns.motion,
        ns.kinfer,
        ns.robot,
        ns.out,
        ns.render,
        local_model_dir=ns.local_model_dir,
        command_type=ns.command_type,
    )
    print(f"[kinfer-eval] Starting eval run for motion '{ns.motion}'")
    url = asyncio.run(run_eval(ns.motion, args))
    if url:
        print(f"[kinfer-eval] ✅ Notion URL: {url}")
        print(url)


if __name__ == "__main__":
    main()
