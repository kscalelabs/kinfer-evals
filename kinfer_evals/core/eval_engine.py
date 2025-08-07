"""Runs the eval, then processes, saves and publishes the results."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from kinfer.rust_bindings import PyModelRunner
from kinfer_sim.provider import ModelProvider
from kinfer_sim.simulator import MujocoSimulator
from kmv.app.viewer import DefaultMujocoViewer
from kmv.utils.logging import VideoWriter

from kinfer_evals.core import metrics
from kinfer_evals.core.eval_types import PrecomputedInputState, RunArgs
from kinfer_evals.core.eval_utils import load_sim_and_runner
from kinfer_evals.core.recorder import Recorder
from kinfer_evals.publishers.notion import push_summary

if TYPE_CHECKING:
    from kinfer_evals.evals import CommandMaker

logger = logging.getLogger(__name__)


async def run_episode(
    sim: MujocoSimulator,
    runner: PyModelRunner,
    seconds: float,
    outdir: Path,
    provider: ModelProvider | None = None,
    run_info: dict | None = None,
    *,
    record_video: bool = True,
) -> None:
    """Physics → inference → actuation loop, plots and optional video."""
    outdir.mkdir(parents=True, exist_ok=True)

    rec = Recorder(outdir / "episode.h5", sim._model)

    video_writer = None
    decim = 1

    if record_video and isinstance(sim._viewer, DefaultMujocoViewer):
        fps_target = 30
        decim = max(1, int(round(sim._control_frequency / fps_target)))
        video_writer = VideoWriter(outdir / "video.mp4", fps=fps_target)
    elif record_video and not isinstance(sim._viewer, DefaultMujocoViewer):
        logger.warning("Cannot record video: QtViewer is active; run without --render")

    carry = runner.init()
    dt_ctrl = 1.0 / sim._control_frequency

    n_ctrl_steps = int(round(seconds * sim._control_frequency))
    step_idx = 0

    try:
        while step_idx < n_ctrl_steps:
            # Step physics
            for _ in range(sim.sim_decimation):
                await sim.step()

            # Advance command index if we're using a PrecomputedInputState
            if provider and hasattr(provider.keyboard_state, "step"):
                provider.keyboard_state.step()

            # Get commands
            cmd_vx_body = cmd_vy_body = cmd_omega = 0.0
            if provider is not None:
                cmd_vx_body, cmd_vy_body = provider.keyboard_state.value[:2]
                cmd_omega = provider.keyboard_state.value[2] if len(provider.keyboard_state.value) > 2 else 0.0

            # Inference
            out, carry = runner.step(carry)
            runner.take_action(out)

            # If saving video, append a frame
            if video_writer and step_idx % decim == 0:
                video_writer.append(sim.read_pixels())

            # Record data including commands
            rec.append(sim._data, step_idx * dt_ctrl, (cmd_vx_body, cmd_vy_body, cmd_omega))
            await asyncio.sleep(0)

            step_idx += 1

    finally:
        await sim.close()
        if video_writer:
            video_writer.close()
        rec.close()


def build_run_info(args: RunArgs, timestamp: str, outdir: Path, duration_seconds: float) -> dict:
    """Save metadata about this run for tracking purposes."""
    run_info = {
        "timestamp": timestamp,
        "eval_name": args.eval_name,
        "kinfer_file": str(args.kinfer.absolute()),
        "robot": args.robot,
        "duration_seconds": duration_seconds,
        "output_directory": str(outdir.absolute()),
    }

    return run_info


async def run_eval(
    make_cmds: "CommandMaker",
    eval_name: str,
    args: RunArgs,
) -> None:
    """Common driver used by every eval.

    • spin up sim/runner with a dummy keyboard state
    • build the full command list upfront
    • wrap it in PrecomputedInputState
    • run the episode & save artifacts
    """
    sim, runner, provider = await load_sim_and_runner(
        args.kinfer,
        args.robot,
        cmd_factory=lambda: PrecomputedInputState([[0.0, 0.0, 0.0]]),
        render=args.render,
        free_camera=False,
    )

    freq = sim._control_frequency
    commands = make_cmds(freq)
    provider.keyboard_state = PrecomputedInputState(commands)
    duration_seconds = len(commands) / freq

    # Create timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = args.out / eval_name / timestamp

    # Save & keep run metadata
    run_info = build_run_info(args, timestamp, outdir, duration_seconds)

    await run_episode(
        sim,
        runner,
        duration_seconds,
        outdir,
        provider,
        run_info,
        record_video=not args.render,
    )

    # Run post-processing metrics
    run_meta = {
        "kinfer": str(args.kinfer.absolute()),
        "robot": args.robot,
        "eval_name": eval_name,
        "timestamp": timestamp,
        "outdir": str(outdir.absolute()),
    }

    summary = metrics.run(outdir / "episode.h5", outdir, run_meta)

    # Save combined summary
    combined = {**run_info, **summary}
    (outdir / "run_summary.json").write_text(json.dumps(combined, indent=2))
    logger.info("Saved combined summary to %s", outdir / "run_summary.json")

    try:
        vid = outdir / "video.mp4"
        artifacts = ([vid] if vid.exists() else []) + sorted(outdir.glob("*.png"))
        url = push_summary(combined, artifacts)
        logger.info("Logged run to Notion: %s", url)
    except Exception as exc:
        logger.warning("Failed to push results to Notion: %s", exc)
