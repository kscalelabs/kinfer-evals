"""Runs the eval, then processes, saves and publishes the results."""

import json
import logging
from datetime import datetime
from pathlib import Path

from kmv.app.viewer import DefaultMujocoViewer

from kmotions.motions import MOTIONS

from kinfer_evals.artifacts.plots import render_artifacts
from kinfer_evals.core.eval_types import RunArgs, RunInfo
from kinfer_evals.core.eval_utils import load_joint_names, load_sim_and_runner
from kinfer_evals.core.io_h5 import EpisodeReader
from kinfer_evals.core.metrics import compute_metrics
from kinfer_evals.core.rollout import EpisodeRollout, H5Sink, KinferLogSink, StepSink, VideoSink
from kinfer_evals.publishers.notion import push_summary

logger = logging.getLogger(__name__)


def _build_run_info(args: RunArgs, timestamp: str, outdir: Path) -> RunInfo:
    """Build a RunInfo dictionary with the given arguments."""
    return {
        "timestamp": timestamp,
        "eval_name": args.eval_name,
        "kinfer_file": str(args.kinfer.absolute()),
        "robot": args.robot,
        "outdir": str(outdir.absolute()),
    }


async def _run_episode_to_h5(
    motion_name: str,
    args: RunArgs,
    outdir: Path,
    run_info: RunInfo,
) -> Path:
    """Spin up sim & runner, play motion, and record to HDF5."""
    print(f"[eval-engine] Loading motion factory for '{motion_name}'")
    # Get the motion factory from kmotions
    motion_factory = MOTIONS[motion_name]
    
    print(f"[eval-engine] Loading joint names from kinfer file: {args.kinfer}")
    # Load joint names from kinfer file
    joint_names = load_joint_names(args.kinfer)
    print(f"[eval-engine] Loaded {len(joint_names)} joint names")
    
    print(f"[eval-engine] Loading simulator and runner for robot '{args.robot}'")
    sim, runner, command_provider, provider = await load_sim_and_runner(
        args.kinfer,
        args.robot,
        motion_factory=motion_factory,
        render=args.render,
        free_camera=False,
        local_model_dir=args.local_model_dir,
    )
    print(f"[eval-engine] Simulator and runner initialized successfully")
    
    # Run until motion completes (returns None) - determined by rollout
    duration_seconds = None  # Will run until motion completes

    outdir.mkdir(parents=True, exist_ok=True)
    h5_path = outdir / "episode.h5"

    # Prepare run name: kinfer_motion_timestamp
    kinfer_stem = args.kinfer.stem
    timestamp = run_info["timestamp"]
    run_name = f"{kinfer_stem}_{motion_name}_{timestamp}"
    print(f"[eval-engine] Run name: {run_name}")
    print(f"[eval-engine] Output directory: {outdir}")

    # sinks: HDF5 (always) + kinfer logs (always) + video if allowed
    print(f"[eval-engine] Setting up data sinks (HDF5, kinfer logs, video)")
    sinks: list[StepSink] = [
        H5Sink(h5_path, sim, run_info=run_info),
        KinferLogSink(outdir, run_name, joint_names),
    ]
    want_video = not args.render
    if want_video:
        try:
            # Only supported with GLFW viewer
            if isinstance(sim._viewer, DefaultMujocoViewer):
                sinks.append(VideoSink(outdir / "video.mp4", sim))
                print(f"[eval-engine] Video recording enabled")
            else:
                logger.warning("Cannot record video: QtViewer is active; run without --render")
        except Exception as exc:
            logger.warning("Failed to init VideoSink: %s", exc)
    else:
        print(f"[eval-engine] Video recording disabled (render mode active)")

    print(f"[eval-engine] Starting episode rollout...")
    rollout = EpisodeRollout(sim, runner, command_provider, provider, sinks, joint_names)
    await rollout.run(duration_seconds)
    print(f"[eval-engine] Episode rollout completed, saved to {h5_path}")
    return h5_path


async def run_eval(
    motion_name: str,
    args: RunArgs,
) -> str | None:
    """Top-level orchestrator.

    1) run episode → episode.h5
    2) read episode → compute metrics
    3) render artifacts
    4) (optional) publish to Notion
    """
    print(f"\n{'='*60}")
    print(f"[eval-engine] Starting kinfer-evals evaluation")
    print(f"{'='*60}")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    kinfer_stem = args.kinfer.stem
    # Include kinfer name in directory structure: motion_name/kinfer_timestamp
    outdir = args.out / motion_name / f"{kinfer_stem}_{timestamp}"
    run_info = _build_run_info(args, timestamp, outdir)
    print(f"[eval-engine] Timestamp: {timestamp}")
    print(f"[eval-engine] Kinfer: {kinfer_stem}")
    print(f"[eval-engine] Output: {outdir}")

    print(f"\n[eval-engine] PHASE 1: Running episode and recording to HDF5")
    h5_path = await _run_episode_to_h5(motion_name, args, outdir, run_info)

    # Read & compute
    print(f"\n[eval-engine] PHASE 2: Reading episode data and computing metrics")
    episode = EpisodeReader.read(h5_path)
    print(f"[eval-engine] Episode data loaded: {len(episode.time)} timesteps")
    metrics = compute_metrics(episode)
    print(f"[eval-engine] Computed {len(metrics)} metrics")

    # Render plots
    print(f"\n[eval-engine] PHASE 3: Rendering artifacts (plots)")
    artifacts = render_artifacts(episode, run_info, outdir)
    print(f"[eval-engine] Generated {len(artifacts)} artifact files")

    # Also include the recorded video, if present
    video_path = outdir / "video.mp4"
    if video_path.exists() and video_path.is_file():
        artifacts.append(video_path)
        print(f"[eval-engine] Including video artifact: {video_path}")
        logger.info("Including video artifact for Notion upload: %s", video_path)
    else:
        print(f"[eval-engine] No video artifact found (skipping)")
        logger.info("No video artifact found at %s (skipping).", video_path)

    # Save combined summary
    print(f"\n[eval-engine] PHASE 4: Saving summary")
    notion_url: str | None = None
    combined = {**run_info, **metrics, "notion_url": notion_url or ""}
    # Optional metadata
    if args.local_model_dir is not None:
        combined["local_model_dir"] = str(Path(args.local_model_dir).absolute())
    if getattr(args, "command_type", None):
        combined["command_type"] = args.command_type
    summary_path = outdir / "run_summary.json"
    summary_path.write_text(json.dumps(combined, indent=2))
    print(f"[eval-engine] Saved summary to: {summary_path}")
    logger.info("Saved combined summary to %s", summary_path)

    # Publish
    print(f"\n[eval-engine] PHASE 5: Publishing to Notion")
    try:
        notion_url = push_summary(combined, artifacts)
        print(f"[eval-engine] ✅ Published to Notion: {notion_url}")
        logger.info("Logged run to Notion: %s", notion_url)
        try:
            if notion_url:
                (outdir / "notion_url.txt").write_text(notion_url + "\n")
                # Update summary with actual notion_url
                combined["notion_url"] = notion_url
                summary_path.write_text(json.dumps(combined, indent=2))
                print(f"[eval-engine] Saved Notion URL to: {outdir / 'notion_url.txt'}")
        except Exception as exc:
            logger.warning("Failed to write notion_url.txt or update summary: %s", exc)
    except Exception as exc:
        print(f"[eval-engine] ⚠️  Failed to push to Notion: {exc}")
        logger.warning("Failed to push results to Notion: %s", exc)

    print(f"\n{'='*60}")
    print(f"[eval-engine] Evaluation complete!")
    print(f"{'='*60}\n")
    return notion_url
