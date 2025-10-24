"""Shared utilities for running evals."""

import asyncio
import json
import logging
import tarfile
from pathlib import Path
from typing import Callable, Optional, Union, cast

import numpy as np
from kinfer.rust_bindings import PyModelRunner
from kinfer_sim.provider import ModelProvider
from kinfer_sim.server import find_mjcf, get_model_metadata
from kinfer_sim.simulator import MujocoSimulator
from kscale import K
from kscale.web.gen.api import RobotURDFMetadataOutput

from kinfer_evals.core.eval_types import CommandFactory, CommandProvider

logger = logging.getLogger(__name__)


def load_command_names(kinfer_path: Path) -> list[str]:
    """Load command_names from kinfer metadata.json.
    
    Args:
        kinfer_path: Path to the .kinfer file
        
    Returns:
        List of command names
    """
    try:
        with tarfile.open(kinfer_path, "r:gz") as tar:
            metadata_file = tar.extractfile("metadata.json")
            if metadata_file is None:
                raise FileNotFoundError("'metadata.json' not found in archive")
            metadata = json.loads(metadata_file.read().decode("utf-8"))
    except (tarfile.TarError, FileNotFoundError) as exc:
        raise ValueError(f"Could not read metadata from {kinfer_path}: {exc}") from exc

    command_names = metadata.get("command_names", None)
    if not command_names:
        raise ValueError(f"'command_names' missing in metadata for {kinfer_path}")

    logger.info("Loaded %d command names from model metadata", len(command_names))
    return list(command_names)


def load_joint_names(kinfer_path: Path) -> list[str]:
    """Load joint_names from kinfer metadata.json.
    
    Args:
        kinfer_path: Path to the .kinfer file
        
    Returns:
        List of joint names
    """
    try:
        with tarfile.open(kinfer_path, "r:gz") as tar:
            metadata_file = tar.extractfile("metadata.json")
            if metadata_file is None:
                raise FileNotFoundError("'metadata.json' not found in archive")
            metadata = json.loads(metadata_file.read().decode("utf-8"))
    except (tarfile.TarError, FileNotFoundError) as exc:
        raise ValueError(f"Could not read metadata from {kinfer_path}: {exc}") from exc

    joint_names = metadata.get("joint_names", None)
    if not joint_names:
        raise ValueError(f"'joint_names' missing in metadata for {kinfer_path}")

    logger.info("Loaded %d joint names from model metadata", len(joint_names))
    return list(joint_names)


def get_yaw_from_quaternion(quat: np.ndarray) -> float:
    """Extract yaw angle from quaternion data."""
    return float(
        np.arctan2(
            2 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
        )
    )


def default_sim(
    mjcf: Path,
    meta: RobotURDFMetadataOutput,
    *,
    dt: float = 1e-4,
    render: bool = False,
    free_camera: bool = False,
) -> MujocoSimulator:
    return MujocoSimulator(
        model_path=mjcf,
        model_metadata=meta,
        dt=dt,
        render_mode="window" if render else "offscreen",
        start_height=1.1,
        free_camera=free_camera,
    )


async def load_sim_and_runner(
    kinfer: Path,
    robot: str,
    motion_factory: CommandFactory,
    *,
    make_sim: Callable[..., MujocoSimulator] = default_sim,
    **sim_kwargs: object,
) -> tuple[MujocoSimulator, PyModelRunner, CommandProvider, ModelProvider]:
    """Shared download + construction logic."""
    # Load command_names from kinfer metadata
    command_names = load_command_names(kinfer)
    logger.info("Loaded %d command names from model metadata", len(command_names))
    
    # Optional overrides via sim_kwargs from RunArgs: local_model_dir
    local_model_dir_obj: Optional[Union[str, Path]] = cast(
        Optional[Union[str, Path]], sim_kwargs.pop("local_model_dir", None)
    )
    if local_model_dir_obj is not None:
        model_dir = Path(local_model_dir_obj)
        async with K() as api:
            meta = await get_model_metadata(api, robot)
    else:
        async with K() as api:
            model_dir, meta = await asyncio.gather(
                api.download_and_extract_urdf(robot, cache=True),
                get_model_metadata(api, robot),
            )

    mjcf = find_mjcf(model_dir)
    sim = make_sim(mjcf, meta, **sim_kwargs)
    
    # Create motion with dt based on sim control frequency
    dt = 1.0 / sim._control_frequency
    motion = motion_factory(dt)
    
    # Create command provider with motion and command_names
    command_provider = CommandProvider(motion, command_names)
    # ModelProvider now takes command_provider instead of keyboard_state
    provider = ModelProvider(sim, command_provider=command_provider)
    runner = PyModelRunner(str(kinfer), provider, pre_fetch_time_ms=None)
    return sim, runner, command_provider, provider
