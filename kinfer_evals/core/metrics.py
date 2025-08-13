"""Pure numeric metrics from EpisodeData."""

import numpy as np

from kinfer_evals.core.eval_types import EpisodeData
from kinfer_evals.reference_state import ReferenceStateTracker


def body_frame_vel(qvel: np.ndarray, yaw_series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate world-frame (vx, vy) into body frame given yaw."""
    vx_w, vy_w = qvel[:, 0], qvel[:, 1]
    c, s = np.cos(yaw_series), np.sin(yaw_series)
    vx_b = c * vx_w + s * vy_w
    vy_b = -s * vx_w + c * vy_w
    return vx_b.astype(np.float32), vy_b.astype(np.float32)


def yaw_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """Extract yaw from quaternions (w,x,y,z) per row."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2)).astype(np.float32)


def compute_metrics(ep: EpisodeData) -> dict[str, float]:
    """Compute numeric tracking metrics from an episode.

    Overview:
      - Derive dt and world yaw (from qpos quaternions).
      - Rotate world-frame (vx, vy) → body-frame (vx, vy).
      - Compute velocity & acceleration signals and their errors vs. commands.
      - Integrate commanded [vx, vy, ω] → reference yaw; compute yaw/omega errors.
      - Return MAE/RMSE summary as a flat dict of floats.

    Returns keys:
      mae_vel_x/mae_vel_y, rmse_vel_x/rmse_vel_y, vel_samples,
      mae_accel_x/mae_accel_y/mae_accel_mag, rmse_* counterparts,
      mae_heading/rmse_heading, mae_omega/rmse_omega, omega_samples.
    """
    time_s = ep.time
    dt = ep.dt if ep.dt > 0 else (float(np.mean(np.diff(time_s))) if time_s.size >= 2 else 0.0)

    # MuJoCo free-joint qpos: [x y z  qw qx qy qz]
    quat_wxyz = ep.qpos[:, 3:7]
    yaw_world = yaw_from_quat_wxyz(quat_wxyz)
    yaw_world_unwrapped = np.unwrap(yaw_world)

    # Rotate world-frame velocities into body frame
    body_vx, body_vy = body_frame_vel(ep.qvel, yaw_world)

    # Commanded signals
    command_vx = ep.cmd_vel[:, 0]
    command_vy = ep.cmd_vel[:, 1]
    command_omega = ep.cmd_vel[:, 2]

    # Velocity errors
    error_vx = body_vx - command_vx
    error_vy = body_vy - command_vy

    # Accelerations (finite differences) and errors
    body_ax = np.diff(body_vx) / dt
    body_ay = np.diff(body_vy) / dt
    command_ax = np.diff(command_vx) / dt
    command_ay = np.diff(command_vy) / dt
    error_ax = body_ax - command_ax
    error_ay = body_ay - command_ay
    command_acc_mag = np.hypot(command_ax, command_ay)
    body_acc_mag = np.hypot(body_ax, body_ay)
    error_acc_mag = body_acc_mag - command_acc_mag

    # Reference yaw by integrating commands
    tracker = ReferenceStateTracker()
    ref_yaw_list: list[float] = []
    for i in range(len(time_s)):
        tracker.step((float(command_vx[i]), float(command_vy[i])), float(command_omega[i]), dt)
        ref_yaw_list.append(tracker.yaw)

    ref_yaw = np.unwrap(np.asarray(ref_yaw_list, dtype=np.float32))
    yaw_error = yaw_world_unwrapped - ref_yaw

    # Omega errors
    omega_actual = np.diff(yaw_world_unwrapped) / dt
    omega_error = omega_actual - command_omega[:-1]

    def mae(x: np.ndarray) -> float:
        return float(np.mean(np.abs(x))) if x.size else 0.0

    def rmse(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0

    summary = {
        # velocity
        "mae_vel_x": mae(error_vx),
        "mae_vel_y": mae(error_vy),
        "rmse_vel_x": rmse(error_vx),
        "rmse_vel_y": rmse(error_vy),
        "vel_samples": int(error_vx.size),
        # acceleration
        "mae_accel_x": mae(error_ax),
        "mae_accel_y": mae(error_ay),
        "mae_accel_mag": mae(error_acc_mag),
        "rmse_accel_x": rmse(error_ax),
        "rmse_accel_y": rmse(error_ay),
        "rmse_accel_mag": rmse(error_acc_mag),
        # heading / omega
        "mae_heading": mae(yaw_error),
        "rmse_heading": rmse(yaw_error),
        "mae_omega": mae(omega_error),
        "rmse_omega": rmse(omega_error),
        "omega_samples": int(omega_error.size),
    }
    return summary
