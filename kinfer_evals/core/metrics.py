"""Compute numeric metrics and plots from an episode.h5 file."""

from pathlib import Path

import h5py
import numpy as np

from kinfer_evals.artifacts.plots import (
    _plot_xy_trajectory,
    plot_accel,
    plot_contact_force_per_body,  # NEW
    plot_heading,
    plot_omega,
    plot_velocity,
)
from kinfer_evals.core.eval_utils import get_yaw_from_quaternion
from kinfer_evals.reference_state import ReferenceStateTracker


def _body_frame_vel(qvel: np.ndarray, yaw_series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vx_w, vy_w = qvel[:, 0], qvel[:, 1]
    c, s = np.cos(yaw_series), np.sin(yaw_series)
    vx_b = c * vx_w + s * vy_w
    vy_b = -s * vx_w + c * vy_w
    return vx_b, vy_b


def run(h5: Path, outdir: Path, run_meta: dict[str, object]) -> dict[str, float]:
    """Post-process *h5* → plots + metrics.

    Returns the numeric summary (to be merged with run_meta then json-dumped).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5, "r") as f:
        t = f["time"][:]  # (T,)
        qpos = f["qpos"][:]  # (T, nq)
        qvel = f["qvel"][:]  # (T, nv)
        cmd_vel = f["cmd_vel"][:]  # (T, 3) - [vx, vy, omega]
        ncon = f["contact_count"][:]  # (T,)
        fmag = f["contact_force_mag"][:]  # (T,)
        # -------- per-body contact forces --------------------------- #
        body_names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["body_names"][:]]
        contact_body = f["contact_body"][:]  # ragged
        wrench_flat = f["contact_wrench"][:]  # ragged

    nb = len(body_names)
    per_body = np.zeros((nb, len(t)), dtype=np.float32)  # (nb, T)
    for step, (pairs, flat) in enumerate(zip(contact_body, wrench_flat)):
        if pairs.size == 0:
            continue
        forces = flat.reshape(-1, 6)[:, :3]  # Fx,Fy,Fz
        mags = np.linalg.norm(forces, axis=1)  # |F|
        ids = pairs.reshape(-1, 2)  # (ncon,2)
        for (a, b), mag in zip(ids, mags):
            per_body[a, step] += mag
            per_body[b, step] += mag

    dt = np.mean(np.diff(t))

    # ---------- actual yaw, ω, body-frame velocity -------------------- #
    # MuJoCo free-joint qpos: [x y z  qw qx qy qz]
    quat = qpos[:, 3:7]  # (T,4)  (w,x,y,z)
    yaw_series = np.array([get_yaw_from_quaternion(q) for q in quat], dtype=np.float32)

    vx_b, vy_b = _body_frame_vel(qvel, yaw_series)

    # ------------ extract commands from h5 data --------------------------- #
    cmd_vx = cmd_vel[:, 0]
    cmd_vy = cmd_vel[:, 1]
    cmd_omega = cmd_vel[:, 2]

    err_vx = vx_b - cmd_vx
    err_vy = vy_b - cmd_vy

    # acceleration
    ax_b = np.diff(vx_b) / dt
    ay_b = np.diff(vy_b) / dt
    cmd_ax = np.diff(cmd_vx) / dt
    cmd_ay = np.diff(cmd_vy) / dt
    err_ax, err_ay = ax_b - cmd_ax, ay_b - cmd_ay
    cmd_am = np.hypot(cmd_ax, cmd_ay)
    act_am = np.hypot(ax_b, ay_b)
    err_am = act_am - cmd_am

    # ----------------- heading & ω errors ----------------------------- #
    # Reference heading comes from command integration
    tracker = ReferenceStateTracker()
    ref_yaw, ref_x, ref_y = [], [], []
    for i in range(len(t)):
        tracker.step((cmd_vx[i], cmd_vy[i]), cmd_omega[i], dt)
        ref_yaw.append(tracker.yaw)
        ref_x.append(tracker.pos_x)
        ref_y.append(tracker.pos_y)

    ref_yaw = np.unwrap(ref_yaw)
    yaw_series_u = np.unwrap(yaw_series)
    yaw_err = yaw_series_u - ref_yaw

    act_omega = np.diff(yaw_series_u) / dt
    err_om = act_omega - cmd_omega[:-1]

    # ------------ numeric summary --------------------------------------- #
    def mae(x: np.ndarray) -> float:
        return float(np.mean(np.abs(x)))

    def rmse(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x))))

    summary = {
        # velocity
        "mae_vel_x": mae(err_vx),
        "mae_vel_y": mae(err_vy),
        "rmse_vel_x": rmse(err_vx),
        "rmse_vel_y": rmse(err_vy),
        "vel_samples": int(len(err_vx)),
        # acceleration
        "mae_accel_x": mae(err_ax),
        "mae_accel_y": mae(err_ay),
        "mae_accel_mag": mae(err_am),
        "rmse_accel_x": rmse(err_ax),
        "rmse_accel_y": rmse(err_ay),
        "rmse_accel_mag": rmse(err_am),
        # heading / ω
        "mae_heading": mae(yaw_err),
        "rmse_heading": rmse(yaw_err),
        "mae_omega": mae(err_om),
        "rmse_omega": rmse(err_om),
        "omega_samples": int(len(err_om)),
    }

    time_s = t
    plot_velocity(time_s, cmd_vx, vx_b, err_vx, "x", plots_dir, run_meta)
    plot_velocity(time_s, cmd_vy, vy_b, err_vy, "y", plots_dir, run_meta)

    plot_accel(time_s[1:], cmd_ax, ax_b, err_ax, "x", plots_dir, run_meta)
    plot_accel(time_s[1:], cmd_ay, ay_b, err_ay, "y", plots_dir, run_meta)
    plot_accel(time_s[1:], cmd_am, act_am, err_am, "mag", plots_dir, run_meta)

    plot_heading(time_s, ref_yaw, yaw_series_u, yaw_err, plots_dir, run_meta)
    plot_omega(time_s[1:], cmd_omega[:-1], act_omega, err_om, plots_dir, run_meta)

    # ----------- contact plots ---------------------------------------- #
    from kinfer_evals.artifacts.plots import (
        plot_contact_count,
        plot_contact_force_mag,
    )

    plot_contact_count(time_s, ncon, plots_dir, run_meta)
    plot_contact_force_mag(time_s, fmag, plots_dir, run_meta)

    # per-body contact-force lines
    plot_contact_force_per_body(time_s, per_body, body_names, plots_dir, run_meta)

    _plot_xy_trajectory(ref_x, ref_y, qpos[:, 0], qpos[:, 1], plots_dir, run_meta)

    return summary
