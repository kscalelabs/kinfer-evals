"""Compute numeric metrics and plots from an episode.h5 file."""

import logging
from pathlib import Path
from typing import Sequence, cast

import h5py
import numpy as np
from kinfer_sim.server import load_joint_names

from kinfer_evals.artifacts.plots import (
    _plot_xy_trajectory,
    plot_accel,
    plot_actions,
    plot_contact_count,
    plot_contact_force_mag,
    plot_contact_force_per_body,
    plot_double_support_intervals,
    plot_gait_frequency,
    plot_heading,
    plot_input_series,
    plot_n_feet_in_contact,
    plot_omega,
    plot_velocity,
)
from kinfer_evals.core.eval_utils import get_yaw_from_quaternion
from kinfer_evals.reference_state import ReferenceStateTracker

logger = logging.getLogger(__name__)


def _body_frame_vel(qvel: np.ndarray, yaw_series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vx_w, vy_w = qvel[:, 0], qvel[:, 1]
    c, s = np.cos(yaw_series), np.sin(yaw_series)
    vx_b = c * vx_w + s * vy_w
    vy_b = -s * vx_w + c * vy_w
    return vx_b, vy_b


def compute_gait_frequency(foot_con: Sequence[set], dt: float, cmd_vel: np.ndarray) -> dict:
    foot_ids = set().union(*foot_con)
    foot_states = {foot_id: np.array([foot_id in contact for contact in foot_con]) for foot_id in foot_ids}
    strikes = {foot_id: np.where((arr[1:] == 1) & (arr[:-1] == 0))[0] for foot_id, arr in foot_states.items()}

    moving_mask = np.any(np.abs(cmd_vel) > 1e-6, axis=1)
    gait_periods = {}
    for foot_id, strike_indices in strikes.items():
        intervals = np.diff(strike_indices) * dt
        for t, interval in zip(strike_indices, intervals):
            if moving_mask[t]:
                gait_periods[t] = interval

    gait_frequencies = {k: 1 / v for k, v in gait_periods.items()}
    return gait_frequencies


def compute_double_support_intervals(n_feet_con: Sequence[int], dt: float, cmd_vel: np.ndarray) -> dict:
    double_support_mask = [i == 2 for i in n_feet_con]

    carry = 0
    double_support_intervals = {}
    for i in double_support_mask:
        if i:
            carry += 1
        elif carry > 0:
            double_support_intervals[carry] = carry * dt
            carry = 0

    return double_support_intervals


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
        bcon = f["contact_body"][:]  # (T, 2*ncon)
        fmag = f["contact_force_mag"][:]  # (T,)
        # -------- per-body contact forces --------------------------- #
        body_names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["body_names"][:]]
        contact_body = f["contact_body"][:]  # ragged
        wrench_flat = f["contact_wrench"][:]  # ragged
        actions = f["action_target"][:] if "action_target" in f else None

        # Collect policy input datasets
        input_datasets = {}
        input_keys = [k for k in f.keys() if k.startswith("input_")]
        for key in input_keys:
            name = key[6:]  # strip "input_"
            vals = f[key][:]  # shape (T, N)  or (T,)
            if vals.ndim == 1:
                vals = vals[:, None]
            input_datasets[name] = vals

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

    # ------------ gait metrics --------------------------------------- #
    foot_con = [set(array[array != 0].astype(int)) for array in bcon]
    n_foot_con = [len(c) for c in foot_con]

    # Compute gait metrics
    gait_frequencies = compute_gait_frequency(foot_con, dt, cmd_vel)
    double_support_intervals = compute_double_support_intervals(n_foot_con, dt, cmd_vel)

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

    time_s = t.tolist()  # list[float] → satisfies Sequence[float]
    plot_velocity(time_s, cmd_vx.tolist(), vx_b.tolist(), err_vx.tolist(), "x", plots_dir, run_meta)
    plot_velocity(time_s, cmd_vy.tolist(), vy_b.tolist(), err_vy.tolist(), "y", plots_dir, run_meta)

    plot_accel(time_s[1:], cmd_ax.tolist(), ax_b.tolist(), err_ax.tolist(), "x", plots_dir, run_meta)
    plot_accel(time_s[1:], cmd_ay.tolist(), ay_b.tolist(), err_ay.tolist(), "y", plots_dir, run_meta)
    plot_accel(time_s[1:], cmd_am.tolist(), act_am.tolist(), err_am.tolist(), "mag", plots_dir, run_meta)

    plot_heading(time_s, ref_yaw.tolist(), yaw_series_u.tolist(), yaw_err.tolist(), plots_dir, run_meta)
    plot_omega(time_s[1:], cmd_omega[:-1].tolist(), act_omega.tolist(), err_om.tolist(), plots_dir, run_meta)

    # ----------- action plot --------------------------------------- #
    if actions is not None:
        try:
            joint_names = load_joint_names(Path(cast(str, run_meta["kinfer"])))
        except Exception:
            joint_names = [f"joint_{i}" for i in range(actions.shape[1])]
        plot_actions(time_s, actions.tolist(), joint_names, plots_dir, run_meta)

    # ----------- contact plots ---------------------------------------- #
    plot_contact_count(time_s, ncon.tolist(), plots_dir, run_meta)
    plot_contact_force_mag(time_s, fmag.tolist(), plots_dir, run_meta)

    # per-body contact-force lines
    plot_contact_force_per_body(time_s, per_body, body_names, plots_dir, run_meta)

    _plot_xy_trajectory(ref_x, ref_y, qpos[:, 0].tolist(), qpos[:, 1].tolist(), plots_dir, run_meta)

    # ----------------- gait plots --------------------------------------- #
    plot_n_feet_in_contact(time_s, n_foot_con, plots_dir, run_meta)
    plot_gait_frequency(time_s, gait_frequencies, plots_dir, run_meta)
    plot_double_support_intervals(time_s, double_support_intervals, plots_dir, run_meta)

    # ----------------- additional policy-input plots ------------------ #
    try:
        joint_names_full = load_joint_names(Path(cast(str, run_meta["kinfer"])))
    except Exception:
        logger.warning("Failed to load joint names from %s", run_meta["kinfer"])
        joint_names_full = []

    # explicit per-input component labels
    input_labels: dict[str, list[str]] = {
        "command": ["x_vel", "y_vel", "z_ang_vel"],
        "gyroscope": ["x_ang_vel", "y_ang_vel", "z_ang_vel"],
        "projected_gravity": ["x", "y", "z"],
    }

    for name, vals in input_datasets.items():
        # ────────── label selection ────────── #
        if name in {"joint_angles", "joint_angular_velocities", "joint_velocities", "action"} and joint_names_full:
            labels = joint_names_full[: vals.shape[1]]
        elif name in input_labels and len(input_labels[name]) == vals.shape[1]:
            labels = input_labels[name]
        else:
            labels = [f"{name}_{i}" for i in range(vals.shape[1])]

        plot_input_series(time_s, vals, labels, name, plots_dir, run_meta)

    return summary
