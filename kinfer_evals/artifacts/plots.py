"""Plotting utilities."""

import re
import textwrap
import unicodedata
from pathlib import Path
from typing import Sequence

import numpy as np
from kinfer_sim.server import load_joint_names
from matplotlib import colors, pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from kinfer_evals.core.eval_types import EpisodeData, RunInfo
from kinfer_evals.core.metrics import (
    body_frame_vel,
    compute_double_support_intervals,
    compute_gait_frequency,
    yaw_from_quat_wxyz,
)
from kinfer_evals.reference_state import ReferenceStateTracker

# Shared defaults
_FIGSIZE_SINGLE = (7, 4)


def _wrap_footer(pairs: list[tuple[str, str]], fig: Figure, *, font_size_pt: int = 11) -> str:
    """Return a single multiline string where each pair (label, text) is rendered as `label: text`.

    Wrapped so that no line exceeds the current figure width.

    We approximate the number of characters that fit:
        usable_px ≈ fig_width_inch * dpi  ·  0.66   (leave a tiny margin)
        char_px   ≈ 0.6 · font_size_pt    (empirical for most sans-serif fonts)
    """
    fig_w_px = fig.get_size_inches()[0] * fig.dpi * 0.66
    max_chars = max(20, int(fig_w_px / (0.6 * font_size_pt)))
    out: list[str] = []
    for label, text in pairs:
        prefix = f"{label}: "
        chunks = textwrap.wrap(text, width=max_chars - len(prefix), break_long_words=True, break_on_hyphens=False) or [
            ""
        ]
        for i, chunk in enumerate(chunks):
            out.append(prefix + chunk if i == 0 else " " * len(prefix) + chunk)
    return "\n".join(out)


def _add_footer(fig: Figure, run_info: RunInfo) -> None:
    """Add a standardized footer with run information to the figure."""
    pairs: list[tuple[str, str]] = [
        ("kinfer_file", str(run_info["kinfer_file"])),
        ("robot", str(run_info["robot"])),
        ("eval_name", str(run_info["eval_name"])),
        ("timestamp", str(run_info["timestamp"])),
        ("outdir", str(run_info["outdir"])),
    ]
    fig.text(
        0.0,
        -0.02,
        _wrap_footer(pairs, fig, font_size_pt=12),
        ha="left",
        va="top",
        fontsize=12,
        family="monospace",
        linespacing=1.4,
    )


def _make_fig_with_footer() -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create a standardized 2-panel figure with space reserved for footer."""
    fig, (ax_top, ax_err) = plt.subplots(2, 1, sharex=True, figsize=(7, 5), height_ratios=[3, 1])
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    return fig, (ax_top, ax_err)


def _finalize_and_save(fig: Figure, outdir: Path, filename: str, run_info: RunInfo) -> Path:
    """Append footer, ensure directory, save PNG, close figure, and return path."""
    _add_footer(fig, run_info)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _single_line_plot(
    time_s: np.ndarray,
    series: Sequence[tuple[np.ndarray, str]],
    *,
    title: str,
    xlabel: str,
    ylabel: str | None,
    png_name: str,
    outdir: Path,
    run_info: RunInfo,
    legend_kwargs: dict | None = None,
) -> Path:
    """Generic 1-axis line plot helper (no error subplot)."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    for y, lbl in series:
        ax.plot(time_s, y, label=lbl, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    # Show legend if caller asked for it, or if there are multiple series.
    if legend_kwargs is not None or len(series) > 1:
        ax.legend(**(legend_kwargs or {"loc": "upper right"}))
    return _finalize_and_save(fig, outdir, png_name, run_info)


def _plot_series_pair(
    time: np.ndarray,
    series: Sequence[tuple[np.ndarray, str]],
    err: np.ndarray,
    *,
    title: str,
    y_label: str,
    png_name: str,
    outdir: Path,
    run_info: RunInfo,
) -> Path:
    """Generic helper for plotting time series with error subplot."""
    fig, (ax, ax_err) = _make_fig_with_footer()
    for y, lbl in series:
        ax.plot(time, y, label=lbl)
    ax.set_title(title, pad=8)
    ax.set_ylabel(y_label)
    ax.legend(loc="upper right")

    ax_err.plot(time, err, label="error", linewidth=1)
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("err")
    ax_err.legend(loc="upper right")

    return _finalize_and_save(fig, outdir, png_name, run_info)


def plot_velocity(
    time_s: np.ndarray,
    cmd: np.ndarray,
    act: np.ndarray,
    err: np.ndarray,
    axis: str,
    outdir: Path,
    info: RunInfo,
) -> Path:
    """Plot velocity tracking for a given axis."""
    return _plot_series_pair(
        time_s,
        [(cmd, f"command v{axis}"), (act, f"actual v{axis}")],
        err,
        title=f"Body-frame velocity tracking – v{axis}",
        y_label=f"v{axis}  [m·s⁻¹]",
        png_name=f"velocity_{axis}.png",
        outdir=outdir,
        run_info=info,
    )


def plot_accel(
    time_s: np.ndarray,
    cmd: np.ndarray,
    act: np.ndarray,
    err: np.ndarray,
    axis: str,
    outdir: Path,
    info: RunInfo,
) -> Path:
    """Plot acceleration tracking for a given axis."""
    return _plot_series_pair(
        time_s,
        [(cmd, f"command a{axis}"), (act, f"actual a{axis}")],
        err,
        title=f"Body-frame acceleration tracking – a{axis}",
        y_label=f"a{axis}  [m·s⁻²]",
        png_name=f"accel_{axis}.png",
        outdir=outdir,
        run_info=info,
    )


def plot_heading(
    time_s: np.ndarray,
    ref: np.ndarray,
    act: np.ndarray,
    err: np.ndarray,
    outdir: Path,
    info: RunInfo,
) -> Path:
    """Plot heading tracking (yaw)."""
    return _plot_series_pair(
        time_s,
        [(ref, "reference yaw"), (act, "actual yaw")],
        err,
        title="Heading tracking (yaw)",
        y_label="yaw  [rad]",
        png_name="heading_yaw.png",
        outdir=outdir,
        run_info=info,
    )


def plot_omega(
    time_s: np.ndarray,
    cmd: np.ndarray,
    act: np.ndarray,
    err: np.ndarray,
    outdir: Path,
    info: RunInfo,
) -> Path:
    """Plot angular-velocity tracking (ω)."""
    return _plot_series_pair(
        time_s,
        [(cmd, "command ω"), (act, "actual ω")],
        err,
        title="Angular-velocity tracking (ω)",
        y_label="ω  [rad s⁻¹]",
        png_name="angular_velocity.png",
        outdir=outdir,
        run_info=info,
    )


def plot_actions(
    time_s: np.ndarray,
    actions: np.ndarray,
    joint_names: Sequence[str],
    outdir: Path,
    info: RunInfo,
) -> Path:
    """Plot per-joint action targets."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.tight_layout(rect=(0, 0.15, 1, 1))  # footer room for legend

    for j, name in enumerate(joint_names):
        ax.plot(time_s, actions[:, j], label=name, linewidth=1)

    ax.set_title("Per-joint action targets")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("target position [rad]")

    n_cols = min(4, max(1, len(joint_names) // 2))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=n_cols,
        fontsize=7,
        frameon=False,
    )

    return _finalize_and_save(fig, outdir, "actions.png", info)


def _plot_xy_trajectory(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    act_x: np.ndarray,
    act_y: np.ndarray,
    outdir: Path,
    run_info: RunInfo,
) -> Path:
    """A top-down plot comparing reference vs. actual XY trajectories.

    Reference path: green → blue, actual path: yellow → red (early → late).
    """
    fig, ax = plt.subplots(figsize=(5, 6))
    fig.tight_layout(rect=(0, 0.20, 1, 1))

    def add_gradient_line(
        x: np.ndarray,
        y: np.ndarray,
        start_col: str,
        end_col: str,
        label: str,
        lw: float = 2.0,
    ) -> None:
        """Add a line with a gradient color based on time progression."""
        pts = np.column_stack([x, y])
        segs = np.concatenate([pts[:-1, None], pts[1:, None]], axis=1)
        cmap = colors.LinearSegmentedColormap.from_list(f"{label}_cmap", [start_col, end_col])
        lc = LineCollection(segs, cmap=cmap, norm=colors.Normalize(0, len(x) - 1), linewidth=lw)
        lc.set_array(np.arange(len(x)))
        ax.add_collection(lc)
        ax.plot([], [], color=cmap(0.75), lw=lw, label=label)

    add_gradient_line(ref_x, ref_y, "#00b050", "#0070ff", "reference", lw=2.5)
    add_gradient_line(act_x, act_y, "#ffd700", "#ff0000", "actual", lw=1.5)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")
    ax.set_title("XY trajectory (colour = time progression)", pad=10)
    ax.legend(loc="best")

    return _finalize_and_save(fig, outdir, "traj_xy.png", run_info)


def _safe_fname(name: str) -> str:
    t = unicodedata.normalize("NFKD", name)
    t = re.sub(r"[\\/:*?\"<>|]", "-", t)
    t = re.sub(r"\s+", "_", t)
    return t


def plot_contact_count(time_s: np.ndarray, ncon: np.ndarray, outdir: Path, info: RunInfo) -> Path:
    """Plot the number of contacts over time."""
    return _single_line_plot(
        time_s,
        [(ncon, "#contacts")],
        title="Contact count",
        xlabel="time [s]",
        ylabel="# contacts",
        png_name="contact_count.png",
        outdir=outdir,
        run_info=info,
    )


def plot_contact_force_mag(time_s: np.ndarray, fmag: np.ndarray, outdir: Path, info: RunInfo) -> Path:
    """Plot the total contact-force magnitude over time."""
    return _single_line_plot(
        time_s,
        [(fmag, "Σ|F|")],
        title="Total contact-force magnitude",
        xlabel="time [s]",
        ylabel="Σ |F|  [N]",
        png_name="contact_force_mag.png",
        outdir=outdir,
        run_info=info,
    )


def plot_contact_force_per_body(
    time_s: np.ndarray,
    per_body: np.ndarray,
    body_names: Sequence[str],
    outdir: Path,
    info: RunInfo,
) -> list[Path]:
    """Plot the per-body contact-force magnitude over time."""
    paths: list[Path] = []

    # combined
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    nz = [i for i in range(per_body.shape[0]) if np.any(per_body[i] > 0)]
    for i in nz:
        ax.plot(time_s, per_body[i], linewidth=1, label=body_names[i])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("|F|  [N]")
    ax.set_title("Per-body contact-force magnitude")
    ax.legend(loc="upper right", fontsize=6, ncol=min(4, len(nz)))
    paths.append(_finalize_and_save(fig, outdir, "contact_force_per_body_all.png", info))

    # individual
    for i in nz:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.tight_layout(rect=(0, 0.20, 1, 1))
        ax.plot(time_s, per_body[i])
        ax.set_xlabel("time [s]")
        ax.set_ylabel("|F|  [N]")
        ax.set_title(f"Contact-force magnitude – {body_names[i]}")
        paths.append(_finalize_and_save(fig, outdir, f"contact_force_{_safe_fname(body_names[i])}.png", info))

    return paths


def plot_n_feet_in_contact(
    time_s: np.ndarray,
    n_foot_con: np.ndarray,
    outdir: Path,
    info: RunInfo,
) -> Path:
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    ax.plot(time_s, n_foot_con, linewidth=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("# feet in contact")
    ax.set_title("Number of feet in contact")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return _finalize_and_save(fig, outdir, "n_feet_in_contact.png", info)


def plot_gait_frequency(
    time_s: np.ndarray,
    gait_frequencies: dict[int, float],
    outdir: Path,
    info: RunInfo,
) -> Path:
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    strike_idx = sorted(gait_frequencies.keys())
    times = np.array([time_s[i] for i in strike_idx])
    freqs = np.array([gait_frequencies[i] for i in strike_idx])
    ax.bar(times, freqs, width=0.05, alpha=0.6, label="Instantaneous")
    if freqs.size:
        mean_f = float(np.mean(freqs))
        ax.axhline(mean_f, linestyle="--", linewidth=2, label=f"Mean ({mean_f:.2f} Hz)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Gait frequency over time (gaps = stand cmd)")
    ax.legend(loc="upper right")
    ax.set_xlim(time_s[0] - 0.5, time_s[-1] + 0.5)
    if freqs.size:
        ax.set_ylim(0.0, max(freqs) * 1.1)
    return _finalize_and_save(fig, outdir, "gait_frequency.png", info)


def plot_double_support_intervals(
    time_s: np.ndarray,
    double_support_intervals: dict[int, float],
    outdir: Path,
    info: RunInfo,
) -> Path:
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE)
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    idx = sorted(double_support_intervals.keys())
    times = np.array([time_s[i] for i in idx if i < len(time_s)])
    vals = np.array([double_support_intervals[i] for i in idx if i < len(time_s)])
    ax.bar(times, vals, width=0.05, alpha=0.6, label="Double support")
    if vals.size:
        mean_v = float(np.mean(vals))
        ax.axhline(mean_v, linestyle="--", linewidth=2, label=f"Mean ({mean_v:.2f})")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Double support interval")
    ax.set_title("Double support intervals over time")
    ax.legend(loc="upper right")
    ax.set_xlim(time_s[0] - 0.5, time_s[-1] + 0.5)
    if vals.size:
        ax.set_ylim(0.0, max(vals) * 1.1)
    return _finalize_and_save(fig, outdir, "double_support_intervals.png", info)


def plot_input_series(
    time_s: np.ndarray,
    data: np.ndarray,  # (T, N)
    labels: Sequence[str],
    name: str,
    outdir: Path,
    info: RunInfo,
) -> Path:
    """Plot each component of a policy-input vector on one figure."""
    series = [(data[:, i], labels[i]) for i in range(data.shape[1])]
    return _single_line_plot(
        time_s,
        series,
        title=f"Policy input – {name}",
        xlabel="time [s]",
        ylabel=None,
        png_name=f"input_{_safe_fname(name)}.png",
        outdir=outdir,
        run_info=info,
        legend_kwargs={"loc": "upper right", "fontsize": 7, "ncol": min(4, len(labels))},
    )


def render_artifacts(episode: EpisodeData, run_info: RunInfo, output_dir: Path) -> list[Path]:
    """Render all standard figures for an episode.

    Overview:
      1) Derive time step (dt) from episode.
      2) Compute world yaw (from qpos quats) and rotate qvel → body-frame v.
      3) Compute velocity/acceleration signals and errors vs. commands.
      4) Integrate commanded [vx, vy, ω] → reference yaw + XY trajectory.
      5) Plot velocity, acceleration, heading, omega; actions (if present).
      6) Aggregate per-body contact-force magnitudes over time and plot.
      7) Plot reference vs. actual XY trajectory.
      8) Plot each recorded policy-input series (if any).
      9) Return file paths of all saved PNGs under <output_dir>/plots.

    Pure plotting: reads EpisodeData, derives series, writes images only.

    Args:
      episode: Episode time series and metadata (already loaded from HDF5).
      run_info: Run metadata used for figure footers.
      output_dir: Base directory where a 'plots/' subfolder will be created.

    Returns:
      A list of Paths to all generated PNG artifacts.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    time_s = episode.time
    dt = episode.dt if episode.dt > 0 else (float(np.mean(np.diff(time_s))) if time_s.size >= 2 else 0.0)

    quat_wxyz = episode.qpos[:, 3:7]
    yaw_world = np.unwrap(yaw_from_quat_wxyz(quat_wxyz))
    body_vx, body_vy = body_frame_vel(episode.qvel, yaw_world)

    # Calculate velocity/acceleration errors
    command_vx = episode.cmd_vel[:, 0]
    command_vy = episode.cmd_vel[:, 1]
    command_omega = episode.cmd_vel[:, 2]

    error_vx = body_vx - command_vx
    error_vy = body_vy - command_vy

    body_ax = np.diff(body_vx) / dt
    body_ay = np.diff(body_vy) / dt
    command_ax = np.diff(command_vx) / dt
    command_ay = np.diff(command_vy) / dt
    error_ax = body_ax - command_ax
    error_ay = body_ay - command_ay
    command_acc_mag = np.hypot(command_ax, command_ay)
    body_acc_mag = np.hypot(body_ax, body_ay)
    error_acc_mag = body_acc_mag - command_acc_mag

    # Apply commands to reference state tracker
    tracker = ReferenceStateTracker()
    ref_yaw_world = np.zeros_like(time_s)
    ref_pos_x = np.zeros_like(time_s)
    ref_pos_y = np.zeros_like(time_s)
    for i in range(len(time_s)):
        tracker.step((float(command_vx[i]), float(command_vy[i])), float(command_omega[i]), dt)
        ref_yaw_world[i] = tracker.yaw
        ref_pos_x[i] = tracker.pos_x
        ref_pos_y[i] = tracker.pos_y
    ref_yaw_world = np.unwrap(ref_yaw_world)
    yaw_error = yaw_world - ref_yaw_world

    omega_act = np.diff(yaw_world) / dt
    omega_error = omega_act - command_omega[:-1]

    # Plot velocity, acceleration, heading, omega; actions (if present)
    artifact_paths: list[Path] = []
    artifact_paths.append(plot_velocity(time_s, command_vx, body_vx, error_vx, "x", plots_dir, run_info))
    artifact_paths.append(plot_velocity(time_s, command_vy, body_vy, error_vy, "y", plots_dir, run_info))

    artifact_paths.append(plot_accel(time_s[1:], command_ax, body_ax, error_ax, "x", plots_dir, run_info))
    artifact_paths.append(plot_accel(time_s[1:], command_ay, body_ay, error_ay, "y", plots_dir, run_info))
    artifact_paths.append(
        plot_accel(time_s[1:], command_acc_mag, body_acc_mag, error_acc_mag, "mag", plots_dir, run_info)
    )

    artifact_paths.append(plot_heading(time_s, ref_yaw_world, yaw_world, yaw_error, plots_dir, run_info))
    artifact_paths.append(plot_omega(time_s[1:], command_omega[:-1], omega_act, omega_error, plots_dir, run_info))

    if episode.action_target is not None:
        try:
            joint_names = load_joint_names(Path(run_info["kinfer_file"]))
        except Exception:
            joint_names = [f"joint_{i}" for i in range(episode.action_target.shape[1])]
        artifact_paths.append(plot_actions(time_s, episode.action_target, joint_names, plots_dir, run_info))

    artifact_paths.append(plot_contact_count(time_s, episode.contact_count.astype(np.int16), plots_dir, run_info))
    artifact_paths.append(plot_contact_force_mag(time_s, episode.contact_force_mag, plots_dir, run_info))

    # Per-body force magnitude over time
    num_bodies = len(episode.body_names)
    force_per_body = np.zeros((num_bodies, len(time_s)), dtype=np.float32)
    for step_index, (pair_ids, wrench_flat) in enumerate(zip(episode.contact_body, episode.contact_wrench)):
        if pair_ids.size == 0:
            continue
        forces_3d = wrench_flat.reshape(-1, 6)[:, :3]
        magnitudes = np.linalg.norm(forces_3d, axis=1)
        contact_pairs = pair_ids.reshape(-1, 2)
        for (a, b), mag in zip(contact_pairs, magnitudes):
            force_per_body[int(a), step_index] += float(mag)
            force_per_body[int(b), step_index] += float(mag)

    artifact_paths += plot_contact_force_per_body(time_s, force_per_body, episode.body_names, plots_dir, run_info)

    # XY trajectories (reference vs actual)
    artifact_paths.append(
        _plot_xy_trajectory(ref_pos_x, ref_pos_y, episode.qpos[:, 0], episode.qpos[:, 1], plots_dir, run_info)
    )

    # Treat every contacted body id (>0) as a "foot";
    foot_con = [set(map(int, arr[arr > 0])) for arr in episode.contact_body]
    n_foot_con = np.array([len(s) for s in foot_con], dtype=int)
    artifact_paths.append(plot_n_feet_in_contact(time_s, n_foot_con, plots_dir, run_info))

    gait_freqs = compute_gait_frequency(foot_con, dt, episode.cmd_vel)
    if gait_freqs:
        artifact_paths.append(plot_gait_frequency(time_s, gait_freqs, plots_dir, run_info))

    ds_intervals = compute_double_support_intervals(n_foot_con, dt, episode.cmd_vel)
    if ds_intervals:
        artifact_paths.append(plot_double_support_intervals(time_s, ds_intervals, plots_dir, run_info))

    if episode.inputs:
        label_suggestions: dict[str, list[str]] = {
            "command": ["x_vel", "y_vel", "z_ang_vel"],
            "gyroscope": ["x_ang_vel", "y_ang_vel", "z_ang_vel"],
            "projected_gravity": ["x", "y", "z"],
        }
        joint_name_cache: list[str] | None = None
        for name, values in episode.inputs.items():
            if name in {"joint_angles", "joint_angular_velocities", "joint_velocities", "action"}:
                if joint_name_cache is None:
                    try:
                        joint_name_cache = load_joint_names(Path(run_info["kinfer_file"]))
                    except Exception:
                        joint_name_cache = []
                labels = (
                    joint_name_cache[: values.shape[1]]
                    if joint_name_cache
                    else [f"{name}_{i}" for i in range(values.shape[1])]
                )
            elif name in label_suggestions and len(label_suggestions[name]) == values.shape[1]:
                labels = label_suggestions[name]
            else:
                labels = [f"{name}_{i}" for i in range(values.shape[1])]

            artifact_paths.append(plot_input_series(time_s, values, labels, name, plots_dir, run_info))

    return artifact_paths
