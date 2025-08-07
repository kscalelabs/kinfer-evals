"""Plotting utilities."""

import textwrap
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib import colors, pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure


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
        for i, chunk in enumerate(
            textwrap.wrap(text, width=max_chars - len(prefix), break_long_words=True, break_on_hyphens=False) or [""]
        ):
            out.append(prefix + chunk if i == 0 else " " * len(prefix) + chunk)
    return "\n".join(out)


def _add_footer(fig: Figure, run_info: dict[str, object]) -> None:
    """Add a standardized footer with run information to the figure."""
    fig.text(
        0.0,
        -0.02,
        _wrap_footer(
            [(k, str(run_info[k])) for k in ("kinfer", "robot", "eval_name", "timestamp", "outdir")],
            fig,
            font_size_pt=12,
        ),
        ha="left",
        va="top",
        fontsize=12,
        family="monospace",
        linespacing=1.4,
    )


def _make_fig_with_footer() -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create a standardized 2-panel figure with space reserved for footer."""
    fig, (ax_top, ax_err) = plt.subplots(2, 1, sharex=True, figsize=(7, 5), height_ratios=[3, 1])
    fig.tight_layout(rect=(0, 0.20, 1, 1))  # 20 % footer
    return fig, (ax_top, ax_err)


def _plot_series_pair(
    time: Sequence[float],
    series: Sequence[tuple[Sequence[float], str]],  # [(y_values, label), …]
    err: Sequence[float],
    *,
    title: str,
    y_label: str,
    png_name: str,
    outdir: Path,
    run_info: dict[str, object],
) -> None:
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

    _add_footer(fig, run_info)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / png_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_velocity(
    time_s: Sequence[float],
    cmd: Sequence[float],
    act: Sequence[float],
    err: Sequence[float],
    axis: str,
    outdir: Path,
    info: dict[str, object],
) -> None:
    """Plot velocity tracking for a given axis."""
    _plot_series_pair(
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
    time_s: Sequence[float],
    cmd: Sequence[float],
    act: Sequence[float],
    err: Sequence[float],
    axis: str,
    outdir: Path,
    info: dict[str, object],
) -> None:
    """Plot acceleration tracking for a given axis."""
    _plot_series_pair(
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
    time_s: Sequence[float],
    ref: Sequence[float],
    act: Sequence[float],
    err: Sequence[float],
    outdir: Path,
    info: dict[str, object],
) -> None:
    """Plot heading tracking."""
    _plot_series_pair(
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
    time_s: Sequence[float],
    cmd: Sequence[float],
    act: Sequence[float],
    err: Sequence[float],
    outdir: Path,
    info: dict[str, object],
) -> None:
    """Plot angular velocity tracking."""
    _plot_series_pair(
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
    time_s: Sequence[float],
    actions: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    outdir: Path,
    info: dict[str, object],
) -> None:
    """Plot target joint positions over time, one coloured line per joint."""
    # leave extra vertical room for a legend row below the axes
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.tight_layout(rect=(0, 0.15, 1, 1))  # 15 % footer for legend

    for j, name in enumerate(joint_names):
        ax.plot(time_s, [a[j] for a in actions], label=name, linewidth=1)

    ax.set_title("Per-joint action targets")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("target position [rad]")
    # ----- legend below the plot ------------------------------------- #
    n_cols = min(4, max(1, len(joint_names) // 2))  # auto-fit ≤4 cols
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),  # centred, just below axes
        ncol=n_cols,
        fontsize=7,
        frameon=False,
    )

    _add_footer(fig, info)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "actions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_xy_trajectory(
    ref_x: Sequence[float],
    ref_y: Sequence[float],
    act_x: Sequence[float],
    act_y: Sequence[float],
    outdir: Path,
    run_info: dict[str, str],
) -> None:
    """Save a top-down plot comparing reference vs. actual XY trajectories.

    Reference path: green → blue, actual path: yellow → red (early → late).
    """
    # give the footer some breathing room → make the figure taller
    fig, ax = plt.subplots(figsize=(5, 6))  # ↑ extra 1 inch
    # keep 20 % of the figure's height free at the bottom
    fig.tight_layout(rect=(0, 0.20, 1, 1))  # left, bottom, right, top

    def add_gradient_line(
        x: Sequence[float],
        y: Sequence[float],
        start_col: str,
        end_col: str,
        label: str,
        lw: float = 2.0,
    ) -> None:
        """Plot a LineCollection that fades from *start_col* → *end_col*."""
        pts = np.column_stack([x, y])
        segs = np.concatenate([pts[:-1, None], pts[1:, None]], axis=1)

        cmap = colors.LinearSegmentedColormap.from_list(
            f"{label}_cmap",
            [start_col, end_col],
        )

        lc = LineCollection(
            segs,
            cmap=cmap,
            norm=colors.Normalize(0, len(x) - 1),
            linewidth=lw,
        )
        lc.set_array(np.arange(len(x)))
        ax.add_collection(lc)

        # add dummy handle for legend
        mid_colour = cmap(0.75)
        ax.plot([], [], color=mid_colour, lw=lw, label=label)

    # reference: green → blue
    add_gradient_line(ref_x, ref_y, "#00b050", "#0070ff", "reference", lw=2.5)

    # actual: yellow → red
    add_gradient_line(act_x, act_y, "#ffd700", "#ff0000", "actual", lw=1.5)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")
    ax.set_title("XY trajectory (colour = time progression)", pad=10)
    ax.legend(loc="best")

    # ---------- footer ----------------------------------------------------
    footer_text = _wrap_footer(
        [
            ("kinfer", str(run_info["kinfer"])),
            ("robot", str(run_info["robot"])),
            ("eval", str(run_info["eval_name"])),
            ("timestamp", str(run_info["timestamp"])),
            ("outdir", str(run_info["outdir"])),
        ],
        fig,
        font_size_pt=12,
    )
    fig.text(
        0.0,
        -0.02,  # x-pos (left), y-pos just below the axes
        footer_text,
        ha="left",
        va="top",
        fontsize=12,  # double-sized as requested
        linespacing=1.4,
        family="monospace",
    )

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "traj_xy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------- Contact utility plots ------------------------------------ #

def _make_single_axis_fig() -> tuple["plt.Figure", "plt.Axes"]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.tight_layout(rect=(0, 0.20, 1, 1))   # footer strip
    return fig, ax


def plot_contact_count(
    time_s: Sequence[float],
    ncon: Sequence[int],
    outdir: Path,
    info: dict[str, object],
) -> None:
    """Plot number of contacts over time."""
    fig, ax = _make_single_axis_fig()

    ax.plot(time_s, ncon, color="tab:blue")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("# contacts")
    ax.set_title("Contact count")

    _add_footer(fig, info)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "contact_count.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_contact_force_mag(
    time_s: Sequence[float],
    fmag: Sequence[float],
    outdir: Path,
    info: dict[str, object],
) -> None:
    """Plot summed |F| over time."""
    fig, ax = _make_single_axis_fig()

    ax.plot(time_s, fmag, color="tab:red")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Σ |F|  [N]")
    ax.set_title("Total contact-force magnitude")

    _add_footer(fig, info)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "contact_force_mag.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
