"""Plotting utilities."""

from pathlib import Path
from typing import Mapping

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors
import textwrap



def _wrap_footer(pairs: list[tuple[str, str]], fig, font_size_pt: int = 11) -> str:
    """
    Return a single multiline string where each pair (label, text) is rendered
    as `label: text`, wrapped so that no line exceeds the current figure width.

    We approximate the number of characters that fit:
        usable_px ≈ fig_width_inch * dpi  ·  0.96   (leave a tiny margin)
        char_px   ≈ 0.6 · font_size_pt    (empirical for most sans-serif fonts)
    """
    fig_w_px = fig.get_size_inches()[0] * fig.dpi * 0.66
    max_chars = max(20, int(fig_w_px / (0.6 * font_size_pt)))

    wrapped_lines: list[str] = []
    for label, text in pairs:
        prefix = f"{label}: "
        # wrap text so *content* fits; subtract prefix length
        # (break_long_words=True lets us split a long path with no spaces)
        chunks = textwrap.wrap(
            text,
            width=max_chars - len(prefix),
            break_long_words=True,
            break_on_hyphens=False,
        )
        if not chunks:                           # empty string -> still print label
            wrapped_lines.append(prefix.rstrip())
            continue
        wrapped_lines.append(prefix + chunks[0])
        pad = " " * len(prefix)
        for chunk in chunks[1:]:
            wrapped_lines.append(pad + chunk)

    return "\n".join(wrapped_lines)


def _plot_velocity_series(
    time_s: list[float],
    command_body: list[float],
    actual_body: list[float],
    error_body: list[float],
    axis: str,
    outdir: Path,
    run_info: dict[str, str],
) -> None:
    """Save PNG with two stacked plots."""
    fig, (ax_top, ax_err) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7, 5),            # +1 inch for footer
        height_ratios=[3, 1],
    )
    # reserve bottom 20 % for the footer
    fig.tight_layout(rect=(0, 0.20, 1, 1))

    ax_top.plot(time_s, command_body, label=f"command v{axis}")
    ax_top.plot(time_s, actual_body, label=f"actual  v{axis}")
    ax_top.set_title(f"Body-frame velocity tracking – v{axis}", pad=8)
    ax_top.set_ylabel(f"v{axis}  [m·s⁻¹]")
    ax_top.legend(loc="upper right")

    ax_err.plot(time_s, error_body, label="error", linewidth=1)
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("err")
    ax_err.legend(loc="upper right")

    # ---------- footer ----------------------------------------------------
    footer_text = _wrap_footer(
        [
            ("kinfer",    run_info["kinfer"]),
            ("robot",     run_info["robot"]),
            ("eval",      run_info["eval_name"]),
            ("timestamp", run_info["timestamp"]),
            ("outdir",    run_info["outdir"]),
        ],
        fig,
        font_size_pt=12,
    )
    fig.text(
        0.0, -0.02,
        footer_text,
        ha="left", va="top",
        fontsize=12, family="monospace", linespacing=1.4,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"velocity_{axis}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)




def _plot_accel_series(
    time_s: list[float],
    command_body: list[float],
    actual_body: list[float],
    error_body: list[float],
    axis: str,
    outdir: Path,
    run_info: dict[str, str],
) -> None:
    """
    Plot commanded-vs-actual body-frame acceleration (m s⁻²) and the error
    for a single axis *axis* ∈ {'x', 'y', 'mag'}.
    """
    fig, (ax_top, ax_err) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 5), height_ratios=[3, 1]
    )
    fig.tight_layout(rect=(0, 0.20, 1, 1))   # reserve 20 % for footer

    ax_top.plot(time_s, command_body, label=f"command a{axis}")
    ax_top.plot(time_s, actual_body,  label=f"actual  a{axis}")
    ax_top.set_title(f"Body-frame acceleration tracking – a{axis}", pad=8)
    ax_top.set_ylabel(f"a{axis}  [m·s⁻²]")
    ax_top.legend(loc="upper right")

    ax_err.plot(time_s, error_body, label="error", linewidth=1)
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("err")
    ax_err.legend(loc="upper right")


    footer_text = _wrap_footer(
        [
            ("kinfer",    run_info["kinfer"]),
            ("robot",     run_info["robot"]),
            ("eval",      run_info["eval_name"]),
            ("timestamp", run_info["timestamp"]),
            ("outdir",    run_info["outdir"]),
        ],
        fig,
        font_size_pt=12,
    )
    fig.text(
        0.0, -0.02,
        footer_text,
        ha="left", va="top",
        fontsize=12, family="monospace", linespacing=1.4,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"accel_{axis}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)



def _plot_heading_series(
    time_s: list[float],
    ref_yaw: list[float],
    act_yaw: list[float],
    err_yaw: list[float],
    outdir: Path,
    run_info: dict[str, str],
) -> None:
    """Reference vs. actual heading (rad) and the tracking error."""
    fig, (ax_top, ax_err) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 5), height_ratios=[3, 1]
    )
    fig.tight_layout(rect=(0, 0.20, 1, 1))

    ax_top.plot(time_s, ref_yaw, label="reference yaw")
    ax_top.plot(time_s, act_yaw, label="actual yaw")
    ax_top.set_title("Heading tracking (yaw)", pad=8)
    ax_top.set_ylabel("yaw  [rad]")
    ax_top.legend(loc="upper right")

    ax_err.plot(time_s, err_yaw, label="error", linewidth=1)
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("err")
    ax_err.legend(loc="upper right")

    footer_text = _wrap_footer(
        [("kinfer", run_info["kinfer"]), ("robot", run_info["robot"]),
         ("eval", run_info["eval_name"]), ("timestamp", run_info["timestamp"]),
         ("outdir", run_info["outdir"])],
        fig, font_size_pt=12
    )
    fig.text(0.0, -0.02, footer_text, ha="left", va="top",
             fontsize=12, family="monospace", linespacing=1.4)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "heading_yaw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_omega_series(
    time_s: list[float],
    cmd_omega: list[float],
    act_omega: list[float],
    err_omega: list[float],
    outdir: Path,
    run_info: dict[str, str],
) -> None:
    """Commanded vs. actual angular velocity (rad s⁻¹) and the error."""
    fig, (ax_top, ax_err) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 5), height_ratios=[3, 1]
    )
    fig.tight_layout(rect=(0, 0.20, 1, 1))

    ax_top.plot(time_s, cmd_omega, label="command ω")
    ax_top.plot(time_s, act_omega, label="actual  ω")
    ax_top.set_title("Angular-velocity tracking (ω)", pad=8)
    ax_top.set_ylabel("ω  [rad s⁻¹]")
    ax_top.legend(loc="upper right")

    ax_err.plot(time_s, err_omega, label="error", linewidth=1)
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("err")
    ax_err.legend(loc="upper right")

    footer_text = _wrap_footer(
        [("kinfer", run_info["kinfer"]), ("robot", run_info["robot"]),
         ("eval", run_info["eval_name"]), ("timestamp", run_info["timestamp"]),
         ("outdir", run_info["outdir"])],
        fig, font_size_pt=12
    )
    fig.text(0.0, -0.02, footer_text, ha="left", va="top",
             fontsize=12, family="monospace", linespacing=1.4)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "angular_velocity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)




def _plot_xy_trajectory(
    ref_x: list[float],
    ref_y: list[float],
    act_x: list[float],
    act_y: list[float],
    outdir: Path,
    run_info: dict[str, str],
) -> None:
    """
    Save a top-down plot comparing reference vs. actual XY trajectories.
    Reference path: green → blue, actual path: yellow → red (early → late).
    """
    # give the footer some breathing room → make the figure taller
    fig, ax = plt.subplots(figsize=(5, 6))          # ↑ extra 1 inch
    # keep 20 % of the figure's height free at the bottom
    fig.tight_layout(rect=(0, 0.20, 1, 1))          # left, bottom, right, top

    def add_gradient_line(
        x: list[float],
        y: list[float],
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
            ("kinfer",    run_info["kinfer"]),
            ("robot",     run_info["robot"]),
            ("eval",      run_info["eval_name"]),
            ("timestamp", run_info["timestamp"]),
            ("outdir",    run_info["outdir"]),
        ],
        fig,
        font_size_pt=12,
    )
    fig.text(
        0.0, -0.02,                       # x-pos (left), y-pos just below the axes
        footer_text,
        ha="left",
        va="top",
        fontsize=12,                      # double-sized as requested
        linespacing=1.4,
        family="monospace",
    )

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "traj_xy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
