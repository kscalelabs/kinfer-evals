"""Reference-state utilities."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ReferenceStateTracker:
    """Integrate body-frame (vx, vy) commands into world-frame (x, y)."""

    def _make_zeros() -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    pos: np.ndarray = field(default_factory=_make_zeros)
    heading_rad: float = 0.0 # world-frame yaw used for rotation


    def reset(
        self,
        origin_xy: tuple[float, float] | None = None,
        heading_rad: float = 0.0,
    ) -> None:
        """Re-initialise the reference state."""
        self.pos[:] = origin_xy if origin_xy is not None else (0.0, 0.0)
        self.heading_rad = float(heading_rad)

    def step(
        self,
        v_cmd_body_xy: tuple[float, float] | list[float],
        dt: float,
        heading_rad: float | None = None,
    ) -> None:
        """Advance one control tick."""
        h = float(self.heading_rad if heading_rad is None else heading_rad)
        c, s = np.cos(h), np.sin(h)

        # Body-frame to world-frame rotation
        vx_b, vy_b = v_cmd_body_xy
        vx_w = c * vx_b - s * vy_b
        vy_w = s * vx_b + c * vy_b

        self.pos += np.array([vx_w, vy_w], dtype=np.float32) * dt
