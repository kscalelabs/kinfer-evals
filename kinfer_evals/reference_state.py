"""Reference-state utilities."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReferenceStateTracker:
    """Integrate body-frame (vx, vy) commands into world-frame (x, y),
    while keeping our own reference heading."""

    @staticmethod
    def _make_zeros() -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    pos: np.ndarray = field(default_factory=_make_zeros)
    yaw: float = 0.0

    def reset(
        self,
        origin_xy: tuple[float, float] | None = None,
        yaw: float = 0.0,
    ) -> None:
        """Re-initialise the reference state."""
        self.pos[:] = origin_xy if origin_xy is not None else (0.0, 0.0)
        self.yaw = float(yaw)

    def step(
        self,
        v_cmd_body_xy: tuple[float, float] | list[float],
        omega_cmd: float,
        dt: float,
    ) -> None:
        """Advance one control tick."""
        # 1) Integrate reference yaw from commanded angular velocity
        self.yaw += omega_cmd * dt
        h = self.yaw
        c, s = np.cos(h), np.sin(h)

        # Body-frame to world-frame rotation
        vx_b, vy_b = v_cmd_body_xy
        vx_w = c * vx_b - s * vy_b
        vy_w = s * vx_b + c * vy_b

        self.pos += np.array([vx_w, vy_w], dtype=np.float32) * dt
