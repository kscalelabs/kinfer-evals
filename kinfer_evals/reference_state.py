"""Reference-state utilities."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ReferenceStateTracker:
    """Integrate body-frame (vx, vy) commands into world-frame (x, y),
    while keeping our own reference heading."""

    pos_x: float = 0.0     # world-frame x-position [m]
    pos_y: float = 0.0     # world-frame y-position [m]
    yaw: float = 0.0       # reference heading [rad]

    def reset(
        self,
        origin_xy: tuple[float, float] | None = None,
        yaw: float = 0.0,
    ) -> None:
        """Re-initialise the reference state."""
        if origin_xy is not None:
            self.pos_x, self.pos_y = map(float, origin_xy)
        else:
            self.pos_x = self.pos_y = 0.0
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

        self.pos_x += vx_w * dt
        self.pos_y += vy_w * dt
