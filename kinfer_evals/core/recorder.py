# src/kinfer_evals/core/recorder.py
from __future__ import annotations
from pathlib import Path
import numpy as np, h5py, math
from typing import Any

_CHUNK = 1024          # steps per chunk → good compression ÷ I/O

class Recorder:
    """Append-only HDF5 writer for MuJoCo episodes (float32, gzip)."""

    def __init__(self, file: Path, model: Any, *, compress: str = "gzip", lvl: int = 4) -> None:
        self._f = h5py.File(file, "w")
        self._i = 0

        def _ds(name: str, shape1: tuple[int, ...]):
            return self._f.create_dataset(
                name,
                shape=(0, *shape1),
                maxshape=(None, *shape1),
                chunks=(_CHUNK, *shape1),
                dtype="f4",
                compression=compress,
                compression_opts=lvl,
            )

        nq, nv, nu, nb = model.nq, model.nv, model.nu, model.nbody
        self.time     = _ds("time",        ())           # scalar
        self.qpos     = _ds("qpos",        (nq,))
        self.qvel     = _ds("qvel",        (nv,))
        self.act_frc  = _ds("act_force",   (nu,))
        self.cacc     = _ds("cacc",        (nb, 6))      # 6-D per body
        self.wrench   = _ds("contact_wrench", (0, 6))    # ragged; will resize per-step

    # ------- public API -------------------------------------------------- #
    def append(self, data, t: float) -> None:
        """Copy the current mjData into the datasets (O(#floats) memcpy)."""
        s = slice(self._i, self._i + 1)

        # resize all fixed-shape datasets once per step
        for d, arr in (
            (self.time,     np.array(t, dtype=np.float32)),
            (self.qpos,     data.qpos),
            (self.qvel,     data.qvel),
            (self.act_frc,  data.actuator_force),
            (self.cacc,     data.cacc),
        ):
            d.resize(self._i + 1, axis=0)
            d[s] = arr

        # ------- contact wrench (ragged) -------------------------------- #
        cf = np.zeros(6, dtype=np.float32)
        forces = []
        for j in range(data.ncon):
            # mujoco.mj_contactForce(model.ptr, data.ptr, j, cf)  # C-API
            forces.append(cf.copy())                               # stub until C call wired
        self.wrench.resize(self._i + 1, axis=0)
        self.wrench[s] = np.stack(forces) if forces else np.zeros((0, 6), "f4")

        self._i += 1

    def close(self) -> None:
        self._f.close()
