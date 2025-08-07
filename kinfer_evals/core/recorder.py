"""HDF5 recorder for MuJoCo simulation data."""

from pathlib import Path

import h5py
import mujoco
import numpy as np

_CHUNK = 1024  # steps per chunk → good compression ÷ I/O


class Recorder:
    """Append-only HDF5 writer for MuJoCo episodes (float32, gzip)."""

    def __init__(self, file: Path, model: mujoco.MjModel, *, compress: str = "gzip", lvl: int = 4) -> None:
        self._f = h5py.File(file, "w")
        self._i = 0
        self._model = model  # store model reference for mj_contactForce

        def _ds(name: str, shape1: tuple[int, ...], dtype: str = "f4") -> h5py.Dataset:
            return self._f.create_dataset(
                name,
                shape=(0, *shape1),
                maxshape=(None, *shape1),
                chunks=(_CHUNK, *shape1),
                dtype=dtype,
                compression=compress,
                compression_opts=lvl,
            )

        nq, nv, nu, nb = model.nq, model.nv, model.nu, model.nbody
        self.time = _ds("time", ())  # scalar
        self.qpos = _ds("qpos", (nq,))
        self.qvel = _ds("qvel", (nv,))
        self.act_frc = _ds("act_force", (nu,))
        self.cacc = _ds("cacc", (nb, 6))  # 6-D per body

        # Command data storage
        self.cmd_vel = _ds("cmd_vel", (3,))  # [vx, vy, omega]

        # --- ragged contact wrench ------------------------------------ #
        vlen_f4 = h5py.vlen_dtype(np.dtype("f4"))  # VLEN float32
        self.wrench = self._f.create_dataset(
            "contact_wrench",
            shape=(0,),  # 1-D over timesteps
            maxshape=(None,),
            chunks=(_CHUNK,),  # single chunk axis
            dtype=vlen_f4,
            compression=compress,
            compression_opts=lvl,
        )
        self.ncon = _ds("contact_count", (), dtype="i2")          # #contacts/step
        self.fmag = _ds("contact_force_mag", (), dtype="f4")      # Σ|F| per step

    # ------- public API -------------------------------------------------- #
    def append(self, data: mujoco.MjData, t: float, cmd_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        """Copy the current mjData into the datasets (O(#floats) memcpy)."""
        s = slice(self._i, self._i + 1)

        # resize all fixed-shape datasets once per step
        for d, arr in (
            (self.time, np.array(t, dtype=np.float32)),
            (self.qpos, data.qpos),
            (self.qvel, data.qvel),
            (self.act_frc, data.actuator_force),
            (self.cacc, data.cacc),
            (self.cmd_vel, np.array(cmd_vel, dtype=np.float32)),
        ):
            d.resize(self._i + 1, axis=0)
            d[s] = arr

        # ------- contact wrench (ragged) -------------------------------- #
        cf = np.empty(6, dtype=np.float64)  # mjtNum = float64
        frames = []
        for j in range(data.ncon):
            mujoco.mj_contactForce(self._model, data, j, cf)  # 6-D FT
            frames.append(cf.copy().astype(np.float32))  # cast to f32 for storage

        # flatten (ncon,6) → (6*ncon,)  for storage; reader reshapes later
        flat = np.concatenate(frames).astype("f4") if frames else np.zeros(0, dtype="f4")

        self.wrench.resize(self._i + 1, axis=0)
        self.wrench[s] = [flat]  # each element = 1 VLEN array

        # ---------- per-step aggregates -------------------------------- #
        self.ncon.resize(self._i + 1, axis=0)
        self.ncon[s] = data.ncon

        total_f = 0.0
        if frames:                                    # frames = [(6,), …]
            forces = np.asarray(frames, dtype=np.float32)[:, :3]   # Fx Fy Fz
            total_f = float(np.linalg.norm(forces, axis=1).sum())  # Σ|F|
        self.fmag.resize(self._i + 1, axis=0)
        self.fmag[s] = total_f

        self._i += 1

    def close(self) -> None:
        self._f.close()
