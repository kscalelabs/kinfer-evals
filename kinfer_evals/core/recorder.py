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

        vlen_i2 = h5py.vlen_dtype(np.dtype("i2"))
        self.cbody = self._f.create_dataset(
            "contact_body",
            shape=(0,),
            maxshape=(None,),
            chunks=(_CHUNK,),
            dtype=vlen_i2,
            compression=compress,
            compression_opts=lvl,
        )

        self._force_per_body = np.zeros(nb, dtype=np.float32)

        str_t = h5py.string_dtype(encoding="utf-8")
        self.body_names = self._f.create_dataset("body_names", (nb,), dtype=str_t)
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            for i in range(nb)
        ]
        self.body_names[:] = names

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
        bodies = []
        for j in range(data.ncon):
            mujoco.mj_contactForce(self._model, data, j, cf)  # 6-D FT
            frames.append(cf.copy().astype(np.float32))  # cast to f32 for storage

            con = data.contact[j]
            bodies.extend([
                self._model.geom_bodyid[con.geom1],
                self._model.geom_bodyid[con.geom2],
            ])

            body_a = self._model.geom_bodyid[con.geom1]
            body_b = self._model.geom_bodyid[con.geom2]
            body_id = body_b if body_a == 0 else body_a
            self._force_per_body[body_id] += float(np.linalg.norm(cf[:3]))  # |F|

        # flatten (ncon,6) → (6*ncon,)  for storage; reader reshapes later
        flat = np.concatenate(frames).astype("f4") if frames else np.zeros(0, dtype="f4")

        self.wrench.resize(self._i + 1, axis=0)
        self.wrench[s] = [flat]

        self.cbody.resize(self._i + 1, axis=0)
        self.cbody[s] = [np.asarray(bodies, dtype="i2")]

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
        # write final per-body aggregate before closing the file
        self._f.create_dataset(
            "force_per_body",
            data=self._force_per_body.astype("f4"),   # (nbody,)  float32
            dtype="f4",
        )
        self._f.close()
