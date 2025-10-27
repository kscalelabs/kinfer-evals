"""HDF5 schema + read/write utilities."""

import json
from pathlib import Path
from typing import Optional

import h5py
import mujoco
import numpy as np

from kinfer_evals.core.eval_types import F32, Array1, Array2, EpisodeData, Int1, RunInfo

H5_SCHEMA_VERSION = 1


class EpisodeWriter:
    """Append-only writer for episode data with a self-described schema."""

    def __init__(
        self,
        file: Path,
        model: mujoco.MjModel,
        *,
        control_rate_hz: float,
        run_info: RunInfo,
        compress: str = "gzip",
        lvl: int = 4,
    ) -> None:
        self._f = h5py.File(file, "w")
        self._i = 0
        self._model = model
        self._compress = compress
        self._compress_lvl = lvl

        nq, nv, nu, nb = model.nq, model.nv, model.nu, model.nbody

        # File-level attributes (schema, units, metadata)
        units = {
            "time": "s",
            "qpos": "rad",
            "qvel": "rad/s",
            "actuator_force": "N",
            "action_target": "rad",
            "commands": "varies by command",
            "contact_force_mag": "N",
        }
        self._f.attrs["schema_version"] = H5_SCHEMA_VERSION
        self._f.attrs["units_json"] = json.dumps(units)
        self._f.attrs["control_rate_hz"] = float(control_rate_hz)
        self._f.attrs["dt_ctrl"] = float(1.0 / control_rate_hz)
        self._f.attrs["run_info_json"] = json.dumps(dict(run_info))

        # Dataset helper
        def _ds(name: str, shape1: tuple[int, ...], dtype: str = "f4") -> h5py.Dataset:
            return self._f.create_dataset(
                name,
                shape=(0, *shape1),
                maxshape=(None, *shape1),
                chunks=(1024, *shape1),
                dtype=dtype,
                compression=compress,
                compression_opts=lvl,
            )

        # Dense datasets
        self.time = _ds("time", ())  # (T,)
        self.qpos = _ds("qpos", (nq,))  # (T,nq)
        self.qvel = _ds("qvel", (nv,))  # (T,nv)
        self.actuator_force = _ds("actuator_force", (nu,))  # (T,nu)
        self.action_target = _ds("action_target", (nu,))  # (T,nu)

        # ragged contact arrays
        vlen_f4 = h5py.vlen_dtype(np.dtype("f4"))
        vlen_i2 = h5py.vlen_dtype(np.dtype("i2"))
        self.contact_wrench = self._f.create_dataset(
            "contact_wrench",
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype=vlen_f4,
            compression=compress,
            compression_opts=lvl,
        )
        self.contact_body = self._f.create_dataset(
            "contact_body",
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype=vlen_i2,
            compression=compress,
            compression_opts=lvl,
        )
        self.contact_count = _ds("contact_count", (), dtype="i2")
        self.contact_force_mag = _ds("contact_force_mag", (), dtype="f4")

        # Body names
        str_t = h5py.string_dtype("utf-8")
        self.body_names = self._f.create_dataset("body_names", (nb,), dtype=str_t)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}" for i in range(nb)]
        self.body_names[:] = names

        # Lazy policy-input datasets
        self._input_dsets: dict[str, h5py.Dataset] = {}
        # Lazy command datasets (will be created on first append)
        self._command_dsets: dict[str, h5py.Dataset] = {}
        self._force_per_body = np.zeros(nb, dtype=np.float32)

    def append(
        self,
        data: mujoco.MjData,
        *,
        t: float,
        command_frame: dict[str, float],
        action: np.ndarray,
        inputs: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        """Append one control tick snapshot."""
        s = slice(self._i, self._i + 1)

        for d, arr in (
            (self.time, np.array(t, dtype=np.float32)),
            (self.qpos, data.qpos),
            (self.qvel, data.qvel),
            (self.actuator_force, data.actuator_force),
            (self.action_target, action.astype(np.float32)),
        ):
            d.resize(self._i + 1, axis=0)
            d[s] = arr

        # Command frame: each command as a separate dataset (created on first see)
        if command_frame:
            for cmd_name, cmd_value in command_frame.items():
                ds_name = f"command_{cmd_name}"
                if ds_name not in self._command_dsets:
                    self._command_dsets[ds_name] = self._f.create_dataset(
                        ds_name,
                        shape=(0,),
                        maxshape=(None,),
                        chunks=(1024,),
                        dtype="f4",
                        compression=self._compress,
                        compression_opts=self._compress_lvl,
                    )
                ds = self._command_dsets[ds_name]
                ds.resize(self._i + 1, axis=0)
                ds[s] = np.float32(cmd_value)

        # Policy inputs: fixed-width 2D datasets created on first see
        if inputs:
            for name, arr in inputs.items():
                ds_name = f"input_{name}"
                flat = arr.astype(np.float32).ravel()
                if ds_name not in self._input_dsets:
                    self._input_dsets[ds_name] = self._f.create_dataset(
                        ds_name,
                        shape=(0, flat.shape[0]),
                        maxshape=(None, flat.shape[0]),
                        chunks=(1024, flat.shape[0]),
                        dtype="f4",
                        compression=self._compress,
                        compression_opts=self._compress_lvl,
                    )
                ds = self._input_dsets[ds_name]
                if ds.shape[1] != flat.shape[0]:
                    raise ValueError(f"Input '{name}' size changed during recording")
                ds.resize(self._i + 1, axis=0)
                ds[s] = flat

        # Contact Forces
        cf = np.empty(6, dtype=np.float64)
        frames = []
        bodies = []
        for j in range(data.ncon):
            mujoco.mj_contactForce(self._model, data, j, cf)
            frames.append(cf.copy().astype(np.float32))
            con = data.contact[j]
            body_a = self._model.geom_bodyid[con.geom1]
            body_b = self._model.geom_bodyid[con.geom2]
            bodies.extend([body_a, body_b])
            mag = float(np.linalg.norm(cf[:3]))
            self._force_per_body[body_a] += mag
            self._force_per_body[body_b] += mag

        flat = np.concatenate(frames).astype("f4") if frames else np.zeros(0, dtype="f4")

        self.contact_wrench.resize(self._i + 1, axis=0)
        self.contact_wrench[s] = [flat]

        self.contact_body.resize(self._i + 1, axis=0)
        self.contact_body[s] = [np.asarray(bodies, dtype="i2")]

        self.contact_count.resize(self._i + 1, axis=0)
        self.contact_count[s] = data.ncon

        total_f = 0.0
        if frames:
            forces = np.asarray(frames, dtype=np.float32)[:, :3]
            total_f = float(np.linalg.norm(forces, axis=1).sum())
        self.contact_force_mag.resize(self._i + 1, axis=0)
        self.contact_force_mag[s] = total_f

        self._i += 1

    def close(self) -> None:
        self._f.create_dataset(
            "force_per_body",
            data=self._force_per_body.astype("f4"),
            dtype="f4",
        )
        self._f.close()


class EpisodeReader:
    """Load an episode from HDF5 into a typed container."""

    @staticmethod
    def read(h5: Path) -> EpisodeData:
        with h5py.File(h5, "r") as f:
            time: Array1 = f["time"][:].astype(F32)
            qpos: Array2 = f["qpos"][:].astype(F32)
            qvel: Array2 = f["qvel"][:].astype(F32)
            actuator_force: Array2 = f["actuator_force"][:].astype(F32)
            action_target: Optional[Array2] = f["action_target"][:].astype(F32) if "action_target" in f else None
            contact_wrench = list(f["contact_wrench"][:])
            contact_body = list(f["contact_body"][:])
            contact_count: Int1 = f["contact_count"][:].astype(np.int16)
            contact_force_mag: Array1 = f["contact_force_mag"][:].astype(F32)
            body_names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["body_names"][:]]

            # Collect command frame values (all datasets named command_*)
            commands: dict[str, Array1] = {}
            for key in f.keys():
                if key.startswith("command_"):
                    name = key[8:]  # Remove "command_" prefix
                    commands[name] = f[key][:].astype(F32)

            # Collect policy inputs (all fixed-width 2D datasets named input_*)
            inputs: dict[str, Array2] = {}
            for key in f.keys():
                if not key.startswith("input_"):
                    continue
                name = key[6:]
                vals = f[key][:]
                if vals.ndim == 1:
                    vals = vals[:, None]
                inputs[name] = vals.astype(F32)

            dt_attr = f.attrs.get("dt_ctrl", None)
            if dt_attr is not None:
                dt = float(dt_attr)
            else:
                dt = float(np.mean(np.diff(time))) if time.size >= 2 else 0.0

        return EpisodeData(
            time=time,
            qpos=qpos,
            qvel=qvel,
            actuator_force=actuator_force,
            action_target=action_target,
            commands=commands,
            contact_wrench=contact_wrench,
            contact_body=contact_body,
            contact_count=contact_count,
            contact_force_mag=contact_force_mag,
            body_names=body_names,
            dt=dt,
            inputs=inputs,
        )
