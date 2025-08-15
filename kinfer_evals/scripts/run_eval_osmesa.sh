# scripts/run_eval_osmesa.sh
#!/usr/bin/env bash
set -euo pipefail

# --- Headless MuJoCo via OSMesa (CPU) ---
# Use exactly these exports; they matched your "green" run.

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYOPENGL_OSMESA_LIBRARY=/usr/lib/x86_64-linux-gnu/libOSMesa.so.8

# Important: preload system libstdc++ so OSMesa/LLVM symbols resolve
# (You can remove this line later if you upgrade libstdc++ in the conda env.)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Optional: keep NVIDIA/GLVND stuff out of the way
unset __EGL_VENDOR_LIBRARY_FILENAMES EGL_PLATFORM DISPLAY

# Hand off to the eval runner, passing all args through
exec python -m kinfer_evals.core.eval_runner "$@"