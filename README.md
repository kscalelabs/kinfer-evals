# kinfer-evals
Evals for kinfer policies 


## Installation

```bash
# clone the repo
# cd into the repo root 
pip install -e . 
```

# Usage

Run via 

```bash
kinfer-eval   /path/to/policy.kinfer   robot-name eval_type   --out /path/to/output

# for example 
kinfer-eval  ~/policies/frosty_feynman.kinfer   kbot-headless walk_forward_right   --out ~/eval_runs
```

# Troubleshooting 

## "GL context" / headless rendering errors

If you see errors like:

- `ImportError: Cannot initialize a EGL device display … PLATFORM_DEVICE`
- `ImportError: Cannot use OSMesa rendering platform …`
- `AttributeError: 'NoneType' object has no attribute 'glGetError'`
- Symbol errors like `GLIBCXX_3.4.30 not found`, `__malloc_hook`, `FunctionType`

This is usually because:

- The machine is a VM (no NVIDIA GPU) but the environment forces the NVIDIA EGL path (e.g., `LD_PRELOAD=/usr/lib/.../libEGL_nvidia.so.0`), so MuJoCo can't create a headless EGL context.
- Or the environment mixes EGL and OSMesa variables.
- Or Mesa/LLVM needs a newer libstdc++ than the one in your Conda env.

### Recommended fix (VM / no GPU: use software OSMesa)

Use the `kinfer-evals/kinfer_evals/scripts/run_eval_osmesa.sh` script to run the eval, it basically does the following:

```bash
# pick one: put these in your wrapper script or export before running
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYOPENGL_OSMESA_LIBRARY=/usr/lib/x86_64-linux-gnu/libOSMesa.so.8

# keep EGL/NVIDIA bits out of the way
unset LD_PRELOAD __EGL_VENDOR_LIBRARY_FILENAMES EGL_PLATFORM DISPLAY

# if you see 'GLIBCXX_3.4.30 not found', do ONE of the following:
# (A) quick workaround:
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# (B) better: upgrade Conda runtime so preload is unnecessary:
# conda install -n <env> -c conda-forge "libstdcxx-ng>=12" "libgcc-ng>=12"
```

### If you actually have an NVIDIA GPU (use EGL)

```bash
export MUJOCO_GL=egl
unset PYOPENGL_PLATFORM  # don't set 'osmesa' here
# optional: point GLVND at NVIDIA's JSON if needed
# export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_nvidia.json
```

### Sanity check

```bash
python - <<'PY'
from mujoco import gl_context
gl_context.GLContext(64, 64)
print("MuJoCo GLContext OK")
PY
```

If this prints "OK", your GL stack is configured correctly.

