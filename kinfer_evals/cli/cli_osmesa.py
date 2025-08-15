# kinfer_evals/cli_osmesa.py
"""
CLI entry point for running kinfer-evals with software (OSMesa) rendering.

It:
- Forces OSMesa GL (good for VMs / no GPU).
- Unsets EGL/NVIDIA variables that can confuse things.
- Preloads the system libstdc++ with RTLD_GLOBAL so Mesa/LLVM can resolve
  GLIBCXX_3.4.30+ even inside a Conda env (behaves like LD_PRELOAD).
  Disable by setting USE_LIBSTDCXX_PRELOAD=0.
"""

import os
import sys
import ctypes
from ctypes.util import find_library


def _preload_libstdcxx():
    # Skip if user opts out
    if os.environ.get("USE_LIBSTDCXX_PRELOAD", "1") != "1":
        return
    # Allow override
    candidates = [
        os.environ.get("LIBSTDCXX_PATH"),
        "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        find_library("stdc++"),
        "libstdc++.so.6",
    ]
    mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
    for c in candidates:
        if not c:
            continue
        try:
            ctypes.CDLL(c, mode=mode)
            os.environ["KINFER_EVALS_LIBSTDCXX"] = c  # for debugging
            break
        except OSError:
            continue


def _setup_osmesa_env():
    # Force software rendering
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ.setdefault(
        "PYOPENGL_OSMESA_LIBRARY",
        "/usr/lib/x86_64-linux-gnu/libOSMesa.so.8",
    )
    # Keep EGL/NVIDIA paths from hijacking
    for k in ("__EGL_VENDOR_LIBRARY_FILENAMES", "EGL_PLATFORM", "DISPLAY", "LD_PRELOAD"):
        # Note: removing LD_PRELOAD here prevents surprises if user has stale values
        os.environ.pop(k, None)


def main():
    _preload_libstdcxx()
    _setup_osmesa_env()
    # Delegate to your existing argparse-based runner
    from kinfer_evals.cli.eval_runner import main as eval_main
    eval_main()


if __name__ == "__main__":
    main()