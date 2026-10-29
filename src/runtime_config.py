"""Resource-aware defaults for local training on modest workstations."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys

_GPU_PROBE_RESULT: tuple[bool, bool] | None = None


def available_ram_gb() -> float | None:
    """Return currently available physical memory in GiB, if detectable."""
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        state = MEMORYSTATUSEX()
        state.dwLength = ctypes.sizeof(state)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            return state.ullAvailPhys / (1024 ** 3)
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        return None


def gpu_free_memory_gb() -> float | None:
    """Read free VRAM through NVIDIA's supported command-line utility."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        values = [int(line.strip()) for line in output.splitlines() if line.strip()]
        return max(values) / 1024 if values else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def training_profile() -> dict[str, object]:
    """Select conservative parallelism and GPU use based on live headroom."""
    free_ram = available_ram_gb()
    free_vram = gpu_free_memory_gb()
    requested = os.getenv("FRAUDSHIELD_GPU", "auto").strip().lower()
    gpu_enabled = requested not in {"0", "false", "off", "cpu"}
    if requested == "auto":
        gpu_enabled = free_vram is not None and free_vram >= 4.0
    elif gpu_enabled and (free_vram is None or free_vram < 3.5):
        print("[RUNTIME] GPU requested, but less than 3.5 GiB VRAM is free; using CPU.")
        gpu_enabled = False

    xgb_gpu = catboost_gpu = False
    if gpu_enabled:
        global _GPU_PROBE_RESULT
        if _GPU_PROBE_RESULT is None:
            _GPU_PROBE_RESULT = _probe_gpu_backends()
        xgb_gpu, catboost_gpu = _GPU_PROBE_RESULT
        gpu_enabled = xgb_gpu or catboost_gpu

    cpu_count = os.cpu_count() or 2
    n_jobs = max(1, min(4, cpu_count // 2 or 1))
    profile = {
        "available_ram_gb": round(free_ram, 2) if free_ram is not None else None,
        "gpu_free_gb": round(free_vram, 2) if free_vram is not None else None,
        "use_gpu": bool(gpu_enabled),
        "xgb_gpu": xgb_gpu,
        "catboost_gpu": catboost_gpu,
        "n_jobs": n_jobs,
    }
    return profile


def _probe_gpu_backends() -> tuple[bool, bool]:
    """Smoke-test each tree library on a tiny matrix before a long fit."""
    import numpy as np
    x = np.asarray([[0, 1], [1, 0], [0, 0], [1, 1]] * 32, dtype=np.float32)
    y = np.asarray([0, 1, 0, 1] * 32, dtype=np.int8)
    xgb_ok = catboost_ok = False
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=2, max_depth=2, tree_method="hist", device="cuda",
            max_bin=16, eval_metric="logloss", n_jobs=1, verbosity=0,
        )
        model.fit(x, y)
        del model
        xgb_ok = True
    except Exception as exc:
        print(f"[RUNTIME] XGBoost CUDA probe unavailable; this model will use CPU ({exc}).")
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=2, depth=2, task_type="GPU", devices="0",
            gpu_ram_part=0.15, thread_count=1, verbose=False,
            allow_writing_files=False,
        )
        model.fit(x, y)
        del model
        catboost_ok = True
    except Exception as exc:
        print(f"[RUNTIME] CatBoost CUDA probe unavailable; this model will use CPU ({exc}).")
    return xgb_ok, catboost_ok


def require_memory_headroom(min_free_gb: float, stage: str) -> None:
    """Fail early with an actionable message instead of risking an OS crash."""
    free_ram = available_ram_gb()
    if free_ram is not None and free_ram < min_free_gb:
        raise MemoryError(
            f"Only {free_ram:.1f} GiB system RAM is available before {stage}; "
            f"at least {min_free_gb:.1f} GiB is required. Close memory-heavy apps and retry."
        )
