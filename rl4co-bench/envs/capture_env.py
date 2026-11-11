#!/usr/bin/env python
"""Frozen environment capture for the rl4co cross-architecture benchmark.

Run this INSIDE the `rl4co` conda env on each machine:

    conda activate rl4co
    python capture_env.py

It writes ``<slug>.json`` next to itself (slug auto-derived from OS + accelerator)
and prints the same JSON to stdout.

This file is the provenance record for every measurement under ``results/``.
It deliberately lives OUTSIDE the rl4co repository and writes nothing into it.

Re-run it whenever the environment changes, and bump ``rl4co-bench/env-capture``
if the schema changes so old records stay interpretable.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "rl4co-bench/env-capture@1"

# Matmul calibration size. 4096^3 x 3 arrays x 4 B ~= 600 MB of working set,
# which fits comfortably in 8 GB unified memory and in 11 GB VRAM.
CALIB_N = 4096
CALIB_ITERS = {"cuda": 20, "mps": 20, "cpu": 3}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _sh(cmd: list[str]) -> str | None:
    """Run a command, return stripped stdout, or None on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or None
    except Exception:
        return None


def _sysctl(key: str) -> str | None:
    return _sh(["sysctl", "-n", key])


def _version_of(mod_name: str) -> str:
    """Import the module and report its version.

    Prefer the module's own ``__version__``; fall back to distribution metadata,
    because several packages (pyvrp, vrplib, ruff, ...) ship under a different
    distribution name or expose no ``__version__`` at all.
    """
    try:
        mod = __import__(mod_name)
    except Exception as exc:
        return f"MISSING ({type(exc).__name__})"

    ver = getattr(mod, "__version__", None)
    if ver:
        return str(ver)

    from importlib.metadata import version as _dist_version

    for candidate in (mod_name, mod_name.replace("_", "-"), mod_name.replace("_", "")):
        try:
            return f"{_dist_version(candidate)} (via metadata)"
        except Exception:
            continue
    return "UNKNOWN"


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def host_section() -> dict:
    sysname = platform.system()
    info: dict = {
        "os": f"{sysname} {platform.release()}",
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "hostname": platform.node(),
    }

    if sysname == "Darwin":
        info["os_detail"] = _sh(["sw_vers", "-productVersion"])
        info["os_build"] = _sh(["sw_vers", "-buildVersion"])
        info["chip"] = _sysctl("machdep.cpu.brand_string")
        info["model"] = _sh(["sysctl", "-n", "hw.model"])
        mem = _sysctl("hw.memsize")
        info["ram_bytes"] = int(mem) if mem and mem.isdigit() else None
        ncpu = _sysctl("hw.ncpu")
        info["cpu_count"] = int(ncpu) if ncpu and ncpu.isdigit() else os.cpu_count()
        # Apple Silicon reports performance/efficiency split here
        info["p_cores"] = _sysctl("hw.perflevel0.physicalcpu")
        info["e_cores"] = _sysctl("hw.perflevel1.physicalcpu")

    elif sysname == "Windows":
        info["chip"] = _sh(["wmic", "cpu", "get", "name"]) or platform.processor()
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
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

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            info["ram_bytes"] = int(stat.ullTotalPhys)
        except Exception:
            info["ram_bytes"] = None
        info["cpu_count"] = os.cpu_count()

    else:
        info["cpu_count"] = os.cpu_count()

    return info


def packages_section() -> dict:
    pkgs = {}
    for mod in (
        "torch",
        "tensordict",
        "torchrl",
        "lightning",
        "pytorch_lightning",
        "numpy",
        "scipy",
        "hydra",
        "omegaconf",
        "einops",
        "matplotlib",
        "wandb",
        "rl4co",
    ):
        pkgs[mod] = _version_of(mod)
    return pkgs


def rl4co_section() -> dict:
    info: dict = {"version": _version_of("rl4co"), "location": None, "editable": None, "git_head": None}

    try:
        import rl4co

        loc = Path(rl4co.__file__).resolve()
        info["location"] = str(loc.parent)
        # An editable install resolves to <repo>/rl4co/__init__.py, i.e. two levels
        # below the repo root, and the repo root carries a .git dir.
        repo_root = loc.parent.parent
        info["editable"] = (repo_root / ".git").exists()
        if info["editable"]:
            info["repo_root"] = str(repo_root)
            head = _sh(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
            info["git_head"] = head
            # Provenance cares about MODIFIED TRACKED FILES, not untracked scratch.
            tracked = _sh(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"])
            info["git_tracked_modified"] = bool(tracked)
            info["git_tracked_diff"] = tracked
            untracked = _sh(["git", "-C", str(repo_root), "ls-files", "--others", "--exclude-standard"])
            info["git_untracked"] = untracked.splitlines() if untracked else []
            info["git_describe"] = _sh(["git", "-C", str(repo_root), "describe", "--tags", "--always"])
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"

    return info


def backends_section() -> dict:
    import torch

    info: dict = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_built": torch.version.cuda,
        "cudnn_version": None,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "threads": torch.get_num_threads(),
    }

    if info["cuda_available"]:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info["device_name"] = props.name
        info["capability"] = list(torch.cuda.get_device_capability(idx))
        info["total_memory_bytes"] = int(props.total_memory)
        info["multi_processor_count"] = props.multi_processor_count
        info["arch_list"] = torch.cuda.get_arch_list()
        # Is THIS device's SM natively compiled, or would it fall back to PTX JIT?
        sm = f"sm_{info['capability'][0]}{info['capability'][1]}"
        info["native_sm_supported"] = sm in info["arch_list"]
        info["native_sm"] = sm
        try:
            info["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:
            pass
        try:
            import pynvml

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            info["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
            info["power_limit_w"] = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
            info["power_default_w"] = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000.0
            pynvml.nvmlShutdown()
        except Exception as exc:
            info["nvidia_smi_note"] = f"pynvml unavailable ({type(exc).__name__})"

    elif info["mps_available"]:
        info["device_name"] = "Apple GPU (Metal)"

    return info


def extras_section() -> dict:
    """Optional deps. These change what code paths *exist*, so they are provenance."""
    info = {}
    # NB: the pytest-xdist plugin imports as `xdist`.
    for mod in ("numba", "pyvrp", "vrplib", "torch_geometric", "pytest", "xdist", "ruff", "pre_commit"):
        info[mod] = _version_of(mod)

    # The concrete consequence: TSPEnv.local_search is None without numba.
    try:
        from rl4co.envs.routing.tsp.env import local_search

        info["tsp_local_search"] = "AVAILABLE" if local_search is not None else "None"
    except Exception as exc:
        info["tsp_local_search"] = f"ERROR ({type(exc).__name__})"

    return info


def calibration_section() -> dict:
    """Pure-hardware fp32 matmul throughput => the hardware-capability anchor.

    This is NOT a benchmark of rl4co. It exists so that when a workload runs
    4.6x slower on one machine, we can say whether that is the hardware or the
    backend. See CALIBRATION.md.
    """
    import torch

    n = CALIB_N
    flops = 2 * n**3

    def sync(dev: str) -> None:
        if dev == "cuda":
            torch.cuda.synchronize()
        elif dev == "mps":
            torch.mps.synchronize()

    def bench(dev: str) -> dict:
        iters = CALIB_ITERS.get(dev, 3)
        try:
            a = torch.randn(n, n, device=dev)
            b = torch.randn(n, n, device=dev)
            for _ in range(min(3, iters)):
                a @ b
            sync(dev)
            start = time.perf_counter()
            for _ in range(iters):
                a @ b
            sync(dev)
            ms = (time.perf_counter() - start) / iters * 1000.0
            return {"matmul_ms": round(ms, 2), "tflops": round(flops / (ms / 1000.0) / 1e12, 2), "iters": iters}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    import torch as _t

    out = {"n": n, "dtype": "float32"}
    if _t.cuda.is_available():
        out["cuda"] = bench("cuda")
    if _t.backends.mps.is_available():
        out["mps"] = bench("mps")
    if os.environ.get("SKIP_CPU_CALIB") != "1":
        out["cpu"] = bench("cpu")
    out["matmul_precision"] = str(_t.get_float32_matmul_precision())
    return out


# --------------------------------------------------------------------------
def main() -> int:
    import torch

    # Slug keys on the ACCELERATOR actually in use, not on CUDA availability:
    # a Mac has cuda_available == False but is not a CPU-only machine.
    system = platform.system().lower()
    if torch.cuda.is_available():
        slug = f"{system}-cuda"
    elif torch.backends.mps.is_available():
        slug = "macos-arm64-mps"
    else:
        slug = f"{system}-{platform.machine().lower()}-cpu"

    record = {
        "schema": SCHEMA,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": host_section(),
        "python": {"version": sys.version.split()[0], "executable": sys.executable, "prefix": sys.prefix},
        "packages": packages_section(),
        "rl4co": rl4co_section(),
        "backends": backends_section(),
        "extras": extras_section(),
        "calibration": calibration_section(),
    }

    text = json.dumps(record, indent=2, ensure_ascii=False, default=str)
    out_path = Path(__file__).resolve().parent / f"{slug}.json"
    out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n[written] {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
