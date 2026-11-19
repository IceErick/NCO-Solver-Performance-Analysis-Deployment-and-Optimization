#!/usr/bin/env python3
"""Record and compare the fixed calibration workload for one measurement session."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "rl4co-bench/session@1"
BENCH = Path(__file__).resolve().parents[1]
CUDA_WORKLOAD = {"n": 4096, "warmup": 5, "blocks": 5, "iters": 10, "seed": 314159}
CPU_WORKLOAD = {"n": 128, "warmup": 2, "blocks": 3, "iters": 4, "seed": 314159}
# Protocol tolerances for drift screening. They are not claims about run variance.
MAX_TIME_DRIFT = 0.10
MAX_CLOCK_DRIFT = 0.10
MAX_TEMPERATURE_DRIFT_C = 8.0
SMI_FIELDS = (
    "uuid", "pci.bus_id", "driver_version", "temperature.gpu", "clocks.current.sm",
    "clocks.current.memory", "pstate", "power.limit", "utilization.gpu",
)
PACKAGES = ("numpy", "tensordict", "torchrl", "lightning", "hydra-core", "einops")


def nvidia_snapshot(torch, device: int) -> dict:
    command = ["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Cannot capture NVIDIA driver/clocks/thermals with nvidia-smi: {exc}") from exc
    devices = [
        dict(zip(SMI_FIELDS, (item.strip() for item in row), strict=True))
        for row in csv.reader(result.stdout.splitlines())
        if row
    ]
    if torch.cuda.device_count() == len(devices) == 1:
        return devices[0]
    props = torch.cuda.get_device_properties(device)
    # PyTorch may report the UUID with or without NVIDIA's GPU- prefix.
    uuid = str(getattr(props, "uuid", "")).lower().removeprefix("gpu-")
    match = [row for row in devices if row["uuid"].lower().removeprefix("gpu-") == uuid]
    if len(match) != 1:
        raise RuntimeError("Cannot identify the selected CUDA device in nvidia-smi telemetry")
    return match[0]


def calibrate(torch, device, workload: dict) -> dict:
    n = workload["n"]
    rng = torch.Generator(device="cpu").manual_seed(workload["seed"])
    a = torch.randn((n, n), generator=rng, dtype=torch.float32).to(device)
    b = torch.randn((n, n), generator=rng, dtype=torch.float32).to(device)
    c = torch.empty_like(a)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    blocks_ms = []
    with torch.inference_mode():
        for _ in range(workload["warmup"]):
            torch.mm(a, b, out=c)
        sync()
        for _ in range(workload["blocks"]):
            start = time.perf_counter()
            for _ in range(workload["iters"]):
                torch.mm(a, b, out=c)
            sync()
            blocks_ms.append((time.perf_counter() - start) * 1000 / workload["iters"])
    median_ms = statistics.median(blocks_ms)
    return {
        "workload": {**workload, "dtype": "float32", "matmul_precision": "highest"},
        "block_ms_per_matmul": blocks_ms,
        "median_ms_per_matmul": median_ms,
        "median_tflops": 2 * n**3 / (median_ms / 1000) / 1e12,
    }


def numeric_telemetry(record: dict, when: str, field: str) -> float | None:
    try:
        return float(record["telemetry"][when][field])
    except (KeyError, TypeError, ValueError):
        return None


def compare(record: dict, reference: dict) -> tuple[str, list[str]]:
    if (
        reference.get("schema") != SCHEMA
        or reference.get("measurement") is not True
        or reference.get("calibration", {}).get("status") != "reference"
    ):
        return "incomparable", ["Expected an original CUDA calibration reference"]
    identity = ("system", "os_release", "machine", "host", "device_name", "gpu_uuid",
                "driver", "torch_version", "torch_cuda_version", "cudnn_version", "packages",
                "python_version", "threads", "interop_threads")
    changed = [key for key in identity if reference["environment"].get(key) != record["environment"].get(key)]
    if reference["provenance"] != record["provenance"]:
        changed.append("source pin")
    if reference["calibration"]["workload"] != record["calibration"]["workload"]:
        changed.append("workload")
    if changed:
        return "incomparable", [f"Calibration identity differs: {', '.join(changed)}"]

    curr = record["calibration"]["median_ms_per_matmul"]
    prev = reference["calibration"]["median_ms_per_matmul"]
    rel = curr / prev - 1
    warnings = []
    if abs(rel) > MAX_TIME_DRIFT:
        warnings.append(f"Median matmul latency changed by {rel:+.1%}")
    for when in ("before", "after"):
        temp = numeric_telemetry(record, when, "temperature.gpu")
        ref_temp = numeric_telemetry(reference, when, "temperature.gpu")
        if temp is not None and ref_temp is not None and abs(temp - ref_temp) > MAX_TEMPERATURE_DRIFT_C:
            warnings.append(f"{when} temperature changed by {temp - ref_temp:+.1f} °C")
    for key in ("clocks.current.sm", "clocks.current.memory"):
        clock = numeric_telemetry(record, "after", key)
        ref_clock = numeric_telemetry(reference, "after", key)
        if clock is not None and ref_clock and abs(clock / ref_clock - 1) > MAX_CLOCK_DRIFT:
            warnings.append(f"Post-workload {key} changed by {clock / ref_clock - 1:+.1%}")
    return ("drift" if warnings else "pass"), warnings


def capture(output: Path, reference: Path | None = None, *, smoke: bool = False, device_index: int = 0) -> dict:
    if output.exists():
        raise FileExistsError(f"Choose a new calibration output path: {output}")
    if not smoke and reference is not None and not reference.is_file():
        raise FileNotFoundError(reference)
    if smoke and reference is not None:
        raise ValueError("CPU smoke calibration cannot be compared to a CUDA reference")

    sys.path.insert(0, str(BENCH / "instances"))
    from fixed_inputs import verify_rl4co

    pin = verify_rl4co()
    import torch

    if not smoke and (not torch.cuda.is_available() or not 0 <= device_index < torch.cuda.device_count()):
        raise RuntimeError("A CUDA-enabled PyTorch installation and a visible NVIDIA GPU are required")
    backend = "cpu" if smoke else "cuda"
    device = torch.device("cpu" if smoke else f"cuda:{device_index}")
    torch.set_float32_matmul_precision("highest")
    if not smoke:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    before = None if smoke else nvidia_snapshot(torch, device_index)
    pkg = {name: importlib.metadata.version(name) for name in PACKAGES} if not smoke else {}
    # Keep paths and outer harness Git HEAD out of the comparison across sessions.
    # The pin returned by verify_rl4co is also recorded in inference provenance.
    record = {
        "schema": SCHEMA,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "measurement": not smoke,
        "platform": platform.platform(),
        "backend": backend,
        "environment": {
            "system": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "host": platform.node(),
            "device_index": None if smoke else device_index,
            "device_name": platform.processor() if smoke else torch.cuda.get_device_name(device),
            "gpu_uuid": before["uuid"] if before else None,
            "driver": before["driver_version"] if before else None,
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": None if smoke else torch.backends.cudnn.version(),
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "packages": pkg,
        },
        "provenance": pin,
        "calibration": calibrate(torch, device, CPU_WORKLOAD if smoke else CUDA_WORKLOAD),
        "telemetry": {
            "before": before,
            "after": None if smoke else nvidia_snapshot(torch, device_index),
        },
    }
    status, warnings = ("smoke" if smoke else "reference"), []
    ref_hash = None
    if reference is not None:
        contents = reference.read_bytes()
        ref_hash = hashlib.sha256(contents).hexdigest()
        status, warnings = compare(record, json.loads(contents))
    record["calibration"].update(
        status=status, warnings=warnings, reference_sha256=ref_hash,
        max_relative_matmul_drift=MAX_TIME_DRIFT,
        max_relative_clock_drift=MAX_CLOCK_DRIFT,
        max_temperature_drift_c=MAX_TEMPERATURE_DRIFT_C,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, allow_nan=False)
        fh.write("\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--enrol-reference", action="store_true", help="Record the first CUDA calibration")
    selection.add_argument("--reference", type=Path, help="Compare with an enrolled CUDA reference")
    selection.add_argument("--smoke", action="store_true", help="Run a small CPU-only pipeline check")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()
    try:
        record = capture(args.output, args.reference, smoke=args.smoke, device_index=args.device)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    result = record["calibration"]
    print(f"{result['status']}: {result['median_ms_per_matmul']:.3f} ms/matmul; {args.output}")
    for warning in result["warnings"]:
        print(warning, file=sys.stderr)
    return 2 if result["status"] in ("drift", "incomparable") else 0


if __name__ == "__main__":
    raise SystemExit(main())
