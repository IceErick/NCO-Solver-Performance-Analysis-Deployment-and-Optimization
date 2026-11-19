#!/usr/bin/env python3
"""Reduce AM/TSP timing and Kineto traces to comparable evidence for T2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

HOST_CLASSES = ("sync", "launch", "torch_runtime", "python", "unattributed")
CUDA_DEVICE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
GPU_MATH_HINTS = ("gemm", "gemv", "matmul", "bmm", "fmha", "attention")
RUNTIME_CATEGORIES = {"cuda_runtime", "cuda_driver"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def start(event: dict) -> float:
    return float(event["ts"])


def end(event: dict) -> float:
    return start(event) + float(event["dur"])


def union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    total = 0.0
    last_stop = -math.inf
    for lo, hi in sorted(intervals):
        if hi > last_stop:
            total += max(0, hi - max(lo, last_stop))
            last_stop = hi
    return total


def host_class(event: dict) -> str | None:
    category = event.get("cat", "")
    name = event["name"].lower()
    if category in RUNTIME_CATEGORIES:
        if "synchronize" in name and "streamwait" not in name:
            return "sync"
        if "launchkernel" in name or "graphlaunch" in name or "launchcooperativekernel" in name:
            return "launch"
        return "torch_runtime"
    if category == "cpu_op":
        return "torch_runtime"
    if category == "python_function":
        # Use project frames in the trace, avoiding parent profiler/site-package
        # wrapper frames that can blanket the entire trace.
        if any(token in name for token in ("rl4co/", "rl4co\\", "run_profile.py")):
            return "python"
    return None


def exclusive_host_time(activity: list[tuple[float, float, str]], lo: float, hi: float) -> Counter:
    """Attribute each instant on the inference thread to its highest priority event."""
    endpoints = [(lo, "unattributed", 1), (hi, "unattributed", -1)]
    for a, b, name in activity:
        left, right = max(lo, a), min(hi, b)
        if right > left:
            endpoints.extend(((left, name, 1), (right, name, -1)))
    active = Counter()
    totals = Counter()
    last = lo
    for t, label, delta in sorted(endpoints):
        if t > last:
            owner = next(key for key in HOST_CLASSES if active[key] > 0)
            totals[owner] += t - last
        active[label] += delta
        last = t
    return totals


def trace_metrics(path: Path, *, require_cuda: bool) -> dict:
    events = [
        event for event in read_json(path)["traceEvents"]
        if event.get("ph") == "X" and float(event.get("dur", 0)) > 0
    ]
    scopes = sorted((e for e in events if e["name"] == "benchmark.inference"), key=start)
    boundaries = [e for e in events if e["name"] == "benchmark.boundary_sync"]
    if not scopes:
        raise ValueError(f"No benchmark.inference ranges in {path}")
    host = Counter()
    counts = Counter()
    kernel_us = math_us = device_us = gpu_window_us = boundary_us = 0.0
    host_events = [(e, kind) for e in events if (kind := host_class(e)) is not None]
    gpu_events = [e for e in events if e.get("cat") in CUDA_DEVICE_CATEGORIES]
    if require_cuda and not any(event.get("cat") == "kernel" for event in gpu_events):
        raise ValueError(f"No CUDA kernel events in {path}; check Kineto/CUPTI capture")

    for i, scope in enumerate(scopes):
        lo, hi = start(scope), end(scope)
        identity = scope.get("pid"), scope.get("tid")
        next_lo = start(scopes[i + 1]) if i + 1 < len(scopes) else math.inf
        if next_lo < hi:
            raise ValueError(f"Overlapping benchmark.inference ranges in {path}")
        tail = next(
            (e for e in sorted(boundaries, key=start)
             if (e.get("pid"), e.get("tid")) == identity and hi <= start(e) < next_lo),
            None,
        )
        if tail is None and require_cuda:
            raise ValueError(f"Missing benchmark.boundary_sync after an inference range in {path}")
        device_hi = end(tail) if tail is not None else hi
        if device_hi > next_lo:
            raise ValueError(f"CUDA boundary overlaps the next inference range in {path}")
        if tail is not None:
            boundary_us += float(tail["dur"])
        activity = []
        for event, kind in host_events:
            if (event.get("pid"), event.get("tid")) == identity and start(event) < hi and end(event) > lo:
                activity.append((start(event), end(event), kind))
                counts[kind] += lo <= start(event) < hi
        host.update(exclusive_host_time(activity, lo, hi))
        if gpu_events:
            intervals = []
            kernels = []
            arithmetic = []
            for event in gpu_events:
                a, b = max(lo, start(event)), min(device_hi, end(event))
                if b <= a:
                    continue
                intervals.append((a, b))
                if event.get("cat") == "kernel":
                    kernels.append((a, b))
                    if any(marker in event["name"].lower() for marker in GPU_MATH_HINTS):
                        arithmetic.append((a, b))
            # Intervals are unioned across GPU streams to avoid counting overlap twice.
            device_us += union_duration(intervals)
            kernel_us += union_duration(kernels)
            math_us += union_duration(arithmetic)
            gpu_window_us += device_hi - lo

    host_us = sum(host.values())
    if require_cuda and not kernel_us:
        raise ValueError(f"No CUDA kernel activity during the measured ranges in {path}")
    return {
        "trace_batches": len(scopes),
        "trace_host_ms_per_batch": host_us / len(scopes) / 1000,
        **{f"host_{key}_pct": 100 * host[key] / host_us for key in HOST_CLASSES},
        "sync_calls_per_batch": counts["sync"] / len(scopes),
        "launch_calls_per_batch": counts["launch"] / len(scopes),
        "boundary_sync_ms_per_batch": boundary_us / len(scopes) / 1000,
        "device_busy_pct": 100 * device_us / gpu_window_us if gpu_window_us else None,
        "device_kernel_pct": 100 * kernel_us / gpu_window_us if gpu_window_us else None,
        "device_math_hint_pct": 100 * math_us / gpu_window_us if gpu_window_us else None,
    }


def percentile95(values: list[float]) -> float:
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def case_metrics(folder: Path, metadata: dict, *, measurement: bool) -> dict:
    if metadata.get("measurement") is not measurement:
        raise ValueError(f"Run and case measurement classification disagree in {folder}")
    fields = ("n", "b", "k", "decode_type", "status")
    result = {"case": folder.name, **{key: metadata[key] for key in fields}}
    if metadata["status"] != "ok":
        if metadata["status"] not in ("oom", "failed"):
            raise ValueError(f"Case in {folder} has not finished")
        return result
    timings = read_rows(folder / "timings.jsonl")
    routes = read_rows(folder / "routes.jsonl")
    latencies = [row["latency_ms"] for row in timings]
    total_instances = sum(row["batch_size"] for row in timings)
    if not latencies or any(not math.isfinite(x) or x <= 0 for x in latencies):
        raise ValueError(f"Missing or invalid timings in {folder}")
    if len(routes) != total_instances or not routes:
        raise ValueError(f"Expected one solution record per timed instance in {folder}")
    if any(not math.isfinite(row["cost"]) or row["cost"] < 0 for row in routes):
        raise ValueError(f"Invalid tour costs in {folder}")
    return {
        **result,
        "timed_batches": len(timings),
        "timed_instances": total_instances,
        "p50_ms": statistics.median(latencies),
        "p95_ms": percentile95(latencies),
        "instances_per_second": 1000 * total_instances / sum(latencies),
        "mean_tour_length": statistics.fmean(row["cost"] for row in routes),
        "peak_cuda_memory_bytes": metadata.get("peak_cuda_memory_bytes"),
        **trace_metrics(folder / "trace.json", require_cuda=measurement),
    }


def format_number(value) -> str:
    return "—" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


METRICS = {
    "p50_ms": "p50 ms", "p95_ms": "p95 ms", "instances_per_second": "instances/s",
    "mean_tour_length": "mean tour length", "device_busy_pct": "GPU busy %",
    "device_math_hint_pct": "GPU math kernel hint %",
    "host_sync_pct": "host sync %", "host_launch_pct": "host launch %",
    "host_torch_runtime_pct": "host ops/runtime %", "host_python_pct": "host Python %",
    "host_unattributed_pct": "host unattributed %",
}


def migration_rows(rows: list[dict]) -> list[dict]:
    migrations = []
    for axis in ("n", "b", "k"):
        fixed = [key for key in ("n", "b", "k", "decode_type") if key != axis]
        groups = {}
        for row in rows:
            if row["status"] == "ok":
                groups.setdefault(tuple(row[key] for key in fixed), []).append(row)
        for group in groups.values():
            smaller, larger = min(group, key=lambda x: x[axis]), max(group, key=lambda x: x[axis])
            if smaller[axis] == larger[axis]:
                continue
            metrics = ("host_sync_pct", "host_launch_pct", "host_torch_runtime_pct", "host_python_pct",
                       "host_unattributed_pct", "device_busy_pct", "device_math_hint_pct")
            deltas = {
                f"delta_{key}_points": larger[key] - smaller[key]
                if smaller.get(key) is not None and larger.get(key) is not None else None
                for key in metrics
            }
            migrations.append({
                "axis": axis, "first": smaller["case"], "last": larger["case"],
                "decode_type": smaller["decode_type"],
                **deltas, "latency_p50_ratio": larger["p50_ms"] / smaller["p50_ms"],
            })
    return migrations


def markdown(run: dict, rows: list[dict], migrations: list[dict]) -> str:
    lines = [
        "# T2 inference profiling", "",
        ("CUDA measurement on the NVIDIA workstation." if run["measurement"]
         else "**CPU smoke check only.** These results do not support CUDA or deployment conclusions."),
        "",
        "| Case | Decode strategy | Status | " + " | ".join(METRICS.values()) + " |",
        "| " + " | ".join(["---"] * (len(METRICS) + 3)) + " |",
    ]
    for row in rows:
        values = [row["case"], row["decode_type"], row["status"], *(format_number(row.get(key)) for key in METRICS)]
        lines.append("| " + " | ".join(values) + " |")
    strategies = {row["decode_type"] for row in rows}
    if {"greedy", "multistart_greedy"} <= strategies:
        lines += [
            "",
            "Moving from K=1 greedy to K>1 multistart greedy changes both the number of starts and the "
            "decode strategy. Both appear in the case table. Controlled budget comparisons use only "
            "cases sharing the same strategy.",
        ]
    if migrations:
        lines += [
            "", "## How the time distribution changed", "",
            "Each row varies one factor while holding the other two and the decode strategy fixed. "
            "Share changes are expressed in percentage points.",
            "",
            "| Factor | First | Last | Decode strategy | p50 ratio | Δ GPU busy | Δ GPU math hint | Δ host sync | Δ host launch | Δ host ops/runtime | Δ host Python | Δ host unattributed |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in migrations:
            keys = ("axis", "first", "last", "decode_type", "latency_p50_ratio", "delta_device_busy_pct_points",
                    "delta_device_math_hint_pct_points", "delta_host_sync_pct_points",
                    "delta_host_launch_pct_points", "delta_host_torch_runtime_pct_points",
                    "delta_host_python_pct_points", "delta_host_unattributed_pct_points")
            lines.append("| " + " | ".join(format_number(row[key]) for key in keys) + " |")
    if any(row.get("timed_batches", 0) < 20 for row in rows if row["status"] == "ok"):
        lines += ["", "Cases with fewer than 20 timing samples provide a coarse p95 estimate."]
    lines += [
        "", "## Reading the trace", "",
        "- The p50, p95 and throughput use the unprofiled batch timings for a sequential workload.",
        "- Mean tour length summarizes the instances seen in each case. Batch sizes can cover different "
        "prefixes of the fixed input set. Pair the same instances for quality comparisons.",
        "- Host percentages partition the inference thread during the marked ranges. Nested ranges count once, "
        "in this order: explicit CUDA synchronization, launch calls, other traced operations and runtime calls, "
        "recorded project Python frames, and the unattributed remainder. Python frame time can include untraced work.",
        "- GPU busy time is the union of kernel, copy and memset intervals from inference entry to the end of the marked "
        "boundary synchronization. Its complement means no captured device activity in that window.",
        "- The math kernel hint uses names associated with matrix operations and attention. It misses unnamed or fused "
        "arithmetic and measures occupancy of intervals rather than FLOP counts or utilization.",
        "- Host and GPU work can overlap. Their percentages have different denominators and should not be added. "
        "Profiler overhead means trace durations and unprofiled timings have different uses.",
        "- The boundary synchronization is recorded separately and excluded from the host inference partition. "
        "These measurements characterize batch execution; they contain no network or service queueing.",
        "",
    ]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(run_dir: Path, output: Path) -> None:
    run = read_json(run_dir / "run.json")
    if run.get("schema") != "rl4co-bench/profile-run@1" or run.get("status") not in ("ok", "failed"):
        raise ValueError("Expected a completed RL4CO profiling run")
    rows = []
    for case in run["cases"]:
        if not isinstance(case, str) or Path(case).name != case or case in (".", ".."):
            raise ValueError("Invalid case directory in run.json")
        folder = run_dir / case
        meta = read_json(folder / "case.json")
        if meta.get("schema") != "rl4co-bench/profile-case@1":
            raise ValueError(f"Unsupported case schema in {folder}")
        rows.append(case_metrics(folder, meta, measurement=run["measurement"]))
    if not rows:
        raise ValueError("No cases to summarize")
    migrations = migration_rows(rows)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "cases.csv", rows)
    if migrations:
        write_csv(output / "migration.csv", migrations)
    (output / "report.md").write_text(markdown(run, rows, migrations), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Directory that contains run.json")
    parser.add_argument("--output", required=True, type=Path, help="New directory for report and CSVs")
    args = parser.parse_args()
    try:
        summarize(args.run, args.output)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    print(args.output / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
