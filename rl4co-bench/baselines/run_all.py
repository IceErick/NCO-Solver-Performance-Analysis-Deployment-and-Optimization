"""Build complete classical references, optionally using concurrent CPU workers."""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

from datetime import datetime, timezone
from pathlib import Path

from run import FIXED_DIR, load_instance_set


DATASETS = ("tsp50", "tsp100", "tsp200", "cvrp100")
RUNNER = Path(__file__).resolve().with_name("run.py")


def solve_shards(shards, name, args):
    processes = {}
    try:
        for start, end, path in shards:
            command = [
                sys.executable, "-B", str(RUNNER),
                "--dataset", name,
                "--instances-dir", str(args.instances_dir),
                "--start", str(start),
                "--limit", str(end - start),
                "--lkh", args.lkh,
                "--output", str(path),
            ]
            processes[start, end] = subprocess.Popen(command)

        pending = set(processes)
        while pending:
            for start, end in list(pending):
                status = processes[start, end].poll()
                if status is None:
                    continue
                if status != 0:
                    raise RuntimeError(f"{name}[{start}:{end}] exited with status {status}")
                pending.remove((start, end))
            if pending:
                time.sleep(0.25)
    finally:
        running = [process for process in processes.values() if process.poll() is None]
        for process in running:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        for process in running:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def merge_shards(shards, dataset):
    rows = []
    starts = []
    solver = None
    machine = None
    next_index = 0
    for start, end, path in shards:
        if start != next_index or end <= start:
            raise ValueError(f"{path.name}: noncontiguous shard [{start}, {end})")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["schema_version"] != 1 or data["dataset"] != dataset:
            raise ValueError(f"{path.name}: dataset or schema mismatch")
        shard_rows = data["instances"]
        if [row["index"] for row in shard_rows] != list(range(start, end)):
            raise ValueError(f"{path.name}: expected rows {start} through {end - 1}")
        if any(
            not math.isfinite(row["objective"]) or row["objective"] <= 0
            or not math.isfinite(row["elapsed_ms"]) or row["elapsed_ms"] < 0
            for row in shard_rows
        ):
            raise ValueError(f"{path.name}: invalid objective or elapsed time")

        session = dict(data["session"])
        starts.append(session.pop("started_utc"))
        if solver is None:
            solver, machine = data["solver"], session
        elif data["solver"] != solver or session != machine:
            raise ValueError(f"{path.name}: solver or environment mismatch")
        rows.extend(shard_rows)
        next_index = end
    if next_index != dataset["count"]:
        raise ValueError(f"Expected {dataset['count']} rows and found {next_index}")

    return {
        "schema_version": 1,
        "dataset": dataset,
        "solver": solver,
        "session": {
            **machine,
            "started_utc": min(starts),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "concurrent_workers": len(shards),
            "elapsed_ms_valid_for_latency_claims": False,
            "elapsed_ms_purpose": "Diagnostic only; use a separate calibrated latency run.",
        },
        "instances": rows,
    }


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: args.output_dir / f"{name}-reference.json" for name in args.datasets}
    for path in outputs.values():
        if path.exists():
            raise ValueError(f"Refusing to overwrite {path}; select remaining datasets with --datasets")
    for name, path in outputs.items():
        _, entry = load_instance_set(name, directory=args.instances_dir)
        count = entry["count"]
        workers = min(args.jobs, count)
        print(f"Solving {name}: {count} instances with {workers} worker(s)", flush=True)
        with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=args.output_dir) as folder:
            folder = Path(folder)
            shards = [
                (idx * count // workers, (idx + 1) * count // workers, folder / f"{idx}.json")
                for idx in range(workers)
            ]
            solve_shards(shards, name, args)
            merged = merge_shards(shards, {"name": name, **entry})
            complete = folder / "merged.json"
            with complete.open("x", encoding="utf-8") as stream:
                json.dump(merged, stream, indent=2, allow_nan=False)
                stream.write("\n")
            os.link(complete, path)  # Publish only complete files without overwriting.
        print(f"Wrote {count} validated reference solutions to {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instances-dir", type=Path, default=FIXED_DIR)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--jobs", type=int, default=1, help="Max concurrent shards for one dataset")
    parser.add_argument("--lkh", default="LKH", help="Path to an LKH-3 executable, or name on PATH")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets must not contain duplicates")
    try:
        run(args)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Reference run failed: {exc}\n")


if __name__ == "__main__":
    main()
