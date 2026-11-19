"""Produce per-instance reference solutions using LKH-3 or PyVRP."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

from datetime import datetime, timezone
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "instances"))
from fixed_inputs import FIXED_DIR, load_instance_set  # noqa: E402


DISTANCE_SCALE = 100_000
PYVRP_VERSION = "0.14.0"


def tsp_reference(locs, executable, seed, runs, timeout):
    started = time.perf_counter_ns()
    n = len(locs)
    coords = [
        f"{idx} {x * DISTANCE_SCALE:.17g} {y * DISTANCE_SCALE:.17g}"
        for idx, (x, y) in enumerate(locs.astype(np.float64), 1)
    ]
    with tempfile.TemporaryDirectory(prefix="rl4co-lkh-") as work:
        work = Path(work)
        (work / "instance.tsp").write_text(
            f"NAME: reference\nTYPE: TSP\nDIMENSION: {n}\n"
            "EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n"
            + "\n".join(coords)
            + "\nEOF\n",
            encoding="ascii",
        )
        (work / "instance.par").write_text(
            "PROBLEM_FILE = instance.tsp\nOUTPUT_TOUR_FILE = solution.tour\n"
            f"SEED = {seed}\nRUNS = {runs}\nMAX_TRIALS = {n}\nTRACE_LEVEL = 1\n",
            encoding="ascii",
        )
        result = subprocess.run(
            [str(executable), "instance.par"],
            cwd=work,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
        tour_path = work / "solution.tour"
        if result.returncode or not tour_path.is_file():
            raise RuntimeError(
                f"LKH-3 did not produce a tour (exit code {result.returncode}): "
                + (result.stdout + result.stderr).strip()[-3000:]
            )
        tour_text = tour_path.read_text(encoding="ascii")
        _, section, payload = tour_text.partition("TOUR_SECTION")
        nodes, terminator, _ = payload.split("EOF", maxsplit=1)[0].partition("-1")
        if not section or not terminator:
            raise RuntimeError("LKH-3 output lacks a terminated TOUR_SECTION")
        route = [int(token) - 1 for token in nodes.split()]

    version = re.search(r"\bLKH-(\d+\.\d+(?:\.\d+)?)\b", result.stdout + tour_text)
    if version and not version[1].startswith("3."):
        raise RuntimeError(f"Expected LKH-3 but executable reported {version[0]}")
    if sorted(route) != list(range(n)):
        raise RuntimeError("Invalid LKH-3 tour: expected every city exactly once")
    return {
        "route": route,
        "objective": route_length(locs, route),
        "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
        "reported_version": version[1] if version else None,
    }


def route_length(locs, route):
    ordered = locs[route].astype(np.float64)
    return float(np.linalg.norm(ordered - np.roll(ordered, 1, axis=0), axis=1).sum())


def cvrp_reference(locs, depot, demand, capacity, seed, iterations):
    from pyvrp import Client, Depot, Location, ProblemData, VehicleType, solve
    from pyvrp.stop import MaxIterations

    started = time.perf_counter_ns()
    coords = np.vstack([depot, locs]).astype(np.float64)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    integer_distances = np.floor(distances * DISTANCE_SCALE + 0.5).astype(np.int64)
    data = ProblemData(
        locations=[Location(x, y) for x, y in coords],
        clients=[Client(idx, delivery=[int(load)]) for idx, load in enumerate(demand, 1)],
        depots=[Depot(0)],
        vehicle_types=[VehicleType(num_available=len(locs), capacity=[capacity])],
        distance_matrices=[integer_distances],
        duration_matrices=[np.zeros_like(integer_distances)],
    )
    result = solve(data, stop=MaxIterations(iterations), seed=seed, collect_stats=False)
    if not result.is_feasible():
        raise RuntimeError("PyVRP did not find a feasible solution within the iteration budget")

    # PyVRP 0.14 has zero-based client activity indices, separate from locations.
    routes = [
        [0, *(activity.idx + 1 for activity in route if activity.is_client()), 0]
        for route in result.best.routes()
    ]
    served = [idx for route in routes for idx in route[1:-1]]
    if sorted(served) != list(range(1, len(locs) + 1)) or any(
        sum(demand[idx - 1] for idx in route[1:-1]) > capacity for route in routes
    ):
        raise RuntimeError("Invalid PyVRP routes: missing/repeated customer or excess load")

    return {
        "routes": routes,
        "objective": sum(route_length(coords, route) for route in routes),
        "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
    }


def run(args):
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite {args.output}")

    try:
        arrays, dataset = load_instance_set(args.dataset, directory=args.instances_dir)
    except KeyError as exc:
        raise ValueError(f"Unknown fixed dataset: {args.dataset}") from exc
    count = len(arrays["locs"])
    end = count if args.limit is None else args.start + args.limit
    if not 0 <= args.start < end <= count:
        raise ValueError(f"Requested [{args.start}, {end}) outside {count} rows")

    problem = dataset["problem"].lower()
    if problem == "tsp":
        executable = shutil.which(args.lkh)
        if executable is None:
            raise ValueError("LKH-3 executable not found. Pass --lkh /path/to/LKH.")
        executable = Path(executable).resolve()
        solver = {
            "name": "LKH-3",
            "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "runs": args.lkh_runs,
            "max_trials": dataset["size"],
            "timeout_s": args.lkh_timeout_s,
        }
    elif problem == "cvrp":
        try:
            version = importlib.metadata.version("pyvrp")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"Install pyvrp=={PYVRP_VERSION} for the CVRP reference.") from exc
        if version != PYVRP_VERSION:
            raise ValueError(f"Expected pyvrp=={PYVRP_VERSION}; found {version}.")
        solver = {"name": "PyVRP", "version": version, "iterations": args.pyvrp_iterations}
    else:
        raise ValueError(f"Unsupported reference problem: {problem}")

    started = datetime.now(timezone.utc).isoformat()
    rows = []
    for idx in range(args.start, end):
        locs = arrays["locs"][idx]
        try:
            if problem == "tsp":
                row = tsp_reference(
                    locs, executable, args.seed, args.lkh_runs, args.lkh_timeout_s
                )
                solver["reported_version"] = row.pop("reported_version")
            else:
                row = cvrp_reference(
                    locs,
                    arrays["depot"][idx],
                    arrays["demand"][idx],
                    int(arrays["capacity"][idx].item()),
                    args.seed,
                    args.pyvrp_iterations,
                )
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"{solver['name']} failed on {args.dataset}[{idx}]: {exc}") from exc
        rows.append({"index": idx, **row})

    out = {
        "schema_version": 1,
        "dataset": {"name": args.dataset, **dataset},
        "solver": {**solver, "seed": args.seed, "distance_scale": DISTANCE_SCALE},
        "session": {
            "started_utc": started,
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "instances": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {len(rows)} {solver['name']} solutions to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Fixed dataset name, e.g. tsp100")
    parser.add_argument(
        "--instances-dir", type=Path, default=FIXED_DIR, help="Directory holding the verified NPZ"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0, help="First zero-based dataset row")
    parser.add_argument("--limit", type=int, help="Optional row count for a smoke run or shard")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lkh", default="LKH", help="Path to an LKH-3 executable, or name on PATH")
    parser.add_argument("--lkh-runs", type=int, default=10)
    parser.add_argument("--lkh-timeout-s", type=int, default=120, help="Whole-process safety limit")
    parser.add_argument("--pyvrp-iterations", type=int, default=10_000)
    args = parser.parse_args()
    if not 1 <= args.seed < 2**31:
        parser.error("--seed must be in [1, 2147483647]")
    if min(args.lkh_runs, args.lkh_timeout_s, args.pyvrp_iterations) < 1:
        parser.error("Reference solver budgets and timeout must be positive")
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Reference run failed: {exc}\n")


if __name__ == "__main__":
    main()
