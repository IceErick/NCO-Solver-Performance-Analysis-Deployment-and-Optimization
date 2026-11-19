"""Profile pinned RL4CO Attention Model TSP inference without changing the policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import NamedTuple

# Avoid writing __pycache__ into the vendored source or benchmark checkout.
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
SOURCE = BENCH.parent / "rl4co"

SWEEP_SCHEMA = "rl4co-bench/profile-sweep@1"
RUN_SCHEMA = "rl4co-bench/profile-run@1"
CASE_SCHEMA = "rl4co-bench/profile-case@1"
SESSION_SCHEMA = "rl4co-bench/session@1"


class Case(NamedTuple):
    n: int
    b: int
    k: int

    @property
    def name(self):
        return f"n{self.n}-b{self.b}-k{self.k}"

    @property
    def strategy(self):
        return "greedy" if self.k == 1 else "multistart_greedy"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")


def read_json(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return obj


def read_sweep(path, wanted):
    config = read_json(path)
    if config.get("schema") != SWEEP_SCHEMA:
        raise ValueError(f"Unknown sweep schema in {path}")
    for key in ("seed", "cpu_threads", "warmup_batches", "timed_batches", "profile_batches"):
        value = config.get(key)
        if type(value) is not int or value < (0 if key == "seed" else 1):
            raise ValueError(f"{key} must be a positive integer (seed may be zero)")

    cases = []
    for row in config["cases"]:
        case = Case(**row)
        if any(type(value) is not int or value < 1 for value in case) or case.k > case.n:
            raise ValueError(f"Invalid case n/b/k: {case}")
        cases.append(case)
    names = {case.name for case in cases}
    if not names or len(names) != len(cases):
        raise ValueError("The sweep must have at least one case and no duplicate cases")
    missing = set(wanted) - names
    if missing:
        raise ValueError(f"Unknown cases in sweep: {', '.join(sorted(missing))}")
    return config, [case for case in cases if not wanted or case.name in wanted]


def get_device(torch, smoke, device_index):
    if smoke:
        if device_index is not None:
            raise ValueError("--device-index is only used for CUDA measurements")
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use the NVIDIA workstation or run --smoke for a CPU plumbing check.")
    device_index = torch.cuda.current_device() if device_index is None else device_index
    if not 0 <= device_index < torch.cuda.device_count():
        raise ValueError(f"No visible CUDA device with index {device_index}")
    torch.cuda.set_device(device_index)
    activities = torch.profiler.supported_activities()
    if torch.profiler.ProfilerActivity.CUDA not in activities:
        raise ValueError("This PyTorch build cannot record CUDA profiler activity")
    return torch.device("cuda", device_index)


def check_calibration(path, torch, device, smoke, pin):
    if path is None:
        if smoke:
            return None
        raise ValueError("CUDA measurements require --calibration from envs/session.py")
    content = path.read_bytes()
    session = json.loads(content)
    status = session.get("calibration", {}).get("status")
    if session.get("schema") != SESSION_SCHEMA or session.get("measurement") is not (not smoke):
        raise ValueError("Calibration schema or measurement flag does not match the run")
    if session.get("backend") != device.type or status not in ({"smoke"} if smoke else {"reference", "pass"}):
        raise ValueError(f"Calibration status/backend is unsuitable for {device}: {status}")
    if session.get("provenance", {}).get("git_tree") != pin["git_tree"]:
        raise ValueError("Calibration used a different RL4CO source pin")
    if device.type == "cuda":
        env = session.get("environment", {})
        expected = {
            "host": platform.node(),
            "device_index": device.index,
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": str(torch.__version__),
        }
        for key, value in expected.items():
            if env.get(key) != value:
                raise ValueError(f"Calibration does not match the current {key}: {env.get(key)!r} != {value!r}")
    return content


def checkpoint_policy(torch, policy, path, trust_checkpoint):
    """Load only the policy weights from either a policy state dict or a Lightning save."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=not trust_checkpoint)
    except Exception as exc:
        raise ValueError(
            f"Cannot read checkpoint {path}: {exc}. Use a tensor-only state dict, or "
            "pass --trust-checkpoint only for a verified legacy Lightning file."
        ) from exc
    provenance = state.get("provenance", {}) if isinstance(state, Mapping) else {}
    if isinstance(state, Mapping):
        state = state.get("state_dict", state)
    if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
        raise ValueError("Checkpoint does not contain a model state dict")
    lightning_policy = {key.removeprefix("policy."): value for key, value in state.items() if key.startswith("policy.")}
    if lightning_policy:
        state = lightning_policy
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Policy state dict contains a non-tensor value")
    policy.load_state_dict(state, strict=True)
    fields = ("upstream_url", "upstream_sha256", "trained_on", "rl4co_git_tree")
    return {key: provenance[key] for key in fields if isinstance(provenance.get(key), str)} if isinstance(provenance, Mapping) else {}


def inference(torch, TensorDict, policy, env, coords, device, case, *, annotate):
    scope = torch.profiler.record_function if annotate else lambda _: nullcontext()
    kwargs = {"decode_type": case.strategy, "select_best": True}
    if case.k > 1:
        kwargs["num_starts"] = case.k

    with scope("benchmark.inference"):
        with scope("benchmark.h2d"):
            # TSPEnv uses TensorDict.device to allocate its action mask.
            batch = TensorDict({"locs": coords.to(device)}, batch_size=[case.b], device=device)
        with scope("benchmark.reset"):
            state = env.reset(batch).to(device)
        with scope("benchmark.policy"):
            result = policy(state, env, phase="test", max_steps=case.n, **kwargs)
        with scope("benchmark.d2h"):
            route = result["actions"].cpu()
            reward = result["reward"].cpu()
    if device.type == "cuda":
        with scope("benchmark.boundary_sync"):
            torch.cuda.synchronize(device)
    return route, reward


def validate_routes(coords, route, reward, n):
    batch = coords.size(0)
    if list(route.shape) != [batch, n] or reward.numel() != batch:
        raise ValueError(f"Unexpected policy output shapes: {route.shape} and {reward.shape}")
    routes = route.tolist()
    rewards = reward.reshape(-1).tolist()
    rows = []
    for points, tour, score in zip(coords.tolist(), routes, rewards, strict=True):
        if sorted(tour) != list(range(n)):
            raise ValueError(f"Policy returned an invalid TSP-{n} tour")
        cost = math.fsum(math.dist(points[a], points[b]) for a, b in zip(tour, tour[1:] + tour[:1], strict=True))
        if not math.isclose(-score, cost, rel_tol=2e-6, abs_tol=1e-6):
            raise ValueError(f"Policy reward {score} disagrees with recomputed route cost {cost}")
        rows.append({"route": tour, "cost": cost, "reward": score})
    return rows


def run_case(torch, TensorDict, env, policy, coords, device, config, case, folder, metadata):
    if len(coords) < case.b:
        raise ValueError(f"{case.name} requires at least {case.b} fixed instances")
    num_full_batches = len(coords) // case.b

    def batch_at(i):
        offset = (i % num_full_batches) * case.b
        return coords[offset : offset + case.b], offset

    def call(batch, annotate=False):
        return inference(torch, TensorDict, policy, env, batch, device, case, annotate=annotate)

    traces = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        traces.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.inference_mode():
        metadata["phase"] = "warmup"
        for i in range(config["warmup_batches"]):
            call(batch_at(i)[0])

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metadata["phase"] = "timing"
        timings = []
        outputs = []
        for i in range(config["timed_batches"]):
            batch, offset = batch_at(i)
            start = time.perf_counter_ns()
            route, reward = call(batch)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            timings.append({"batch_index": i, "batch_size": case.b, "first_instance_index": offset, "latency_ms": elapsed_ms})
            outputs.append((offset, route, reward))
        if device.type == "cuda":
            metadata["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)

        metadata["phase"] = "profiling"
        with torch.profiler.profile(activities=traces, with_stack=True) as profiler:
            for i in range(config["profile_batches"]):
                call(batch_at(i)[0], annotate=True)
                profiler.step()
        trace = folder / "trace.json"
        profiler.export_chrome_trace(str(trace))

    metadata["phase"] = "validation"
    with trace.open(encoding="utf-8") as f:
        events = json.load(f)["traceEvents"]
    if sum(event.get("name") == "benchmark.inference" and event.get("ph") == "X" for event in events) != config["profile_batches"]:
        raise ValueError("Profiler trace is missing one or more benchmark.inference markers")
    if device.type == "cuda" and not any(event.get("cat") == "kernel" and event.get("ph") == "X" for event in events):
        raise ValueError("Profiler trace has no GPU kernel events, so CUDA attribution cannot proceed")

    def route_rows():
        for i, (offset, route, reward) in enumerate(outputs):
            batch, _ = batch_at(i)
            for j, row in enumerate(validate_routes(batch, route, reward, case.n)):
                yield {"batch_index": i, "instance_index": offset + j, **row}

    write_rows(folder / "timings.jsonl", timings)
    write_rows(folder / "routes.jsonl", route_rows())
    metadata["num_timed_batches"] = len(timings)
    metadata["phase"] = "complete"


def run(args):
    config_path = args.config or HERE / ("t2_smoke.json" if args.smoke else "t2_sweep.json")
    config, cases = read_sweep(config_path, args.case)
    if not args.smoke and not (args.checkpoint and args.checkpoint_source and args.checkpoint_source.strip()):
        raise ValueError("Measurement requires --checkpoint and --checkpoint-source for a trained TSP Attention Model")
    if args.trust_checkpoint and args.checkpoint is None:
        raise ValueError("--trust-checkpoint requires --checkpoint")
    if args.checkpoint_source and args.checkpoint is None:
        raise ValueError("--checkpoint-source requires --checkpoint")
    output = args.output.resolve()
    if output.is_relative_to(SOURCE.resolve()):
        raise ValueError("Run output must be outside the vendored RL4CO source")

    import torch
    from tensordict import TensorDict

    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(config["seed"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = get_device(torch, args.smoke, args.device_index)
    sys.path.insert(0, str(BENCH / "instances"))
    from fixed_inputs import load_instance_set, verify_rl4co

    pin = verify_rl4co()
    calibration = check_calibration(args.calibration, torch, device, args.smoke, pin)
    sys.path.insert(0, str(SOURCE))
    import rl4co

    if not Path(rl4co.__file__).resolve().is_relative_to(SOURCE.resolve()):
        raise ValueError(f"RL4CO was imported from an unpinned path: {rl4co.__file__}")
    from rl4co.envs.routing.tsp.env import TSPEnv
    from rl4co.models.zoo.am import AttentionModelPolicy

    options = read_json(args.policy_kwargs) if args.policy_kwargs else {}
    if "env_name" in options and options["env_name"] != "tsp":
        raise ValueError("Policy kwargs must specify the TSP environment")
    policy = AttentionModelPolicy(**{"env_name": "tsp", **options})
    checkpoint_hash = None
    checkpoint_provenance = {}
    if args.checkpoint:
        checkpoint_hash = sha256(args.checkpoint)
        checkpoint_provenance = checkpoint_policy(torch, policy, args.checkpoint, args.trust_checkpoint)
        if checkpoint_provenance.get("rl4co_git_tree", pin["git_tree"]) != pin["git_tree"]:
            raise ValueError("Checkpoint was converted for a different RL4CO source pin")

    # Prepare every case's fixed inputs before any timing or output creation.
    inputs = {}
    manifests = {}
    instance_options = {"directory": args.instances} if args.instances else {}
    for n in sorted({case.n for case in cases}):
        arrays, manifests[str(n)] = load_instance_set(f"tsp{n}", **instance_options)
        inputs[n] = torch.tensor(arrays["locs"], dtype=torch.float32, device="cpu")
    policy = policy.to(device=device, dtype=torch.float32).eval()

    output.mkdir(parents=True, exist_ok=False)
    if calibration is not None:
        (output / "calibration.json").write_bytes(calibration)
    metadata = {
        "schema": RUN_SCHEMA,
        "status": "running",
        "measurement": not args.smoke,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
        "platform": platform.platform(),
        "environment": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "tensordict": version("tensordict"), "torchrl": version("torchrl"),
            "rl4co_path": str(rl4co.__file__), "cuda": torch.version.cuda,
        },
        "model": "AttentionModelPolicy",
        "execution": "eager_fp32",
        "checkpoint": {
            "sha256": checkpoint_hash, "source": args.checkpoint_source,
            "name": args.checkpoint.name if args.checkpoint else None,
            "policy_kwargs": options,
            "provenance": checkpoint_provenance,
        },
        "rl4co": pin,
        "calibration_sha256": hashlib.sha256(calibration).hexdigest() if calibration else None,
        "datasets": manifests,
        "sweep": config,
        "cases": [case.name for case in cases],
        "started_utc": utc_now(),
    }
    write_json(output / "run.json", metadata)
    try:
        successes = 0
        for case in cases:
            folder = output / case.name
            folder.mkdir()
            case_meta = {
                "schema": CASE_SCHEMA, "status": "running", "measurement": not args.smoke,
                "device": str(device), **case._asdict(), "decode_type": case.strategy,
                "profile_batches": config["profile_batches"], "peak_cuda_memory_bytes": None,
                "num_timed_batches": 0, "phase": "initializing",
            }
            write_json(folder / "case.json", case_meta)
            try:
                env = TSPEnv(generator_params={"num_loc": case.n}, check_solution=False, seed=config["seed"])
                run_case(torch, TensorDict, env, policy, inputs[case.n], device, config, case, folder, case_meta)
                case_meta["status"] = "ok"
                successes += 1
            except torch.cuda.OutOfMemoryError as exc:
                if device.type != "cuda":
                    raise
                case_meta.update(status="oom", error=str(exc))
            except BaseException as exc:
                case_meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                write_json(folder / "case.json", case_meta)
            if case_meta["status"] == "oom":
                torch.cuda.empty_cache()
            print(f"{case.name}: {case_meta['status']}", flush=True)
        if successes == 0:
            raise RuntimeError("No configured profiling case completed successfully")
        metadata["status"] = "ok"
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["finished_utc"] = utc_now()
        write_json(output / "run.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--instances", type=Path, help="verified fixed-input directory override")
    parser.add_argument("--config", type=Path, help="profile sweep JSON override")
    parser.add_argument("--case", action="append", default=[], help="run a specific case ID; repeat as needed")
    parser.add_argument("--checkpoint", type=Path, help="trained TSP AM policy state dict or Lightning checkpoint")
    parser.add_argument("--checkpoint-source", help="checkpoint provenance, e.g. training run ID and trained problem size")
    parser.add_argument("--policy-kwargs", type=Path, help="AM constructor overrides matching the checkpoint")
    parser.add_argument("--trust-checkpoint", action="store_true", help="allow object unpickling for a trusted legacy checkpoint")
    parser.add_argument("--calibration", type=Path, help="session JSON produced by envs/session.py")
    parser.add_argument("--device-index", type=int, help="visible CUDA device index")
    parser.add_argument("--smoke", action="store_true", help="CPU-only plumbing check; random weights if no checkpoint")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
