# RL4CO deployment benchmark

The November milestone implements T1 and T2 from the [project proposal](proposal/proposal.pdf).
The study uses a fixed, trained solver and one NVIDIA GPU workstation. The
vendored RL4CO source in the sibling directory stays read-only.

| Task | Method | Evidence produced on the workstation |
| --- | --- | --- |
| T1 | Verify the source pin and fixed TSP/CVRP instances, run LKH-3 and PyVRP, and repeat a fixed CUDA calibration | Input hashes, validated reference routes and costs, calibration timings, driver and GPU telemetry |
| T2 | Profile eager FP32 Attention Model on TSP-100 and vary instance size, batch size, and decoding budget | CPU/CUDA traces, unprofiled latency samples, routes, and a hotspot comparison |

See the [Linux RTX 4090 runbook](RUN_NOVEMBER.md) for the commands and output
locations. The [input manifest](instances/fixed/manifest.json) pins 1,024
instances for each of TSP-50, TSP-100, TSP-200 and CVRP-100.
[Baseline parameters](baselines/README.md) document the heuristic budgets and
objective convention. Use the [session guide](envs/README.md) when reviewing
calibration provenance.

The default T2 sweep uses the same pretrained TSP-100 policy at all instance
sizes. The anchor is 100 cities, batch size 16, and one greedy start. The sweep
varies city count across 50, 100 and 200, batch size across 1, 8, 16 and 64, and
decoding budget across 1, 4, 8 and 16 deterministic starts. It also includes a
larger interaction case. Case metadata records out-of-memory limits. The
report separates CUDA kernel activity, host launch and synchronisation calls,
traced host operations and Python frames. It explains the limits of each
attribution.

The historic macOS and cross-backend records under `envs/` predate the
single-workstation design. The current protocol uses `envs/session.py` and
writes to a fresh output path outside the repository.
