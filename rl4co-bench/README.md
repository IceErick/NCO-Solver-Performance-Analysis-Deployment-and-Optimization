# rl4co Deployment-Scale Benchmark Harness

Measurement harness and proposal for the cross-problem / cross-backend study of
rl4co. Lives at the rl4co **repository root as an untracked directory**: it reads
the package but never writes to it, and no tracked source file is ever modified.

## The two arms

| Arm | Machine | Backend | Device | OS |
|---|---|---|---|---|
| **CUDA** | RTX 2080 Ti (Turing, sm_75, 11 GB) + Intel Core Ultra 5, 16 GB | CUDA 13.0 / cuDNN 9.24 | dedicated | Windows |
| **MPS** | MacBook Air M2 (8 GB unified, fanless) | Metal / MPS | integrated | macOS 27.0.1 |

Both run an editable rl4co 0.7.0 at git `dbd18e7`, in a conda env named `rl4co`.

## Layout

```
rl4co-bench/
├── proposal/      LaTeX research proposal                          ← START HERE
├── envs/          environment provenance + hardware calibration
├── profiling/     RQ1: where inference time goes, and how hotspots move
├── optimization/  RQ2: compile / fusion / batching / precision gains
├── deployment/    RQ3: optimal config under a gap ≤ δ SLA
├── results/       raw measurements (the evidence)
└── weekly/        research log
```

## Where things stand (2026-10-08)

**Done**
- Both conda environments built and verified; core dependency versions aligned.
- Mac arm smoke-tested end-to-end, including rl4co's own test suite (18 passed).
- **CUDA arm unblocked:** `arch_list` contains `sm_75`, so the 2080 Ti runs
  natively compiled SASS with no PTX JIT fallback. Measured 11.0 TFLOPS fp32
  (82% of peak) — healthy.
- **Hardware anchor established:** 2080 Ti is **4.58× faster** than the M2 GPU on
  pure fp32 matmul. See `envs/CALIBRATION.md` for how this converts raw time
  ratios into a normalised-efficiency metric `E`.

**Open**
- `pip install -e ".[dev,graph,routing]"` on Windows → supplies `pyvrp` (the RQ4
  HGS baseline) and makes `TSPEnv.local_search` available on both arms.
- Run `python capture_env.py` on Windows → `envs/windows-cuda.json`.
- Confirm the Windows rl4co git HEAD equals `dbd18e7a06ef9c8e35f54ab93adf6136ceeffad0`.

**Not started** (deliberately — environment first)
- Phase 0 MPS compatibility probe.
- Any measurement.

## Study decisions already fixed

- Precision pinned to **FP32** (`precision=32-true`, `matmul_precision=highest`)
  for the comparable baseline.
- **Batch size is a grid variable**, never matched across arms. Mac OOM is a
  datapoint, not a failure.
- **AET / energy is CUDA-side only** — macOS exposes no per-process GPU energy.
  The cross-platform comparable metric set is: latency, throughput, peak memory.
- Cross-model axis (GASE) **dropped**; RQ5 is cross-problem + cross-architecture.
- First target workload: **AttentionModel / TSP100**.

## Operational rule

The repo — and therefore this harness — lives under **OneDrive CloudStorage**.
Pause OneDrive sync before any timed run. Background upload of checkpoints, logs,
and datasets corrupts wall-clock measurements in a way that is invisible in the
results and very hard to diagnose afterwards. Dataset, checkpoint, and log paths
should point outside the synced tree entirely.

## Hard constraint

**Do not modify any tracked file in the rl4co repository.** It is a live research
working copy; edits invalidate provenance and diverge the code from what the
results were produced with. Environment problems get fixed in the conda env.
Scratch scripts belong here or in `/tmp`, never alongside the package source.

Check before publishing results — expect only `??` lines, never ` M`:

```bash
git status --short
```

Never run `git add -A` in the repo: it would stage the harness and its data into
the research commit history.
