# Hardware Calibration Anchor

**Purpose.** Convert "machine B is slower" from an observation into a *measurement*.
This file establishes how fast each machine's hardware actually is on a pure,
well-optimised fp32 kernel. Every rl4co workload number downstream is then read
*relative to* this anchor rather than in absolute isolation.

---

## Method

Identical on both machines, from `capture_env.py`:

```
N = 4096, dtype = float32
C = A @ B           # 2 * N^3 = 1.374e11 FLOP per call
3 warmup calls
synchronize()
t0 = perf_counter()
20 calls (3 on CPU)
synchronize()
t1 = perf_counter()
matmul_ms = (t1 - t0) / iters * 1000
TFLOPS    = 2*N^3 / (matmul_ms/1000) / 1e12
```

The `synchronize()` on both sides of the timed region is what makes this valid on
async backends (MPS and CUDA both queue work asynchronously; without it you would
be timing the *enqueue*, not the compute).

**What this is not.** It is not a prediction of rl4co performance. It bounds the
achievable compute rate; it says nothing about kernel-launch overhead, Python
dispatch, or host-device synchronisation — which is exactly what RQ1/H1 is about.

---

## Results

| Machine | Backend | matmul (ms) | **TFLOPS** | Peak fp32 | Achieved | vs 2080 Ti |
|---|---|---|---|---|---|---|
| RTX 2080 Ti | CUDA (cuBLAS) | 12.5 | **11.0** | 13.45 | 82% | 1.00× |
| Apple M2 | MPS (Metal) | 57.2 | **2.40** | ~3.6 † | ~67% † | **4.58× slower** |
| Apple M2 | CPU (Accelerate) | 121.9 | **1.13** | — | — | 9.7× slower |

† M2 GPU peak is quoted at ~3.6 TFLOPS fp32 for the 8-core variant; treat the
"achieved" figure as indicative, since Apple does not publish a spec that maps
cleanly onto a matmul roofline.

**Provenance.** MPS and CPU rows: machine-captured, `macos-arm64-mps.json`,
2026-10-08 (reproducible to 0.3% across repeated runs). CUDA row: user-reported
from an equivalent probe on the Windows box, **awaiting canonical capture** by
`capture_env.py` → `windows-cuda.json`.

---

## The 4.58× anchor, and how to use it

Define, for any workload measured on both machines:

```
H = TFLOPS(2080 Ti) / TFLOPS(other)      # hardware speed ratio
R = t(other) / t(2080 Ti)                # measured wall-clock ratio
E = R / H                                # normalised efficiency
```

`E` is the number that matters. It separates three genuinely different findings
that a raw time comparison would conflate:

| `E` | Interpretation | What it means for the paper |
|---|---|---|
| **E ≈ 1** | Workload is **compute-bound** on both. Time ratio tracks silicon. | Hardware is the explanation; MPS is a faithful-but-slower backend. |
| **E ≪ 1** (e.g. 0.2) | Workload is **overhead-bound** — both machines spend their time in launch/sync/Python, not arithmetic, so identical latency everywhere. | *Direct evidence for H1.* At these sizes the GPU is idle; the bottleneck is the autoregressive decode loop, not FLOPs. |
| **E ≫ 1** | **Backend-specific inefficiency.** MPS is leaving hardware on the table — missing/fallback kernels, silent CPU round-trips, per-op sync. | The gap is an MPS implementation artefact, not a hardware property. This is the finding worth reporting loudly. |

**The key consequence:** a raw "4.6× slower" headline is uninterpretable on its
own, because it is *also* what you would measure if both machines were perfectly
compute-bound. Only `E` distinguishes "this is the hardware" from "this is the
backend" from "neither machine is doing arithmetic at all". Report `E`.

**Prediction to test (H1):** `E` should start ≪ 1 at small `n`/small batch and
climb toward 1 as `n` and batch grow. The crossover point where `E → 1` is
precisely where the workload becomes compute-bound — that crossover is a
publishable quantity, and it is the natural definition of the "large enough
problem" boundary in RQ1.

---

## Caveats

- **Single kernel type.** Matmul is the most vendor-optimised op that exists.
  Attention, scatter/gather, and the masking ops rl4co actually leans on may sit
  far below this ceiling on MPS. Treat these numbers as an *upper bound* on
  achievable MPS throughput, never a typical case.
- **Thermal.** The MacBook Air M2 is **fanless**. Sustained runs will throttle;
  the 20-iteration loop above is far too short to expose that. If any Mac-side
  experiment runs longer than ~60 s of continuous GPU load, re-measure the
  anchor *after* the workload and report both — or the throttled number will be
  silently attributed to the backend.
- **M2 CPU is off-topic but recorded** — PyTorch on macOS/arm64 may not be
  routing through AMX. If the CPU arm enters the study, verify that separately;
  1.13 TFLOPS looks low for this chip.
- **FP32 only.** The study is pinned to FP32 by decision (unified precision for
  the comparable baseline). These numbers say nothing about fp16/bf16 throughput,
  which on Turing is a different (and much larger) number.
