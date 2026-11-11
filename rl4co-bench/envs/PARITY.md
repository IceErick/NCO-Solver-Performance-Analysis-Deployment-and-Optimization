# Cross-Machine Parity Record

Status as of **2026-10-08**. Left column is machine-captured; right column is
user-reported and **not yet machine-captured** — fill it in by running
`capture_env.py` on the Windows box.

> **Why parity is a precondition, not a nicety.** Every cross-architecture claim
> is a claim about *the backend*. If the two environments differ in library
> versions, that claim is contaminated — a difference could be an MPS kernel
> gap, or it could be a torch version that changed the decode loop. The record
> below exists so that, months from now, you can prove the two arms were
> comparable rather than merely assert it.

---

## 1. Interpreter and framework

| | macOS (M2) | Windows (RTX 2080 Ti) | Match |
|---|---|---|---|
| Python | 3.11.17 | 3.11 (patch unconfirmed) | ⚠️ confirm patch |
| torch | 2.14.1 | 2.14.1**+cu130** | ✅ same version, different build |
| tensordict | 0.14.2 | 0.14.2 | ✅ |
| torchrl | 0.14.0 | 0.14.0 | ✅ |
| lightning | 2.6.6 | 2.6.6 | ✅ |
| numpy | 2.4.6 | 2.4.6 | ✅ |
| rl4co | 0.7.0 | 0.7.0 | ✅ |
| scipy | 1.17.1 | ? | ❓ |
| hydra-core | 1.3.7 | ? | ❓ |
| omegaconf | 2.3.1 | ? | ❓ |
| einops | 0.8.2 | ? | ❓ |
| matplotlib | 3.11.2 | ? | ❓ |
| wandb | 0.30.0 | ? | ❓ |

The torch split (CPU/MPS wheel vs `+cu130`) is **expected and correct** — it is
the accelerator binding, not a version skew. Everything else should be identical.

## 2. Source identity — the most important row

| | macOS | Windows | Match |
|---|---|---|---|
| rl4co install | editable, from repo | ? | ❓ |
| **git HEAD** | `dbd18e7a06ef9c8e35f54ab93adf6136ceeffad0` | ? | ❓ **must match** |
| tag | `v0.7.0` | ? | ❓ |
| tracked files modified | **none** | ? | ❓ **must be none** |

This is the row that actually invalidates results if it drifts. Torch patch
versions cause polite confusion; **different source revisions mean you measured
two different programs.** Verify with:

```powershell
git -C <path-to-rl4co> rev-parse HEAD
git -C <path-to-rl4co> status --short
```

## 3. Optional dependencies — the real asymmetry

| Package | macOS | Windows | Consequence |
|---|---|---|---|
| numba | 0.68.0 | ❌ missing | **`TSPEnv.local_search` is `None`** |
| pyvrp | 0.14.0 | ❌ missing | **no HGS baseline (RQ4 blocked)** |
| vrplib | 2.2.0 | ❌ missing | pyvrp instance I/O unavailable |
| torch_geometric | 2.8.0.post1 | ❌ missing | graph-based models unavailable |
| pytest / xdist | 9.1.1 / 3.8.0 | ❌ missing | can't run the test suite |
| ruff / pre-commit | 0.16.10 / 4.6.2 | ❌ missing | lint only, harmless |

**`tsp_local_search`:** macOS `AVAILABLE`, Windows `None`.
This is guarded by `try/except ImportError` at
`rl4co/envs/routing/tsp/env.py:12`, so it fails silently — the env constructs
fine and simply has no local search. For AttentionModel/TSP100 constructive
decoding this is almost certainly off the hot path, **but "almost certainly" is
not a provenance argument.** Match it before the first measurement; it is one
command.

### Bring Windows to parity

```powershell
conda activate rl4co
pip install -e ".[dev,graph,routing]"
```

`pyvrp` does double duty here: it closes the extras gap *and* supplies the HGS
baseline RQ4 needs.

## 4. Verification checklist

Run on **Windows**, in the `rl4co` conda env, from `rl4co-bench/envs/` (repo root):

```powershell
python capture_env.py          # writes windows-cuda.json — the canonical record
```

Then confirm these five things, each of which has bitten a cross-platform study
before:

1. `packages.*` matches the table in §1 exactly.
2. `rl4co.git_head` == `dbd18e7a06ef9c8e35f54ab93adf6136ceeffad0`.
3. `rl4co.git_tracked_modified` == `false`.
4. `extras.tsp_local_search` == `"AVAILABLE"` (after the parity install).
5. `backends.native_sm_supported` == `true` — already confirmed
   (`arch_list` contains `sm_75`), meaning the 2080 Ti runs **natively compiled
   SASS, with no PTX JIT fallback**. This was the one hard blocker and it is
   cleared; if it ever reads `false`, every CUDA number is JIT-warmed and the
   first-run timings are garbage.

## 5. Environment asymmetries that are NOT fixable

Recorded so they are never mistaken for a finding:

- **Device memory:** 8 GB unified (shared with OS) vs 11 GB discrete VRAM.
  The Mac will OOM first at large batch. Per the study design, **Mac OOM is
  itself a datapoint**, not a failure — but batch is a grid variable, never a
  matched constant, precisely because of this.
- **Thermals:** fanless M2 vs actively cooled 2080 Ti. The M2 throttles under
  sustained load; the 2080 Ti does not. Any Mac run longer than ~60 s of
  continuous GPU load needs a before/after anchor re-measurement.
- **Sync granularity:** Metal and CUDA have different host-device sync costs and
  different async-queue depths. This is *the object of study* (H1), not a
  nuisance to be controlled away.
