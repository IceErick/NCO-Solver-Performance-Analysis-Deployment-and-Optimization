# `envs/` — Environment Provenance

Everything here exists to answer one question: *were the two machines running
the same software?* When that answer is "no", every cross-architecture number
is inadmissible.

## Contents

| File | What it is |
|---|---|
| `capture_env.py` | Cross-platform capture tool. Run in the `rl4co` conda env on each machine. |
| `macos-arm64-mps.json` | ✅ Machine-captured, 2026-10-08. |
| `windows-cuda.json` | ⬜ **Not yet created** — produced by running `capture_env.py` on Windows. |
| `CALIBRATION.md` | Hardware speed anchor + the normalised-efficiency metric (`E`). |
| `PARITY.md` | Side-by-side parity table and the verification checklist. |

## Capturing an environment

```bash
conda activate rl4co
cd rl4co-bench/envs        # run from the rl4co repo root;  Windows: cd rl4co-bench\envs
python capture_env.py
```

Writes `<slug>.json` (slug derived from OS + accelerator) and prints the same
JSON. Takes ~30 s, most of it the matmul calibration.

`SKIP_CPU_CALIB=1` skips the CPU matmul row if you want it faster.

## Schema

```
rl4co-bench/env-capture@1
├── host         OS, chip, core split (P/E), RAM
├── python       version, executable, conda prefix
├── packages     core dependency versions
├── rl4co        install mode, git HEAD, tracked-modified flag, untracked files
├── backends     cuda/mps availability, device props, arch_list, native-SM check
├── extras       optional deps + the concrete `tsp_local_search` consequence
└── calibration  4096³ fp32 matmul timing
```

Two fields carry more weight than the rest:

- **`rl4co.git_head`** — if this differs between machines, you measured two
  different programs. Nothing else on this page matters until it matches.
- **`rl4co.git_tracked_modified`** — must stay `false`. Untracked scratch files
  (recorded separately under `git_untracked`) are fine; modified tracked files
  mean the source has drifted from the tag.

Bump `SCHEMA` in `capture_env.py` if you add or rename a field, so older records
stay interpretable.

## Rule

`rl4co-bench/` now lives **inside** the rl4co repository root, as an untracked
directory. The invariant that matters is unchanged: **no tracked file in the
rl4co source tree may be modified.** Scripts here read the package and never
write to it.

```bash
git -C "/Users/erickli/Library/CloudStorage/OneDrive-TheUniversityofNottinghamNingboChina/Code/rl4co" status --short
```

Expect only `??` lines. Any ` M` line means the source has drifted from the tag
and the environment record no longer describes what was measured.

Two consequences of living inside the tree:

- `rl4co.git_tracked_modified` in the capture JSON stays meaningful (it ignores
  untracked paths), but **`git status` will always look dirty**. Judge the tree
  by whether any line starts with ` M`, not by whether the output is empty.
- Never run `git add -A` in this repository — it would stage the harness and its
  measurement data into the research commit history. Adding `rl4co-bench/` to
  `.gitignore` would prevent this, at the cost of one tracked-file edit.
