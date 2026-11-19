# Classical reference solvers

Use LKH-3 for TSP and PyVRP for CVRP. Run from the repository root in the
benchmark Python environment. The runner verifies each fixed NPZ against the
canonical instance manifest before solving it.

- **LKH-3:** use version 3.0.13. Obtain the source or Windows executable from
  [Keld Helsgaun's research page](https://keldhelsgaun.github.io/) and follow the
  author's build instructions. Supply the binary explicitly with `--lkh`, or
  make `LKH` available on PATH. The solver executable and source are maintained
  independently of this project. If the author's download is unavailable, the
  [ViTSP reproduction archive](https://github.com/yuhengli01/vitsp/blob/master/LKH-3.0.13.tgz)
  used for the local smoke test has SHA-256
  `8e4b70064e963e1bd24d44a6f9492fe793323ad4f0f063a6652d697cf28ad15d`.
- **PyVRP:** install the version in
  [requirements.txt](requirements.txt). This release uses
  [iterated local search](https://github.com/PyVRP/PyVRP/releases/tag/v0.13.0),
  and its integer distances are constructed from the original input coordinates.

On Linux, with `curl`, `tar`, `make`, and a C compiler available, build LKH-3
outside the project. The commands below use the verified research archive as a
fallback if the official download is unreachable.

```bash
lkh_src=$(mktemp -d)
lkh_archive="$lkh_src/LKH-3.0.13.tgz"
if ! curl -fsSL --max-time 30 'https://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz' -o "$lkh_archive"; then
  curl -fsSL 'https://raw.githubusercontent.com/yuhengli01/vitsp/master/LKH-3.0.13.tgz' -o "$lkh_archive" || exit 1
  printf '%s  %s\n' '8e4b70064e963e1bd24d44a6f9492fe793323ad4f0f063a6652d697cf28ad15d' "$lkh_archive" | sha256sum -c - || exit 1
fi
tar -xzf "$lkh_archive" -C "$lkh_src" || exit 1
make -C "$lkh_src/LKH-3.0.13" || exit 1
lkh_bin="$lkh_src/LKH-3.0.13/LKH"
```

In the active benchmark environment, choose an output directory outside the
synced workspace. Run these commands from the repository root. The CVRP
reference can take substantially longer than the TSP references.

```bash
python -m pip install -r rl4co-bench/baselines/requirements.txt
reference_out=/path/to/benchmark-output/reference
for dataset in tsp50 tsp100 tsp200; do
  python -B rl4co-bench/baselines/run.py --dataset "$dataset" --lkh "$lkh_bin" --output "$reference_out/$dataset-reference.json" || exit 1
done
python -B rl4co-bench/baselines/run.py --dataset cvrp100 --output "$reference_out/cvrp100-reference.json"
```

For a faster quality reference run, use `run_all.py` to split each dataset into
contiguous shards. Run these CPU workers in a separate session from any
calibrated GPU profiling.

```bash
python -B rl4co-bench/baselines/run_all.py --jobs 12 --lkh "$lkh_bin" --output-dir "$reference_out"
```

The orchestrator validates dataset and solver provenance, row coverage, and
shard agreement before publishing each complete JSON file. Use
`--datasets tsp100 cvrp100` to select any subset or to resume after earlier
datasets finished. It refuses to overwrite existing reference files. Merged
results mark per-instance `elapsed_ms` as diagnostic because concurrent workers
affect latency and throughput.

Use `--limit 2` for a smoke check, or `--start N --limit M` for a shard. Use
distinct output paths. The defaults give LKH-3 10 runs, each with at most N
trials for N cities, and PyVRP 10,000 iterations. Both receive seed 1. Pin these
budgets for comparisons and record any changes. CVRP has one depot, a
homogeneous fleet with up to one vehicle per customer, and no fixed cost per
vehicle. This lets the solver use any feasible number of routes, as in the
plain RL4CO CVRP environment.

The search uses integer Euclidean distances rounded at a scale of 100,000.
The runner checks that every city or customer is covered exactly once and
that CVRP routes obey the original capacities. It then recomputes each
`objective` from the original inputs using unrounded Euclidean distance in
float64, including all closing legs. Evaluate any candidate route under that
same convention. A positive value of
`100 * (candidate_objective / reference_objective - 1)` is a **reference gap**
expressed as a percentage. A heuristic reference is an achievable comparison value
and can be beaten.

Join the `instances` rows to another result by the dataset SHA-256 and
zero-based dataset `index`. TSP stores one tour of zero-based city IDs with an
implicit closing leg. CVRP stores a list of routes with depot ID 0 at both
ends, and customers numbered 1 through N. The JSON also records the search
budget, seed, software or binary identity, and machine information.

The per-instance `elapsed_ms` field includes conversion, solving, and
validation. LKH-3 additionally includes file and process overhead. It
excludes dataset loading and result output for both solvers and is sensitive
to the host and current load. Collect deployment runtime on the calibrated
workstation. Smoke timings only establish that the adapters run.
