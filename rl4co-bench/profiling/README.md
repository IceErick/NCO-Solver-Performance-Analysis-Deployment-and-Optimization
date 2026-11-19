# Attention Model TSP profiling (T2)

Run on the designated NVIDIA workstation in a CUDA-enabled PyTorch environment.
The main workflow in the [benchmark README](../README.md) creates the session
calibration and runs the profiler and reducer. The profiler imports the pinned
vendored RL4CO, leaves it untouched, and verifies the canonical fixed inputs.
Put checkpoints and output outside the synced repository.

## Checkpoint

Use your own trained TSP-100 RL4CO Attention Model if it matches the model you
want to study. The runner accepts a tensor-only policy state dict or a
Lightning state dict containing `policy.*` keys, loads it strictly, and records
its hash and source. Supply `--policy-kwargs` with the model constructor options
if the checkpoint does not use the default architecture.

A reproducible alternative is the published TSP-100 weights from [Kool et al.,
Attention, Learn to Solve Routing Problems!](https://github.com/wouterkool/attention-learn-to-route/tree/c9abf41ac2f878a55b20dc7e829bc942bb999631).
Run from the repository root:

```bash
WORK="$HOME/nco-t2"
mkdir -p "$WORK"
curl -fL 'https://raw.githubusercontent.com/wouterkool/attention-learn-to-route/c9abf41ac2f878a55b20dc7e829bc942bb999631/pretrained/tsp_100/epoch-99.pt' \
  -o "$WORK/epoch-99.pt"
python -B rl4co-bench/profiling/convert_am_tsp100.py \
  "$WORK/epoch-99.pt" "$WORK/am-tsp100-rl4co.pt"
```

The converter checks the published file's SHA-256 and changes the tensor layout
for the vendored RL4CO implementation. Use `$WORK/am-tsp100-rl4co.pt` as
the checkpoint and `Kool et al. ICLR 2019, TSP-100, epoch 99 (c9abf41)` as the
checkpoint source. Keep the same weights for every case.

## Sweep and timing

The anchor is TSP-100, batch size 16, greedy decoding. The default sweep changes
one factor at a time through instance sizes 50/100/200, batch sizes 1/8/16/64,
and decode budgets 1/4/8/16. One additional case combines TSP-200, batch size 64
and eight starts. Budget one uses the policy's native greedy choice at every
step. Larger budgets run that policy from a deterministic set of distinct
starting cities and keep the best tour. The first action at budget one is learned,
so its candidate set is not necessarily nested inside the larger budgets.

Each case performs four warmups, then 32 separately timed batches of verified
fixed coordinates in deterministic order. The first measured batch is profiled
again in a separate pass. Wall time includes input transfer, reset, eager
FP32 policy execution, output transfer, and a final CUDA synchronisation.
Loading the model, reading files and checking returned routes are outside the
timed region. The traces contain CUDA and CPU operations plus Python
stack events when supported by the PyTorch build. Use the wall timings for
latency and the instrumented traces to examine where time goes. A case that
runs out of GPU memory is labelled with the phase of failure.

The standalone profiler supports `--case n100-b16-k1` to select specific cases.
`--smoke` is an explicit CPU plumbing check and has `measurement=false` in
its output. Its default random weights and timings do not support research
conclusions.

## Outputs

`run.json` records the source pin, checkpoint hash, instance hashes and settings.
Each case has `case.json` with status, dimensions and peak allocated CUDA memory,
`trace.json` for Chrome trace viewers, `timings.jsonl` for wall times and
`routes.jsonl` with route and independently recomputed Euclidean cost per timed
instance. Timed batches can wrap around the fixed dataset for the largest batch
size. Join solver quality on the instance-set hash and instance index.
