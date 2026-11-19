# Run T1 and T2 on the Linux RTX 4090 workstation

Run these commands in the SSH shell on the workstation. Use one GPU and
keep the environment, checkpoints, and results outside the checkout. Keep
other GPU jobs off the selected card while measuring. The existing `rl4co`
Conda environment is suitable if the preflight below succeeds.

## 1. Get the code and verify the environment

From your benchmark checkout:

```bash
git fetch origin
git switch codex/november-t1-t2
git pull --ff-only

conda activate rl4co
export WORK="$HOME/fyp/nco-bench-work"
mkdir -p "$WORK"
nvidia-smi --query-gpu=index,name,driver_version,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)"
python -B -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; assert torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python -B rl4co-bench/instances/fixed_inputs.py verify
```

Only use GPU 0 if it is idle and reserved for the study. Retain the same
physical GPU for all comparison sessions. If the environment is missing,
create one with Python 3.12. This fallback installs a CUDA 13.0 build of
PyTorch, which requires a compatible NVIDIA driver.

```bash
conda create -y -p "$WORK/env" --override-channels -c conda-forge python=3.12 pip
conda activate "$WORK/env"
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
PYTHONDONTWRITEBYTECODE=1 python -m pip install -c rl4co-bench/envs/cuda-constraints.txt -e ./rl4co -r rl4co-bench/baselines/requirements.txt
python -m pip check
```

The benchmark imports the pinned source directly. A stale editable RL4CO
path in the existing environment therefore does not select the measured
implementation. PyVRP should match the version in
[baselines/requirements.txt](baselines/requirements.txt) for reference runs.

## 2. Prepare the trained model

Use the published TSP-100 checkpoint. The converter verifies its SHA-256,
loads tensors, and maps weights into the pinned RL4CO Attention Model.
Conversion does not train the model or change its weights.

```bash
curl -fL 'https://raw.githubusercontent.com/wouterkool/attention-learn-to-route/c9abf41ac2f878a55b20dc7e829bc942bb999631/pretrained/tsp_100/epoch-99.pt' -o "$WORK/epoch-99.pt"
python -B rl4co-bench/profiling/convert_am_tsp100.py "$WORK/epoch-99.pt" "$WORK/am-tsp100-rl4co.pt"
```

Keep both files for future sessions. The converter refuses to overwrite an
existing output.

## 3. Calibrate and run the full T2 sweep

Use a fresh output directory each time. Run the first session with
`--enrol-reference`. Run later sessions with
`--reference "$FIRST_RUN/calibration.json"` instead.

```bash
RUN="$WORK/rtx4090-t2-$(date -u +%Y%m%dT%H%M%SZ)"
python -B rl4co-bench/run_session.py --output "$RUN" --enrol-reference \
  --checkpoint "$WORK/am-tsp100-rl4co.pt" \
  --checkpoint-source 'Kool et al. ICLR 2019, TSP-100, epoch 99 (c9abf41)'
```

A successful run writes:

- `$RUN/calibration.json` and `pip-freeze.txt` for calibration and environment provenance.
- `$RUN/profile/run.json` and one directory per case with the CPU/CUDA trace,
  separate unprofiled timing samples, and validated routes.
- `$RUN/report/report.md`, `cases.csv`, and `migration.csv` for the latency
  and trace analysis. Use the CSVs to compare like-for-like slices of the sweep.

The `--case n100-b16-k1` option runs the anchor alone. Omit it for the full
sweep. A partial run can diagnose setup and is separately identified by its
recorded case list.

## 4. Produce T1 reference routes and costs

Build LKH-3 once using the exact Linux commands in the
[classical solver guide](baselines/README.md). Set `lkh_bin` to that binary.
Install the pinned PyVRP version if needed. Run separately from the CUDA
profiling session so CPU solver work does not affect the measured host timing.

```bash
python -m pip install -r rl4co-bench/baselines/requirements.txt
REF="$WORK/references-$(date -u +%Y%m%dT%H%M%SZ)"
for dataset in tsp50 tsp100 tsp200; do
  python -B rl4co-bench/baselines/run.py --dataset "$dataset" --lkh "$lkh_bin" \
    --output "$REF/$dataset-reference.json" || break
done
python -B rl4co-bench/baselines/run.py --dataset cvrp100 --output "$REF/cvrp100-reference.json"
```

For long jobs, use a `tmux` session so a dropped SSH connection does not end
the run. Archive the raw outputs and JSON manifests together. T1 records
heuristic reference values for each fixed instance. T2 attributes measured
costs to activity on this workstation and reports its profiling limits.
