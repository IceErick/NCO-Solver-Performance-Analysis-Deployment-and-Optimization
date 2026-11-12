# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

RL4CO is a benchmark library for reinforcement learning on combinatorial optimization (CO) problems. It is built on TorchRL (vectorized envs), TensorDict (heterogeneous state containers), PyTorch Lightning (training loops), and Hydra (configuration). The design goal is to decouple problems (environments) from methods (models/policies) so they can be composed freely.

## Commands

Dependencies are managed with `uv` (Python >= 3.10). CI installs via `pip install -e ".[dev, graph, routing]"`.

```bash
uv sync --all-extras && source .venv/bin/activate   # full local dev install

# Training (Hydra; entry point is run.py -> rl4co/tasks/train.py)
python run.py                                                          # default: AttentionModel on TSP20
python run.py experiment=routing/am env=tsp env.generator_params.num_loc=50
python run.py -m experiment=routing/am model.optimizer_kwargs.lr=1e-3,1e-4,1e-5   # multirun sweep
python run.py experiment=routing/am logger=none '~callbacks.learning_rate_monitor'

# Tests (no conftest.py; run from repo root)
pytest tests
pytest --cov=rl4co tests/*.py                  # what CI runs
pytest tests/test_envs.py -k TSPEnv            # single test / parametrization
pytest tests/test_training.py::test_a2c
pytest -n auto tests                           # xdist is available for parallelism

# Lint / format (ruff, line-length 100; rules F,E,W,I001,UP with E501 ignored)
uv run pre-commit run --all-files

# Evaluate a checkpoint
python rl4co/tasks/eval.py --problem tsp --data-path data/tsp/tsp50_test_seed1234.npz \
  --model AttentionModel --ckpt-path checkpoints/am-tsp50.ckpt --method sampling --top-p 0.95

# Generate default datasets into data/
python -c "from rl4co.data.generate_data import generate_default_datasets; generate_default_datasets('data')"
```

Docs are MkDocs (`mkdocs.yml`, built with `mkdocs-material` + `mkdocstrings`); the API reference is auto-generated from docstrings, so docstring args/returns are user-facing documentation.

## Architecture

The package has four layers: `envs` (problems) → `models` (methods) → `tasks` (entry points) → `utils` (shared machinery).

### Environments (`rl4co/envs/`)

Each problem lives in its own directory (`rl4co/envs/routing/tsp/`) with a consistent trio: `env.py`, `generator.py`, `render.py`. The environment and the data generator are **separate classes**: the `Generator` subclass implements `_generate(batch_size) -> TensorDict`, and the env takes `generator` or a `generator_params` dict (e.g. `num_loc`, `loc_distribution`).

- `RL4COEnvBase` (`rl4co/envs/common/base.py`) extends TorchRL's `EnvBase` but deliberately diverges from TorchRL rollout semantics: environments in the `done` state stay available so all instances finish before the reward is computed (rewards are usually only defined for complete solutions). A `_torchrl_mode` flag toggles the stricter behavior.
- Envs expose a class-level `name` (e.g. `TSPEnv.name == "tsp"`) that models use to auto-select embeddings and configs.
- Every env subclass must define `_reset`, `_step`, and `get_reward`, plus a `_make_spec` (TorchRL `Composite` spec) built from the generator.
- `ImprovementEnvBase` is the variant for improvement (local-search) MDPs, where the generator also produces initial solutions.
- New envs must be registered in `ENV_REGISTRY` in [rl4co/envs/__init__.py](rl4co/envs/__init__.py) and consumed via `get_env(name)`; add a matching `configs/env/<name>.yaml`.
- Data-generation helpers accept pluggable distributions via `get_sampler` (`uniform`, `normal`, `exponential`, `poisson`, `cluster`, `gaussian_mixture`, …), or any callable returning a `torch.distributions.Distribution`.

### Models (`rl4co/models/`)

Three tiers, read from the bottom up:

1. **`rl4co/models/rl/`** — RL algorithms as Lightning modules: `REINFORCE`, `A2C`, `PPO`/`n_step_PPO`, `StepwisePPO`. All extend `RL4COLitModule` ([rl4co/models/rl/common/base.py](rl4co/models/rl/common/base.py)), which owns the generic training loop, dataset/dataloader setup (`env.dataset(size, phase=...)`), batch sizes, optimizer/scheduler construction, and metric logging. Subclasses mostly implement `shared_step`. REINFORCE baselines (`rollout`, `exponential`, `mean`, `no`, `critic`) are resolved by `get_reinforce_baseline` in `rl4co/models/rl/reinforce/baselines.py`.
2. **`rl4co/models/common/`** — problem-agnostic policy abstractions, split into `constructive/`, `improvement/`, and `transductive/`. For constructive, `base.py` defines `ConstructiveEncoder` / `ConstructiveDecoder` / `ConstructivePolicy`, and `autoregressive/` specializes them into `AutoregressiveEncoder`/`Decoder`/`Policy` (encoder may be `NoEncoder` for decoder-only models), with a parallel `nonautoregressive/` set. `ConstructivePolicy` runs the whole decode loop: encode → per-step decoder logits → step env → accumulate log-likelihood/reward.
3. **`rl4co/models/zoo/`** — the concrete named methods (`am`, `pomo`, `symnco`, `mdam`, `deepaco`, `glop`, `l2d`, `matnet`, `n2s`, `neuopt`, `dact`, `eas`, `ham`, `mvmoe`, `polynet`, `ptrnet`, `nargnn`, `gfacs`, `active_search`). Each is typically a `*Policy` plus a thin model class that combines that policy with an algorithm from tier 1 (e.g. `POMO(REINFORCE)`). New models must be exported through `rl4co/models/zoo/__init__.py` and `rl4co/models/__init__.py`.

`rl4co/models/nn/` holds the reusable building blocks: attention variants (including flash attention), positional encodings, MoE, `graph/` layers, and — importantly — `env_embeddings/`, which maps a problem's raw TensorDict features into the model embedding space. Swapping an env embedding is the standard way to apply an existing model to a new problem.

### Decoding (`rl4co/utils/decoding.py`)

`get_decoding_strategy` is a registry of `greedy`, `sampling`, `multistart_greedy`, `multistart_sampling`, `beam_search`, `evaluate` (the `multistart_` prefix sets `config["multistart"] = True` on the same class). Policies hold `train_decode_type`/`val_decode_type`/`test_decode_type` (`sampling` for training, `greedy` for val/test by default). Unknown names warn and fall back to `Sampling`. `get_log_likelihood` and `process_logits` (temperature, `tanh_clipping`, top-p/top-k) live in the same module.

### Tasks and configuration

`run.py` calls `rl4co/tasks/train.py:train`, a Hydra app with `config_path="../../configs"`, `config_name="main.yaml"`. Note `pyrootutils.setup_root(..., indicator=".gitignore")` — repo-root detection is keyed on `.gitignore`, not `.git` or `pyproject.toml`.

[configs/main.yaml](configs/main.yaml) composes defaults from `callbacks/`, `logger/` (null by default), `trainer/`, `model/`, `env/`, and `experiment/`. `python run.py` alone gives `AttentionModel` on TSP20 ([configs/model/default.yaml](configs/model/default.yaml), [configs/env/default.yaml](configs/env/default.yaml)). Experiment files in `configs/experiment/{routing,scheduling,eda,graph}/` are the real entry points for a specific method + problem; they pin a model and env via `override`, e.g. `- override /model: am.yaml`. Config keys follow the Hydra-instantiation `_target_` convention (`configs/env/*.yaml` sets `_target_: rl4co.envs.TSPEnv`), so anything passed on the CLI must match the constructor signature — the generator knobs are nested under `env.generator_params.*`.

### Data and checkpoints

`rl4co/data/` provides the dataset classes — `TensorDictDataset` plus performance variants `FastTdDataset` and `TensorDictDatasetFastGeneration`, selectable per env via the `dataset_cls` argument — npz load/save helpers in `data/utils.py`, and `transforms.py` (`StateAugmentation`, used by POMO-style dihedral-8 augmentation). Datasets live under `data/` and default ones are generated on demand when `model.generate_default_data=True`. Dataloaders are plain `torch.utils.data.DataLoader` built in `RL4COLitModule`.

## Conventions and gotchas

- Optional dependencies are guarded by `try/except ImportError` at module scope (see `rl4co/envs/routing/tsp/env.py` importing `numba`-based `local_search`). `torch_geometric` (`graph` extra) and `numba`/`pyvrp` (`routing` extra) are not required for the core library — keep imports of them lazy or guarded.
- Tests run on Linux/macOS/Windows across Python 3.10–3.12 in CI, so avoid platform-specific assumptions. `test_training.py` reads the `MAC_OS_GITHUB_RUNNER` env var to force `accelerator="cpu"` on Apple Silicon.
- Tests default to tiny sizes (`num_loc=20`, `train_data_size=10`) purely for speed; they execute full Lightning `fit`/`test` cycles, so a slow change to `RL4COLitModule` or the decode loop shows up here first.
- Docs and README occasionally lag the config schema (e.g. the README still shows `env.num_loc=50`; the working form is `env.generator_params.num_loc=50`). Trust `configs/` over prose when they disagree.
- CONTRIBUTING.md mentions `black` and `make dev-install`; the actual toolchain is `ruff` (check + format) via pre-commit, and there is no Makefile.
