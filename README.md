# continual-foragax-agents

JAX-based RL agents (DQN, DRQN, PPO, RTU-PPO) trained on the
[continual-foragax](https://pypi.org/project/continual-foragax/) environment
family. Sweep tooling is built on
[PyExpUtils](https://github.com/andnp/pyexputils) and
[RlEvaluation](https://pypi.org/project/RlEvaluation/) and follows the
methodology in *Empirical Design in Reinforcement Learning* (Patterson et al.):
sweep hyperparameters at small scale, pick the best configuration per
algorithm, then run a larger production study with those frozen hypers.

---

## The workflow at a glance

Every recent experiment in `experiments/` follows these five stages. The
canonical worked example, used throughout this README, is
`experiments/E136-big/`, which produced the results for Figure 7 in the RLC submission.

```
1. Sweep                experiments/E136-big/foragax-sweep/ForagaxBig-v5/slurm.sh
                        ↓ submits hyperparameter sweep jobs (10 seeds, 1M steps), we use k%-tuning [1]

2. Pick best hypers     experiments/E136-big/foragax-sweep/ForagaxBig-v5/hypers_job.sh
                        ↓ runs hypers.py + generate_frozen_configs.py
                        ↓ writes configs with selected hypers and frozen configs into experiments/E136-big/foragax/ForagaxBig-v5/

3. Full run             experiments/E136-big/foragax/ForagaxBig-v5/slurm.sh
                        ↓ submits evaluation jobs (30 new seeds, 10M steps)

4. Process data         experiments/E136-big/foragax/ForagaxBig-v5/process_data_job.sh
                        ↓ runs src/process_data.py

5. Plot                 experiments/E136-big/foragax/ForagaxBig-v5/plot.sh
                        ↓ runs src/learning_curve.py + src/learning_bar.py
```

The two top-level directories — `foragax-sweep/` and `foragax/` — are the same
shape (`<experiment>/<env>/<aperture>/<alg>.json`). The sweep side carries
hyperparameter arrays and uses 1M steps; the evaluation side carries single
values selected by `hypers.py` and runs to 10M steps.

---

## Local development

### Prerequisites
- Python 3.11 (`pyproject.toml` requires `>=3.11, <3.13`).
- Rust (`rustup`).
- `swig` installed globally — `pipx install swig` or `brew install swig`. Needed
  to build `box2d-py`.

### Install
The project is a `pyproject.toml` package, so install editable into a venv:

```bash
uv sync
```

If you want pre-commit hooks:

```bash
bash dev-setup.sh
```

### Smoke test
Run a single seed of a real experiment locally:

```bash
uv run src/rtu_ppo.py -e experiments/E136-big/foragax-sweep/ForagaxBig-v5/9/PPO_LN_128.json -i 0
```

You should see training logs printing a progress bar. Let it run for a
minute, then cancel — this is your fast iteration loop while editing code.

The two in-use top-level entry points are:
- `src/rtu_ppo.py` — PPO and RTU-PPO.
- `src/continuing_main.py` — DQN and DRQN (continuing-task setting).

---

## Compute Canada setup

This works on Vulcan, and Fir — pick the matching cluster JSON
in `clusters/` when you submit jobs. The examples below lead with Vulcan
because that is what the `E136-big` example scripts use.

```bash
ssh vulcan
cd ~/scratch
git clone git@github.com:steventango/continual-foragax-agents.git
cd continual-foragax-agents

# One-time: install uv if you don't already have it
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install the project + CUDA-enabled JAX into .venv/
uv sync --group cuda
```

`uv sync --group cuda` resolves `pyproject.toml` + `uv.lock`, prefers
prebuilt wheels (avoiding the long Rust compile of `polars-runtime-*`
that breaks login-node memory limits), and includes the optional `cuda`
dependency group for GPU jobs.

> **Allocation accounts.** `scripts/slurm.py` auto-detects your SLURM
> account from `sacctmgr` (priority `rrg-` > `aip-` > `def-`, randomized
> within a tier per job to spread load). Pass `--account=<name>` on the
> CLI to pin a specific account. The per-experiment `*_job.sh` files
> still hardcode `--account=` and need manual edits.

Confirm `.venv/` is in the project root. Each new shell needs:

```bash
source .venv/bin/activate
sq                               # check job status
```

---

## The sweep workflow

The five stages, each with its file in the canonical example.

### 1. Sweep — `foragax-sweep/<env>/slurm.sh`

`experiments/E136-big/foragax-sweep/ForagaxBig-v5/slurm.sh` is a list of calls
to `scripts/slurm.py`, one per algorithm:

```bash
python scripts/slurm.py \
    --cluster clusters/vulcan-gpu-vmap-32G.json \
    --tasks 24 --time 03:00:00 --runs 10 --force \
    --entry src/rtu_ppo.py \
    -e experiments/E136-big/foragax-sweep/ForagaxBig-v5/9/PPO-RTU_LN_128_HT.json
```

**Sweep configs** under `9/` (the aperture size) carry hyperparameters as
arrays. For example, `9/PPO_LN_128.json` sweeps `entropy_coef` over
`[0.01, 0.1, 1.0]` and three Adam learning rates. `total_steps: 1_000_000`
as per k%-tuning [1] (10% of the 10M-step evaluation budget);
`experiment.seed_offset: 1_000_000` keeps sweep seeds disjoint from
evaluation seeds.

**Cluster JSONs** in `clusters/` choose the resource shape — names encode
the hardware (memory, GPU type); task count and time are passed at the
call site via `--tasks N` and `--time HH:MM:SS`:
- `vulcan-gpu-vmap-32G.json` / `-vmap.json` (64G) / `-vmap-128G.json` —
  vmap'd PPO across multiple seeds on one GPU; pick the memory tier you need.
- `vulcan-gpu.4-vmap.json` — quarter L40S (softMIG)

Run with:

```bash
bash experiments/E136-big/foragax-sweep/ForagaxBig-v5/slurm.sh
```

`scripts/slurm.py` only schedules **missing** seeds — re-running the same
script after jobs finish picks up whatever didn't complete. When everything is
done it reports nothing to schedule.

### 2. Pick best hypers + generate frozen configs — `foragax-sweep/<env>/hypers_job.sh`

Once the sweep results are in, submit:

```bash
sbatch experiments/E136-big/foragax-sweep/ForagaxBig-v5/hypers_job.sh
```

This is a 1-hour CPU job that runs:

1. **`hypers.py`** — uses `RlEvaluation.hypers.select_best_hypers` on the
   `mean_ewm_reward` metric (mean statistic, threshold 0.05, prefer high) per
   algorithm. It writes the chosen flat config to
   `foragax-sweep/<env>/hypers/<aperture>/<alg>.json`, updates the production
   config at `foragax/<env>/<aperture>/<alg>.json`, and emits
   `choices.tex` / `default.tex` / `selected.tex` summary tables.
2. **`scripts/generate_frozen_configs.py experiments/E136-big/foragax/ForagaxBig-v5/9`**
   — for every non-frozen config in that directory, writes a sibling
   `<alg>_frozen_5M.json` that adds `freeze_steps: 5_000_000` and renames the
   agent to `<alg>_frozen_5M`. These are used for the post-freeze transfer
   phase of continual experiments.

### 3. Full run — `foragax/<env>/slurm.sh`

`experiments/E136-big/foragax/ForagaxBig-v5/slurm.sh` mirrors stage 1, but
points at the production configs (single hyperparameter values,
`total_steps: 10_000_000`, `seed_offset: 0`) and bumps the budget to
`--runs 30 --time 06:00:00`.

```bash
bash experiments/E136-big/foragax/ForagaxBig-v5/slurm.sh
```

Idempotent in the same way — re-run to fill in missing seeds.

### 4. Process data — `foragax/<env>/process_data_job.sh`

```bash
sbatch experiments/E136-big/foragax/ForagaxBig-v5/process_data_job.sh
```

A 100 GB / 16-task / 1-hour CPU job that runs:

```bash
python src/process_data.py experiments/E136-big/foragax/ForagaxBig-v5
```

`process_data.py` loads each algorithm's results, downsamples around milestone
steps (1M / 5M / 10M with widening intervals), and emits the aggregated
parquet that the plotting scripts consume.

### 5. Plot — `foragax/<env>/plot.sh`

```bash
bash experiments/E136-big/foragax/ForagaxBig-v5/plot.sh
```

Each line invokes `src/learning_curve.py` or `src/learning_bar.py` with
`--filter-alg-apertures <alg>:<aperture>` and `--metric ewm_reward_5` to
produce one figure per comparison. Tweak the filter list to add or remove
algorithms / apertures. Plots are written under the experiment directory.

---

## Pulling results back to your laptop

```bash
# on your laptop
bash scripts/sync_results.sh
```

You need to update it to point at your cluster and experiment directory. The
intended workflow is: run `process_data_job.sh` on the cluster (stage 4) so
the heavy aggregation happens there, then `sync_results.sh` pulls only the
resulting `*.parquet` files (plus a few videos) back. The raw per-seed
databases are big enough that pulling them locally is impractical for
RAM/disk reasons, but they stay on the cluster so you can change aggregation
or add new metrics by re-running `process_data.py` without redoing the
training runs. Local re-runs of `src/learning_curve.py` /
`src/learning_bar.py` then iterate quickly off-cluster.

---

## Repo layout

```
src/
    continuing_main.py      # DQN / DRQN entry point
    rtu_ppo.py              # PPO / RTU-PPO entry point
    process_data.py         # aggregate per-alg results into parquet
    learning_curve.py       # learning-curve plot
    learning_bar.py         # bar plot
    algorithms/             # agent implementations
    representations/        # encoders / RTU / etc.
    environments/           # environment shims
    problems/               # registry of (env, representation, gamma, ...) tuples
    experiment/             # ExperimentModel + sweep utilities
    utils/                  # shared helpers (results, paths, checkpointing)

experiments/E<NNN>-<name>/
    foragax-sweep/<env>/<aperture>/<alg>.json    # sweep configs (arrays)
    foragax-sweep/<env>/slurm.sh                 # stage 1
    foragax-sweep/<env>/hypers.py                # called by stage 2
    foragax-sweep/<env>/hypers_job.sh            # stage 2
    foragax-sweep/<env>/hypers/<aperture>/*.json # selected best hypers
    foragax/<env>/<aperture>/<alg>.json          # evaluation configs (best hypers)
    foragax/<env>/slurm.sh                       # stage 3
    foragax/<env>/process_data_job.sh            # stage 4
    foragax/<env>/plot.sh                        # stage 5

clusters/                   # Slurm resource templates (vulcan / cedar / fir)
scripts/
    slurm.py                # cluster job submission
    local.py                # local multi-seed driver
    generate_frozen_configs.py
    ...                     # other one-off helpers
```

A sweep config is a single JSON file describing one algorithm and its
hyperparameter ranges:

```jsonc
{
    "agent": "PPO_LN_128",
    "problem": "Foragax",
    "total_steps": 1000000,
    "metaParameters": {
        "environment": { "env_id": "ForagaxBig-v5", "aperture_size": 9, "observation_type": "rgb" },
        "experiment":  { "seed_offset": 1000000 },
        "entropy_coef": [0.01, 0.1, 1.0],          // sweep
        "optimizer_actor": {
            "name": "Adam",
            "alpha": [1e-3, 3e-4, 1e-4],           // sweep
            "beta1": 0.9, "beta2": 0.999, "eps": 1e-5
        }
        // ... fixed hyperparameters
    }
}
```

The production config (auto-written by stage 2) has the same shape but with a
single value for each previously-swept field, `total_steps: 10_000_000`, and
`seed_offset: 0`.

---

## Dependencies

The load-bearing libraries (full list in `pyproject.toml`):

- [PyExpUtils-andnp](https://github.com/andnp/pyexputils) — experiment-running
  framework (sweep expansion, missing-seed detection, Slurm submission).
- [RlEvaluation](https://pypi.org/project/RlEvaluation/) — best-hypers
  selection in `hypers.py`.
- [RlGlue-andnp](https://github.com/andnp/rlglue) — agent / environment
  protocol.
- [ReplayTables-andnp](https://pypi.org/project/ReplayTables-andnp/) — replay
  buffers for off-policy agents.
- [ml-instrument](https://pypi.org/project/ml-instrument/) — metric collection.
- [continual-foragax](https://pypi.org/project/continual-foragax/) — the
  environment.
- [JAX](https://github.com/jax-ml/jax), [Optax](https://github.com/google-deepmind/optax),
  [Flax](https://github.com/google/flax), [dm-haiku](https://github.com/google-deepmind/dm-haiku),
  [flashbax](https://github.com/instadeepai/flashbax) — JAX stack.

---

## FAQ

**What's a good size for cluster jobs?**
Aim for jobs that run between 1 and 12 hours; ~3 hours is the sweet spot on
the Compute Canada queue. The vmap-based JSONs in `clusters/` (e.g.
`vulcan-gpu-vmap-32G.json`) bundle many seeds into one job and are highly
efficient — pass `--tasks N` to set how many seeds get packed per vmap. To
estimate task count for a config:

```python
import experiment.ExperimentModel as Experiment
exp = Experiment.load('experiments/E136-big/foragax-sweep/ForagaxBig-v5/9/PPO_LN_128.json')
print(exp.numPermutations())
```

**How do I get code from my laptop to the cluster?**
Git. Push to a remote (private GitHub fork is fine), pull on the cluster. Tag
checkpoints (`git tag icml-2026-submission`) before destabilising changes.

**Some of my jobs failed / timed out.**
Just re-run the same `slurm.sh`. `scripts/slurm.py` detects missing seeds and
schedules only those. If it exits immediately and reports no work, that means
nothing is missing — clear out (or rename) the relevant entries under
`results/` if you want to force re-runs.

**Can I use GPUs?**
Yes — every `clusters/vulcan-gpu-*` and `clusters/fir-gpu-*` JSON is a
GPU-allocated template. PPO/RTU-PPO benefit most from `vulcan-gpu-vmap-*`,
which packs multiple seeds onto one GPU.

**Where does each experiment write its results?**
Under the experiment directory the config lives in, in the layout produced by
`PyExpUtils`. `src/process_data.py` knows how to find them given the
experiment root.

---

## References

[1] Mesbahi, Golnaz, Panahi, Parham Mohammad, Mastikhina, Olya, Tang, Steven,
White, Martha, & White, Adam. (2025). *Position: Lifetime tuning is
incompatible with continual reinforcement learning.* In International
Conference on Machine Learning.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@misc{tang2026forager,
      title={Forager: a lightweight testbed for continual learning with partial observability in RL},
      author={Steven Tang and Xinze Xiong and Anna Hakhverdyan and Andrew Patterson and Jacob Adkins and Jiamin He and Esraa Elelimy and Parham Mohammad Panahi and Martha White and Adam White},
      year={2026},
      eprint={2605.01131},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.01131},
}
```

---

## Contributing

See `CONTRIBUTING.md`. Things go out of date fast — when you hit a snag
during setup or a workflow stage, please update this README in the same PR
that fixes it.
