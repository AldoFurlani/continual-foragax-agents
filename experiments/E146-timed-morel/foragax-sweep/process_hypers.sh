#!/bin/bash
#SBATCH --account=aip-amw8
#SBATCH --job-name=E146-timed-morel_foragax-sweep_process_hypers
#SBATCH --mem-per-cpu=128G
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/logs/slurm-%j.out
#SBATCH --time=2:00:00

set -e

module load arrow/19

cp -R .venv $SLURM_TMPDIR

export MPLBACKEND=TKAgg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NPROC=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_PLATFORMS=cpu

# Selects the best configuration of each config file by mean_ewm_reward over
# the 1M sweep, writing <condition>/hypers/<fov>/<agent>.json and (via
# update_best_config, stripping "-sweep") the matching 10M eval config under
# experiments/E146-timed-morel/foragax/. Then writes the <agent>_frozen_5M copy
# of each eval config (learning stops at 5M steps).
# Run only after every sweep job has finished.
$SLURM_TMPDIR/.venv/bin/python experiments/E146-timed-morel/foragax-sweep/hypers.py

$SLURM_TMPDIR/.venv/bin/python experiments/E146-timed-morel/foragax/make_frozen_5m_configs.py
