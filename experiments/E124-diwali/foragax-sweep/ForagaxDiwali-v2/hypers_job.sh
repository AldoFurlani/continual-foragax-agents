#!/bin/bash
#SBATCH --account=aip-amw8
#SBATCH --job-name=E124-diwali_foragax-sweep_ForagaxDiwali-v2_hypers
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/logs/slurm-%j.out
#SBATCH --time=01:00:00

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

$SLURM_TMPDIR/.venv/bin/python experiments/E124-diwali/foragax-sweep/ForagaxDiwali-v2/hypers.py

$SLURM_TMPDIR/.venv/bin/python scripts/generate_frozen_configs.py experiments/E124-diwali/foragax/ForagaxDiwali-v2/9
$SLURM_TMPDIR/.venv/bin/python scripts/generate_frozen_configs.py experiments/E124-diwali/foragax/ForagaxDiwali-v2/5
