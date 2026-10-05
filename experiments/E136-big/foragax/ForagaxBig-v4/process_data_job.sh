#!/bin/bash
#SBATCH --account=aip-amw8
#SBATCH --job-name=E136-big_foragax_ForagaxBig-v4_process_data
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --output=/scratch/%u/logs/slurm-%j.out
#SBATCH --time=01:00:00

set -e

module load arrow/19

cp -R .venv $SLURM_TMPDIR

export MPLBACKEND=TKAgg
export OMP_NUM_THREADS=1
export POLARS_MAX_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NPROC=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_PLATFORMS=cpu

$SLURM_TMPDIR/.venv/bin/python src/process_data.py experiments/E136-big/foragax/ForagaxBig-v4
