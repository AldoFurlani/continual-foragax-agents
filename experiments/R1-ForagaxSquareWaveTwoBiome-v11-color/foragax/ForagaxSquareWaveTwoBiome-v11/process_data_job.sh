#!/bin/bash
#SBATCH --account=aip-whitem
#SBATCH --job-name=R1-ForagaxSquareWaveTwoBiome-v11-color_foragax_process_data
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/%u/logs/slurm-%j.out
#SBATCH --time=03:00:00

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

$SLURM_TMPDIR/.venv/bin/python src/process_data.py experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11
