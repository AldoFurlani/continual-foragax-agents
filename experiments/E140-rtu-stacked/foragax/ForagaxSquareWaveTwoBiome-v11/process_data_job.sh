#!/bin/bash
#SBATCH --account=aip-amw8
#SBATCH --job-name=E140-rtu-stacked_foragax_ForagaxSquareWaveTwoBiome-v11_process_data
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=16
#SBATCH --output=/scratch/%u/logs/slurm-%j.out
# 30M runs: ~4x the 10M job (3x the timesteps, and targets now
# includes 30M so sample_types went 46 -> 61). The 10M job took ~40 min.
#SBATCH --time=06:00:00

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

$SLURM_TMPDIR/.venv/bin/python src/process_data.py experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11
