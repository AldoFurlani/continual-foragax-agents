#!/bin/bash
#SBATCH --account=aip-amw8
#SBATCH --job-name=X10-ForagaxWeatherWaveTwoBiome-v3_foragax-sweep_ForagaxWeatherWaveTwoBiome-v3_process_hypers
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

$SLURM_TMPDIR/.venv/bin/python experiments/X10-ForagaxWeatherWaveTwoBiome-v3/foragax-sweep/ForagaxWeatherWaveTwoBiome-v3/hypers.py

$SLURM_TMPDIR/.venv/bin/python experiments/X10-ForagaxWeatherWaveTwoBiome-v3/foragax/ForagaxWeatherWaveTwoBiome-v3/make_frozen_5m_configs.py