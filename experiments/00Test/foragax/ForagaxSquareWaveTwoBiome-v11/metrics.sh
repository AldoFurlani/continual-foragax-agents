#!/bin/bash
#SBATCH --account=aip-whitem
#SBATCH --job-name=R1-ForagaxSquareWaveTwoBiome-v11_foragax_ForagaxSquareWaveTwoBiome-v11_metrics
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=16
#SBATCH --output=/scratch/%u/logs/slurm-%j.out
#SBATCH --time=02:00:00

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

#$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R2-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLPReLU -f 9
#$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R2-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11 -a RealTimeActorCriticMLPReLU -f 9
#$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R2-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP_tanh -f 9

$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R2-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLPReLU_20m -f 9
