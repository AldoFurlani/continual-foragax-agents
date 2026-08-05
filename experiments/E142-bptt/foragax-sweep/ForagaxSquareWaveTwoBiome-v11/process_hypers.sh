#!/bin/bash
#SBATCH --account=aip-amw8
#SBATCH --job-name=E142-bptt_foragax-sweep_ForagaxSquareWaveTwoBiome-v11_process_hypers
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

# Selects, INDEPENDENTLY PER seq_len, the best (alpha, lr_scale, entropy_coef)
# by mean_ewm_reward, writing hypers/9/BPTTActorCriticMLP_T<N>.json and (via
# update_best_config, stripping "-sweep") the matching 10M eval config.
# seq_len is a scalar in every sweep file, so it is never overwritten.
$SLURM_TMPDIR/.venv/bin/python experiments/E142-bptt/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/hypers.py
