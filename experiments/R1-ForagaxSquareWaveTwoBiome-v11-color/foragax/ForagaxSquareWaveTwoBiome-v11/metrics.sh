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

$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP-l2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP-l2-init -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP-l2-init_relu -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP-l2_relu -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP-reset -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP-reset_relu -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP_relu -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP_tanh -f 9

$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a PPO-RTU_LN_2048 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a PPO-RTU_LN_2048-l2-init -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a PPO-RTU_LN_2048-l2-init_relu -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a PPO-RTU_LN_2048_crelu -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a PPO-RTU_LN_2048_relu -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a PPO_LN_2048_crelu -f 9

$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a RealTimeActorCriticMLP -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a RealTimeActorCriticMLP-l2-init -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a RealTimeActorCriticMLP-l2-init_relu -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a RealTimeActorCriticMLP_crelu -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a RealTimeActorCriticMLP_relu -f 9

$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP-l2_2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP-l2-init_2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP-l2-init_relu_2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP-l2_relu_2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP-reset_2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP-reset_relu_2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP_relu_2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a ActorCriticMLP_tanh_2 -f 9

$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a PPO-RTU_LN_2048_2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a PPO-RTU_LN_2048-l2-init_2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a PPO-RTU_LN_2048-l2-init_relu_2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a PPO-RTU_LN_2048_crelu_2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a PPO-RTU_LN_2048_relu_2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a PPO_LN_2048_crelu_2 -f 9

$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a RealTimeActorCriticMLP_2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a RealTimeActorCriticMLP-l2-init_2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a RealTimeActorCriticMLP-l2-init_relu_2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a RealTimeActorCriticMLP_crelu_2 -f 9
$SLURM_TMPDIR/.venv/bin/python scripts/plot_metrics.py -e experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 -a RealTimeActorCriticMLP_relu_2 -f 9
