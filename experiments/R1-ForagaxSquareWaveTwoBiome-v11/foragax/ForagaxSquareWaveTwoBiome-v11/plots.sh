#!/bin/bash
#SBATCH --account=aip-whitem
#SBATCH --job-name=R1-ForagaxSquareWaveTwoBiome-v11_foragax_ForagaxSquareWaveTwoBiome-v11_plots
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

# PPO comparisons
#$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name PPO-comparisons_50seeds --filter-alg-apertures ActorCriticMLP:9 ActorCriticMLP-l2:9 ActorCriticMLP-l2-init:9 ActorCriticMLP-reset:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20
#$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name PPO-LOP_50seeds --filter-alg-apertures ActorCriticMLP:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20
#$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name PPO-LOP_noLN --filter-alg-apertures ActorCriticMLP_noLN:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20

# RTU-PPO Comparisons (128)
#$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name RTU-PPO_comparisons_128_50seeds --filter-alg-apertures PPO-RTU_LN_128_1:9 PPO-RTU_LN_128_1-l2-init:9 PPO-RTU_LN_128_1_crelu:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20

# RTU-PPO Comparisons (2048)
#$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name RTU-PPO_comparisons_2048_50seeds --filter-alg-apertures PPO-RTU_LN_2048:9 PPO-RTU_LN_2048-l2-init:9 PPO-RTU_LN_2048_crelu:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20

# Compare RTU-PPO with baseline PPO LOP and PPO w/ head reset (using 2048 rollout cuz 128 rollout is trash)
#$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name PPO_vs_RTU-PPO_2048_50seeds --filter-alg-apertures ActorCriticMLP:9 ActorCriticMLP-reset:9 PPO-RTU_LN_2048:9 PPO-RTU_LN_2048_crelu:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20

# Compare layernorm vs no layernorm
#$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name noLN_vs_LN_50seeds --filter-alg-apertures ActorCriticMLP:9 ActorCriticMLP_noLN:9 PPO_2048:9 PPO-RTU_LN_2048:9 PPO-RTU_2048:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20

# Compare PPO and RTU-PPO with relu and tanh
$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name PPO_relu_vs_tanh --filter-alg-apertures ActorCriticMLP:9 ActorCriticMLP_relu:9 ActorCriticMLP-reset:9 ActorCriticMLP-reset_relu:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20
$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name RTU-PPO_2048_relu_vs_tanh --filter-alg-apertures PPO-RTU_LN_2048:9 PPO-RTU_LN_2048_relu:9  PPO-RTU_LN_2048_crelu:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20
$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name RTU-PPO_128_relu_vs_tanh --filter-alg-apertures PPO-RTU_LN_128_1:9 PPO-RTU_LN_128_1_relu:9  PPO-RTU_LN_128_1_crelu:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20
