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

#$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name PPO_LOP --filter-alg-apertures ActorCriticMLP_tanh:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20
#$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name PPO_Comparisons --filter-alg-apertures ActorCriticMLP_tanh:9 ActorCriticMLP_relu:9 ActorCriticMLP-l2:9 ActorCriticMLP-l2-init:9 ActorCriticMLP-reset:9 PPO_LN_2048_crelu:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20
#$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name RTU-PPO_Comparisons --filter-alg-apertures PPO-RTU_LN_2048:9 PPO-RTU_LN_2048-l2-init:9 PPO-RTU_LN_2048-l2-init_relu:9 PPO-RTU_LN_2048_crelu:9 PPO-RTU_LN_2048_relu:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20
#$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name PPO_Comparisons_2 --filter-alg-apertures ActorCriticMLP_tanh_2:9 ActorCriticMLP_relu_2:9 ActorCriticMLP-l2_2:9 ActorCriticMLP-l2-init_2:9 ActorCriticMLP-reset_2:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20

#$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name RTU-PPO_Comparisons_MLP --filter-alg-apertures RealTimeActorCriticMLP:9 RealTimeActorCriticMLP-l2-init:9 RealTimeActorCriticMLP-l2-init_relu:9 RealTimeActorCriticMLP_crelu:9 RealTimeActorCriticMLP_relu:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20

#$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name PPO_relu_vs_tanh --filter-alg-apertures ActorCriticMLP_tanh:9 ActorCriticMLP_relu:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20
#$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name RTU-PPO_relu_vs_tanh --filter-alg-apertures RealTimeActorCriticMLP:9 RealTimeActorCriticMLP_relu:9 RealTimeActorCriticMLP_crelu:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20

#$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name tanh_PPOmid_vs_PPO --filter-alg-apertures ActorCriticMLP_tanh:9 ActorCriticMLP_tanh_2:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20
#$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name relu_PPOmid_vs_PPO --filter-alg-apertures ActorCriticMLP_relu:9 ActorCriticMLP_relu_2:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --font-size 20

$SLURM_TMPDIR/.venv/bin/python src/learning_curve.py experiments/R1-ForagaxSquareWaveTwoBiome-v11-color/foragax/ForagaxSquareWaveTwoBiome-v11 --plot-name All_Agents_Barchart --filter-alg-apertures ActorCriticMLP-l2:9 ActorCriticMLP-l2_2:9 ActorCriticMLP-l2-init:9 ActorCriticMLP-l2-init_2:9 ActorCriticMLP-l2-init_relu:9 ActorCriticMLP-l2-init_relu_2:9 ActorCriticMLP-l2_relu:9 ActorCriticMLP-l2_relu_2:9 ActorCriticMLP_relu:9 ActorCriticMLP_relu_2:9 ActorCriticMLP-reset:9 ActorCriticMLP-reset_2:9 ActorCriticMLP-reset_relu:9 ActorCriticMLP-reset_relu_2:9 ActorCriticMLP_tanh:9 ActorCriticMLP_tanh_2:9 PPO_LN_2048_crelu:9 PPO-RTU_LN_2048:9 PPO-RTU_LN_2048_crelu:9 PPO-RTU_LN_2048-l2-init:9 PPO-RTU_LN_2048-l2-init_relu:9 PPO-RTU_LN_2048_relu:9 RealTimeActorCriticMLP:9 RealTimeActorCriticMLP_crelu:9 RealTimeActorCriticMLP-l2-init:9 RealTimeActorCriticMLP-l2-init_relu:9 RealTimeActorCriticMLP_relu:9 --plot-avg --disable-fov --legend-on-bar --end-frame 10000000 --horizontal-bars --plot-bar-only --font-size 10
