#!/bin/bash
# Render plasticity figures for R2-plasticity (PPO-ReLU vs RTU-PPO-ReLU).
# Assumes process_data.py has already produced
# results/R2-plasticity/.../data.parquet.

set -e

EXP=experiments/R2-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11

# Reward curve (sanity / baseline), RTU vs PPO overlaid. Top-level cross-alg
# comparison; the per-alg plasticity figures go in plots/<alg>/ below.
python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --plot-name "relu_RTUPPO_vs_PPO" \
    --filter-alg-apertures RealTimeActorCriticMLPReLU:9 ActorCriticMLPReLU:9 \
    --end-frame 10000000 \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --disable-fov \
    --font-size 20

python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --plot-name "PPO_tanh_vs_relu" \
    --filter-alg-apertures ActorCriticMLP_tanh:9 ActorCriticMLPReLU:9 \
    --end-frame 10000000 \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --disable-fov \
    --font-size 20

python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --plot-name "PPOReLU_20m" \
    --filter-alg-apertures ActorCriticMLPReLU_20m:9 \
    --end-frame 20000000 \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --disable-fov \
    --font-size 20
