#!/bin/bash
# Render E140 stacked-RTU figures. Assumes process_data.py has already produced
# results/E140-rtu-stacked/.../data.parquet.

set -e

EXP=experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11

# Reward switches every 250k steps (square wave: half of the 500k period).
# %.0f so macOS/BSD seq emits plain integers (default %g renders 1e+06).
# NOTE: if you paste this into an interactive zsh prompt, make it an array --
# zsh does not word-split an unquoted $SWITCHES the way bash does.
SWITCHES=$(seq -f "%.0f" 250000 250000 29750000)

# Reward curve at 30M: the depth ablation (L1 vs L2 vs L4) against plain
# RTU-PPO (single non-residual RTU) and the oracle reference. Primary E140
# figure -- "does the residual block help, and does depth help".
# Multi is deliberately excluded: it is still a 10M run and would stop a third
# of the way across the x-axis. The oracle is drawn as a horizontal line rather
# than a curve -- it is a fixed policy, so its reward is stationary (1.61 mean
# over 30 seeds, p10 1.37 / p90 1.87, measured from the 10M runs); this spans
# the full 30M axis and costs no compute.
python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --filter-alg-apertures RealTimeActorCriticMLP:9 RealTimeActorCriticMLPStacked1:9 RealTimeActorCriticMLPStacked2:9 RealTimeActorCriticMLPStacked4:9 \
    --end-frame 30000000 \
    --vertical-lines $SWITCHES \
    --horizontal-lines 1.61:Search-Oracle \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --ylim 2.2

# Learned RTU memory horizon. rtu_r_mean / rtu_r_max are the mean and max
# spectral radius r over every RTU diagonal (both branches, all blocks), logged
# every rollout via experiment.r_stats_freq. r sets the per-unit memory horizon:
# tau = -1/ln(r) steps, so r -> 1 is long memory. At init r_mean ~ 0.67
# (tau ~ 2.5 steps) and r_max ~ 0.9999 (tau ~ 15k steps); the question is
# whether training moves either, and whether depth changes that.
python src/learning_curve.py "$EXP" \
    --metrics rtu_r_mean rtu_r_max \
    --filter-alg-apertures RealTimeActorCriticMLP:9 RealTimeActorCriticMLPStacked1:9 RealTimeActorCriticMLPStacked2:9 RealTimeActorCriticMLPStacked4:9 \
    --end-frame 30000000 \
    --plot-name ForagaxSquareWaveTwoBiome-v11_rtu_spectral_radius \
    --legend

# Plasticity-vs-depth figures. DISABLED until the stacked class carries per-block
# probes (add it to _PROBED_CLASSES and generalize the metric sites in
# rtu_ppo.py, then set compute_plasticity=true in the configs). Kept here so the
# E140 format matches E139 once probing lands.
# for alg in RealTimeActorCriticMLPStacked1 RealTimeActorCriticMLPStacked2 RealTimeActorCriticMLPStacked4; do
#     python src/plasticity_compare.py "$EXP" --alg "$alg"
#     python src/plasticity_compare.py "$EXP" --alg "$alg" --mode fold --window 4500000:5500000:500
#     python src/plasticity_compare.py "$EXP" --alg "$alg" --mode fold-overlay \
#         --windows 1000000:2000000:500 4500000:5500000:500 9000000:10000000:500 \
#         --window-labels early mid late
#     python src/grad_norm_curve.py "$EXP" --alg "$alg" --log-scale
# done
# python src/plasticity_retention.py "$EXP"
