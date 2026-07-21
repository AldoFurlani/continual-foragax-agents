#!/bin/bash
# Render E140 stacked-RTU figures. Assumes process_data.py has already produced
# results/E140-rtu-stacked/.../data.parquet.

set -e

EXP=experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11

# Reward switches every 250k steps (square wave: half of the 500k period).
# %.0f so macOS/BSD seq emits plain integers (default %g renders 1e+06).
SWITCHES=$(seq -f "%.0f" 250000 250000 9750000)

# Reward curve: the depth ablation (L1 vs L2 vs L4) overlaid, with the oracle
# reference. This is the primary E140 figure -- "does stacking depth help".
python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --filter-alg-apertures Search-Oracle RealTimeActorCriticMLPStacked1:9 RealTimeActorCriticMLPStacked2:9 RealTimeActorCriticMLPStacked4:9 \
    --end-frame 10000000 \
    --vertical-lines $SWITCHES \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --ylim 2.2

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
