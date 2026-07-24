#!/bin/bash
# Plot single RTU-PPO vs Multi RTU-PPO from the 50M-step head-to-head, overlaid
# on one axis. Produces TWO pdfs -- one truncated at 20M, one at the full 50M --
# via distinct --plot-name (learning_curve.py otherwise reuses one filename).
#
# Assumes process_data_job.sh has produced
#   results/E140-rtu-stacked/foragax-multi-20M/ForagaxSquareWaveTwoBiome-v11/data.parquet
# (both algs live in that one parquet -- no cross-experiment stitching).
#
# Run from the repo root:
#   bash experiments/E140-rtu-stacked/foragax-multi-20M/ForagaxSquareWaveTwoBiome-v11/plot.sh

set -e

EXP=experiments/E140-rtu-stacked/foragax-multi-20M/ForagaxSquareWaveTwoBiome-v11
ENV=ForagaxSquareWaveTwoBiome-v11
ALGS="RealTimeActorCriticMLP:9 RealTimeActorCriticMLPMulti:9"

# No per-switch vertical lines: at this timescale (up to ~199 switches at 50M)
# they read as a field of dots. The curve's own oscillation shows the switching.
plot_window() {
    local end_frame=$1 label=$2
    python src/learning_curve.py "$EXP" \
        --metrics ewm_reward \
        --filter-alg-apertures $ALGS \
        --end-frame "$end_frame" \
        --plot-name "${ENV}_ewm_reward_curve_${label}" \
        --legend-on-bar \
        --plot-avg \
        --horizontal-bars \
        --ylim 2.2
}

plot_window 20000000 20M
plot_window 50000000 50M

echo "Done -> $EXP/plots/${ENV}_ewm_reward_curve_{20M,50M}.pdf"
