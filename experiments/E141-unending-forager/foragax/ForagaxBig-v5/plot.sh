#!/bin/bash
# Plot single RTU-PPO vs Conv-Multi RTU-PPO on the unending forager ForagaxBig-v5,
# overlaid on one axis (cf. paper Fig 7). Both algs live in one parquet.
#
# Assumes process_data_job.sh has produced
#   results/E141-unending-forager/foragax/ForagaxBig-v5/data.parquet
#
# Run from the repo root:
#   bash experiments/E141-unending-forager/foragax/ForagaxBig-v5/plot.sh

set -e

EXP=experiments/E141-unending-forager/foragax/ForagaxBig-v5
# Search-Oracle has no aperture (world mode) -> no :9 suffix.
ALGS="Search-Oracle RealTimeActorCriticConv:9 RealTimeActorCriticConvMulti:9"

# No per-switch vertical lines (unending task -- switches are behaviour-driven).
# ylim 0.35 matches the paper's Fig 7 average-reward scale (~0-0.3).
python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --filter-alg-apertures $ALGS \
    --end-frame 10000000 \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --ylim 0.35
