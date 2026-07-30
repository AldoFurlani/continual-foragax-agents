#!/bin/bash
# E140: run ONLY the depth-1 Stacked agent (RealTimeActorCriticMLPStacked1 -- a
# single [RTU + MLP] residual block) on ForagaxSquareWaveTwoBiome-v11,
# 30 seeds x 10M steps, d_hidden=512. This is the within-family baseline for the
# depth ablation; the full four-arm run is in slurm.sh.
#
# Hypers come from the 1M sweep in ../../foragax-sweep/ (stacked_1_slurm.sh
# there, then process_data_job.sh, then process_hypers.sh, which writes the
# selected cell into 9/RealTimeActorCriticMLPStacked1.json). Re-run this after
# any re-selection so the 10M results match the config.
#
# NOTE: compute_plasticity is OFF in the config -- the stacked class is not yet
# in _PROBED_CLASSES, so plasticity metrics would be a silent no-op.
# scripts/slurm.py is idempotent -- re-run after timeouts to fill missing seeds.

EXP=experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11

for fov in 9; do
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e "$EXP/${fov}/RealTimeActorCriticMLPStacked1.json"
done
