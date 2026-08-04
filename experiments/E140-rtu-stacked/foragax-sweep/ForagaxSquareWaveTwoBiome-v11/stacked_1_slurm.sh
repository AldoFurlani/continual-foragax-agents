#!/bin/bash
# E140: hyperparameter sweep for ONLY the depth-1 Stacked agent
# (RealTimeActorCriticMLPStacked1 -- a single [RTU + MLP] residual block) on
# ForagaxSquareWaveTwoBiome-v11, at the short 1M horizon. This is the
# within-family baseline for the depth ablation; the full four-arm sweep lives
# in slurm.sh.
#
# Grid: actor alpha {1e-3, 3e-4, 1e-4} x critic lr_scale {0.1, 1.0, 10} x
# entropy_coef {0.01, 0.1, 1.0} = 27 cells x --runs seeds. seed_offset=1M keeps
# the selection seeds disjoint from the 30 eval seeds.
#
# --tasks 5 is the vmap width the 10M depth-1 eval already ran at; depth 1 is the
# lightest arm in the family, so bump it if the assigned GPU has headroom.
# scripts/slurm.py is idempotent -- re-run after timeouts to fill missing seeds.
#
# Then: process_data_job.sh, then process_hypers.sh (which selects over EVERY
# agent present in the sweep directory, not just this one).

SW=experiments/E140-rtu-stacked/foragax-sweep/ForagaxSquareWaveTwoBiome-v11

for fov in 9; do
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 4 --time 02:00:00 --runs 10 --force \
        --entry src/rtu_ppo.py \
        -e "$SW/${fov}/RealTimeActorCriticMLPStacked1.json"
done
