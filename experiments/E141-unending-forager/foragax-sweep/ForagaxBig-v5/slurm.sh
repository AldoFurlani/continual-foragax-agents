#!/bin/bash
# E141: hyperparameter sweep for single RTU-PPO (RealTimeActorCriticConv) and the
# Conv-Multi 2-RTU variant (RealTimeActorCriticConvMulti) on the unending forager
# ForagaxBig-v5 (4 biomes, extinction, global cue, RGB CNN). Tuned at the short 1M
# horizon (tune-short / eval-long: selected configs run at 10M in ../../foragax/).
#
# Grid: actor alpha {1e-3, 3e-4, 1e-4} x critic lr_scale {0.1, 1.0, 10} x
# entropy_coef {0.01, 0.1, 1.0} = 27 cells x --runs seeds. Same protocol as the
# 2-biome E140 sweep and the paper's grid. seed_offset=1M keeps selection seeds
# disjoint from the 30 eval seeds.
#
# --tasks == vmap width per GPU (memory-bound). Single Conv is 1 RTU (light) so it
# vmaps wide like XN35. Conv-Multi is 2 RTUs at d_hidden=512 (heavier) -- start
# low and bump if the assigned GPU has headroom (rollout=128 here is small, so it
# may tolerate more than the E140 rollout-2048 Multi did). scripts/slurm.py is
# idempotent -- re-run after timeouts to fill missing seeds.
# NOTE: walltime is a first guess -- calibrate with one short run on the cluster.

SW=experiments/E141-unending-forager/foragax-sweep/ForagaxBig-v5

for fov in 9; do
    # Single RTU-PPO (conv + main RTU)
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 24 --time 03:00:00 --runs 10 --force \
        --entry src/rtu_ppo.py \
        -e "$SW/${fov}/RealTimeActorCriticConv.json"

    # Conv-Multi RTU-PPO (conv + 2 RTUs)
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 4 --time 04:00:00 --runs 10 --force \
        --entry src/rtu_ppo.py \
        -e "$SW/${fov}/RealTimeActorCriticConvMulti.json"
done
