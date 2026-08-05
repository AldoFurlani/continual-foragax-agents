#!/bin/bash
# Hyperparameter sweep for the T-BPTT RTU-PPO agents on ForagaxBig-v5, tuned at
# 100k steps (10% of the 1M eval; tune-short / eval-long).
#
# One file per seq_len -- see the sibling ForagaxSquareWaveTwoBiome-v11 sweep
# for why the window must not be swept inside a single config.
#
# Windows are {1,4,8,16} here, not {1,8,16,32}. rollout_steps=512 with
# num_mini_batch=32 requires 512//T to be divisible by 32, capping T at 16.
# Reaching T=32 would need rollout_steps=1024, halving the update count again.
#
# Both architectures are swept: Conv matches E141's tuned arch on this env, MLP
# matches the one used on ForagaxSquareWaveTwoBiome-v11 so the two environments
# are comparable at fixed architecture.
#
# Grid per file: alpha x lr_scale x entropy_coef = 27 cells, x --runs seeds.

for fov in 9; do
    for arch in Conv MLP; do
        for T in 1 4 8 16; do
            python scripts/slurm.py \
                --cluster clusters/vulcan-gpu-vmap-32G.json \
                --tasks 5 --time 03:00:00 --runs 10 --force \
                --entry src/rtu_ppo.py \
                -e experiments/E142-bptt/foragax-sweep/ForagaxBig-v5/${fov}/BPTTActorCritic${arch}_T${T}.json
        done
    done
done
