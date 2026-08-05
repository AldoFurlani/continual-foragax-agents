#!/bin/bash
# Submit the T-BPTT window curve at 30 seeds x 1M steps on ForagaxBig-v5.
#
# Run AFTER ../../foragax-sweep/ForagaxBig-v5/process_hypers.sh, which writes
# each window's own selected hyperparameters into these configs.
#
# Unlike ForagaxSquareWaveTwoBiome-v11 there is no matched real-time baseline
# here: E141's RealTimeActorCriticConv runs at rollout_steps=128, whereas these
# need 512 to fit a 16-step window. The T=1 control is therefore the reference
# -- same architecture and data flow, recurrent gradient path removed.
#
# Windows are {1,4,8,16}; see the sweep script for the rollout_steps constraint.

for fov in 9; do
    for arch in Conv MLP; do
        for T in 1 4 8 16; do
            python scripts/slurm.py \
                --cluster clusters/vulcan-gpu-vmap-32G.json \
                --tasks 5 --time 06:00:00 --runs 30 --force \
                --entry src/rtu_ppo.py \
                -e experiments/E142-bptt/foragax/ForagaxBig-v5/${fov}/BPTTActorCritic${arch}_T${T}.json
        done
    done
done
