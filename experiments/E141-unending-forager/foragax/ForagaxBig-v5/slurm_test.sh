#!/bin/bash
# E141 eval: single RTU-PPO (RealTimeActorCriticConv) vs Conv-Multi
# (RealTimeActorCriticConvMulti) on the unending forager ForagaxBig-v5, 30 seeds
# x 10M steps, using the swept hypers written by ../foragax-sweep/.../hypers.py.
# Run the sweep + process_hypers.sh FIRST so these configs carry the winners.
#
# --tasks == vmap width per GPU. Single Conv (1 RTU) vmaps wide; Conv-Multi
# (2 RTUs, d_hidden=512) is heavier -- start low, bump if the GPU has headroom.
# scripts/slurm.py is idempotent -- re-run after timeouts to fill missing seeds.
# NOTE: walltime is a first guess -- calibrate with one short run on the cluster.

EV=experiments/E141-unending-forager/foragax/ForagaxBig-v5

for fov in 9; do
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 12:00:00 --runs 5 --force \
        --entry src/rtu_ppo.py \
        -e "$EV/${fov}/RealTimeActorCriticConv.json"
done