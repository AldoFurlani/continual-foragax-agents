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
        --tasks 24 --time 12:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e "$EV/${fov}/RealTimeActorCriticConv.json"

    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 4 --time 12:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e "$EV/${fov}/RealTimeActorCriticConvMulti.json"
done

# Search-Oracle reward ceiling (privileged current-reward info; cf. paper Fig 7).
# CPU-only, non-learning -- runs via continuing_main.py, no aperture (world mode).
python scripts/slurm.py \
    --cluster clusters/vulcan-cpu-16G.json \
    --time 09:00:00 --runs 30 --force \
    --entry src/continuing_main.py \
    -e "$EV/Baselines/Search-Oracle.json"
