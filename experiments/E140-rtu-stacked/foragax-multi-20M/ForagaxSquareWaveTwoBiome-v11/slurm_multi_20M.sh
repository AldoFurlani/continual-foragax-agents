#!/bin/bash
# E140 scout: extend the Multi (2-RTU concat idiom) baseline to 50M steps on
# ForagaxSquareWaveTwoBiome-v11, 3 seeds only. Purpose: check whether Multi is
# still climbing past 10M (Model A, linear catch-up) or saturating below single
# (Model B). Config uses the swept eval hypers (entropy 0.1, actor alpha 1e-3,
# critic lr_scale 10.0) -- same as the existing 30-seed x 10M run; seeds are
# 0,1,2 (idx==seed, no hyper sweep), so the first 10M overlays that run exactly
# and this is a true extension.
#
# 50M is 5x the 10M walltime (~6h -> ~30h); --time 36:00:00 with headroom.
# If the partition caps below this, rely on rtu_ppo's Checkpoint: re-running
# this (idempotent) script resumes/fills unfinished seeds.
# --tasks 3 => one seed per array task, all three in parallel.

python scripts/slurm.py \
    --cluster clusters/vulcan-gpu-vmap-32G.json \
    --tasks 3 --time 36:00:00 --runs 3 --force \
    --entry src/rtu_ppo.py \
    -e experiments/E140-rtu-stacked/foragax-multi-20M/ForagaxSquareWaveTwoBiome-v11/9/RealTimeActorCriticMLPMulti.json
