#!/bin/bash
# E140 scout: extend the Multi (2-RTU concat idiom) baseline to 20M steps on
# ForagaxSquareWaveTwoBiome-v11, 3 seeds only. Purpose: check whether Multi is
# still climbing past 10M (Model A, linear catch-up) or saturating below single
# (Model B). Config is identical to the 30-seed x 10M run except total_steps;
# seeds are 0,1,2 (idx==seed, no hyper sweep), so the first 10M overlays the
# existing run exactly and this is a true extension.
#
# 20M is 2x the 10M walltime (~6h -> ~12h); --time 14:00:00 with headroom.
# --tasks 3 => one seed per array task, all three in parallel.
# scripts/slurm.py is idempotent -- re-run after timeouts to fill missing seeds.

python scripts/slurm.py \
    --cluster clusters/vulcan-gpu-vmap-32G.json \
    --tasks 3 --time 14:00:00 --runs 3 --force \
    --entry src/rtu_ppo.py \
    -e experiments/E140-rtu-stacked/foragax-multi-20M/ForagaxSquareWaveTwoBiome-v11/9/RealTimeActorCriticMLPMulti.json
