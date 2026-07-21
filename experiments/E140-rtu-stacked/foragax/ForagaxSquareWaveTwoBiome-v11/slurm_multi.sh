#!/bin/bash
# E140: run ONLY the Multi (2-RTU, concat-idiom, non-residual) baseline on
# ForagaxSquareWaveTwoBiome-v11, 30 seeds x 10M steps, d_hidden=512.
# This is the non-stacked 2-RTU reference for the "more recurrence vs residual
# redesign" comparison; the full E140 sweep lives in slurm.sh.
# scripts/slurm.py is idempotent -- re-run after timeouts to fill missing seeds.

for fov in 9; do
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLPMulti.json
done
