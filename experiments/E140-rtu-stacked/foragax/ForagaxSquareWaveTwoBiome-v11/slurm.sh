#!/bin/bash
# E140: stacked RTU-PPO (LRU/S5-style [RTU + MLP] residual blocks) at depths
# 1/2/4 on ForagaxSquareWaveTwoBiome-v11, 30 seeds x 10M steps, plus the
# Multi (2-RTU concat-idiom) non-residual baseline and the Search-Oracle
# reward reference. The three Stacked depths are the depth ablation (same code
# path, only n_blocks varies; L1 is the within-family baseline); Multi vs
# Stacked-2 isolates "more recurrence" from "the residual redesign" (all at
# d_hidden=512). scripts/slurm.py is idempotent -- re-run to fill missing seeds.
#
# To run ONLY the Multi baseline, use slurm_multi.sh instead.
#
# NOTE: compute_plasticity is OFF in the configs -- the stacked class is not yet
# in _PROBED_CLASSES, so plasticity metrics would be a silent no-op. This run is
# reward-curves only. Flip compute_plasticity to true once the per-block probes
# are wired (then the plasticity block in plot.sh applies).

for fov in 9; do
    # Stacked RTU-PPO, depth 1 (within-family baseline)
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLPStacked1.json

    # Stacked RTU-PPO, depth 2
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLPStacked2.json

    # Stacked RTU-PPO, depth 4 (matches Lu et al. S5-in-PPO depth)
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLPStacked4.json

    # Multi RTU-PPO (2 stacked RTUs, concat idiom -- non-residual baseline)
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLPMulti.json
done

# Search-Oracle (reward reference, same as E139)
python scripts/slurm.py \
    --cluster clusters/vulcan-cpu.json \
    --time 01:00:00 --runs 30 --force \
    --entry src/continuing_main.py \
    -e experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11/Baselines/Search-Oracle.json
