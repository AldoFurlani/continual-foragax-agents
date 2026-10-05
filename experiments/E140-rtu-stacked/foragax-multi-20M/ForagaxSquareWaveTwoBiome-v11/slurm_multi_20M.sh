#!/bin/bash
# E140 scout: run single RTU-PPO (RealTimeActorCriticMLP) AND Multi (2-RTU
# concat idiom) to 50M steps on ForagaxSquareWaveTwoBiome-v11, 3 seeds each.
# Purpose: a matched, same-experiment head-to-head at 50M -- does Multi keep
# climbing past 10M and close on single (Model A), or saturate below it
# (Model B)? Both use their own swept eval hypers (single: entropy 0.1, actor
# alpha 3e-4, critic lr_scale 10.0; multi: entropy 0.1, actor alpha 1e-3,
# critic lr_scale 10.0), both compute_plasticity=false. Seeds are 0,1,2
# (idx==seed, no hyper sweep).
#
# NOTE: single here is a fresh 50M run in THIS experiment, not the E139 10M run
# (which had plasticity probing on); its first 10M need not overlay E139. Multi
# matches the existing E140 10M run (same hypers, plasticity off), so its first
# 10M does overlay -- a free determinism check.
#
# 50M is 5x the 10M walltime (~6h -> ~30h); --time 36:00:00 with headroom.
# If the partition caps below this, rely on rtu_ppo's Checkpoint: re-running
# this (idempotent) script resumes/fills unfinished seeds.
# --tasks 3 => one seed per array task, all three in parallel.

EXP=experiments/E140-rtu-stacked/foragax-multi-20M/ForagaxSquareWaveTwoBiome-v11/9

for alg in RealTimeActorCriticMLP RealTimeActorCriticMLPMulti; do
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 3 --time 36:00:00 --runs 3 --force \
        --entry src/rtu_ppo.py \
        -e "$EXP/${alg}.json"
done
