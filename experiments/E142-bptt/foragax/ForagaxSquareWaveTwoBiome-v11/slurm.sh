#!/bin/bash
# Submit the T-BPTT window curve at 30 seeds x 10M steps on
# ForagaxSquareWaveTwoBiome-v11.
#
# Run AFTER ../../foragax-sweep/.../process_hypers.sh, which writes each
# window's own selected (alpha, lr_scale, entropy_coef) into these configs.
# Until then they carry E139's real-time winner as a placeholder, which runs but
# is not the tuned setting for any of these windows.
#
# All four windows are run, not just the best one. The result is the shape of
# performance-vs-T, not a single winning window: a monotone rise that plateaus
# is a mechanism claim (the effect tracks the credit-assignment horizon),
# whereas one window beating the baseline is a single number beating another.
# Three points above the T=1 control is the minimum needed to tell "rises then
# plateaus" from "rises linearly".
#
# The matched baseline is E139's RealTimeActorCriticMLP on this same
# environment at 10M -- same architecture, same budget, RTRL instead of a
# truncated window. It needs no new runs; plot.sh reads it via --baseline-path.
#
# 10M steps at rollout_steps=2048 is ~4900 updates. Measured per-update cost
# rises with the window (T=1 614ms, T=16 761ms, T=32 883ms on CPU), so budget
# T=32 at roughly 1.5x the T=1 walltime.

for fov in 9; do
    for T in 1 8 16 32; do
        python scripts/slurm.py \
            --cluster clusters/vulcan-gpu-vmap-32G.json \
            --tasks 5 --time 06:00:00 --runs 30 --force \
            --entry src/rtu_ppo.py \
            -e experiments/E142-bptt/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/BPTTActorCriticMLP_T${T}.json
    done
done
