#!/bin/bash
# Eval runs for the T-BPTT window curve on ForagaxBig-v5, 30 seeds x 10M (the
# paper's unending-forager). Run AFTER process_hypers.sh writes the selected hypers.
#
# One file per seq_len -- see the v11 sweep for why the window must not be swept
# inside a single config (selection collapses each file to one winner, which is
# how DRQN's sequence_length got tuned away to 1 in XN34).
#
# rollout_steps=128 is one cue period: the environment names the best biome for
# 10 steps out of every 100, so a rollout holds about one cue event. It is also
# E141's tuned value here.
#
# num_mini_batch=8, fixed across windows so every T sees the same optimisation
# regime -- 32 gradient steps per rollout, 16 transitions each -- and the
# gradient path is the only thing that varies.
#
# Windows are {1,2,4,8,16}. With T-BPTT the shuffleable unit is a chunk of T
# consecutive steps, so a rollout holds rollout//T chunks and
#     num_mini_batch <= rollout // T
# T=16 leaves 8 chunks, so num_mini_batch cannot exceed 8. This is also why the
# update rate is capped by the window regardless of rollout:
#     grad steps / env step  =  epochs * num_mini_batch / rollout  <=  epochs / T
# At epochs=4, T=16 can never exceed 0.25 -- a quarter of E141's tuned 1.0.
# Reaching T=16 and matching E141's update granularity are mutually exclusive;
# the window is what this experiment is about, so the window wins.
#
# The ~90-step gap between cue bursts is what motivates the longer windows: the
# agent must catch a 10-step signal and hold it until the next one. T=16 covers
# only part of that, so the curve may still be rising at the top of the range.
#
# Conv only: the paper uses a CNN on RGB, E141 uses Conv, and an MLP would
# flatten the 9x9x3 aperture and discard the spatial structure a 28x28 world
# with a 9-cell view depends on.

for fov in 9; do
    for T in 1 2 4 8 16; do
        python scripts/slurm.py \
            --cluster clusters/vulcan-gpu-vmap-32G.json \
            --tasks 5 --time 12:00:00 --runs 30 --force \
            --entry src/rtu_ppo.py \
            -e experiments/E142-bptt/foragax/ForagaxBig-v5/${fov}/BPTTActorCriticConv_T${T}.json
    done
done
