#!/bin/bash
# Eval runs for the T-BPTT window curve on ForagaxBig-v5 at 30 seeds x 10M (the
# paper's unending-forager). Run AFTER process_hypers.sh writes each window's
#
# One file per seq_len -- see the v11 sweep for why the window must not be swept
# inside a single config (selection collapses each file to one winner, which is
# how DRQN's sequence_length got tuned away to 1 in XN34).
#
# rollout_steps=128 is one cue period: the environment names the best biome for
# 10 steps out of every 100, so a rollout holds about one cue event. It is also
# E141's tuned value here.
#
# num_mini_batch=16 is FIXED across windows, so every T sees the same
# optimisation regime -- 64 gradient steps per rollout, 8 transitions each --
# and the window is the only thing that varies.
#
# Windows are {1,2,4,8}. With T-BPTT the shuffleable unit is a chunk of T
# consecutive steps, so a rollout holds rollout//T chunks and
#     num_mini_batch <= rollout // T
# At rollout 128 and num_mini_batch 16 that caps T at 8. Going to T=16 would
# need num_mini_batch 8, i.e. half the updates (0.25x E141 rather than 0.5x);
# that is the planned follow-up IF the curve is still rising at T=8, and the
# {1,2,4,8} overlap would then measure the regime shift at four points rather
# than assuming it.
#
# Conv only: the paper uses a CNN on RGB, E141 uses Conv, and an MLP would
# flatten the 9x9x3 aperture and discard the spatial structure that a 28x28
# world with a 9-cell view depends on.

for fov in 9; do
    for T in 1 2 4 8; do
        python scripts/slurm.py \
            --cluster clusters/vulcan-gpu-vmap-32G.json \
            --tasks 5 --time 12:00:00 --runs 30 --force \
            --entry src/rtu_ppo.py \
            -e experiments/E142-bptt/foragax/ForagaxBig-v5/${fov}/BPTTActorCriticConv_T${T}.json
    done
done
