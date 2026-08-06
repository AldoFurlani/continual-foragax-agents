#!/bin/bash
# Eval runs for the T-BPTT RTU-PPO agents on ForagaxBig-v5 (the
# paper's unending-forager), at 30 seeds x 10M steps.
#
# One file per seq_len -- see the v11 sweep for why the window must not be swept
# inside a single config (the selection pipeline collapses each file to one
# winner, which is how DRQN's sequence_length got tuned away to 1 in XN34).
#
# rollout_steps=128 is one cue period: the environment signals the best biome
# for 10 steps every 100, so a 128-step rollout holds about one cue event. It
# also matches E141's tuned value on this environment.
#
# num_mini_batch=8 is held FIXED across windows so every T sees the same
# optimisation regime: 16 transitions per update and 0.25 gradient steps per
# environment step. Windows are {1,2,4,8,16}: 128//T must be divisible by 8,
# which caps T at 16.
#
# The update rate is structurally capped by the window:
#     grad steps / env step  =  epochs * num_mini_batch / rollout  <=  epochs / T
# so at epochs=4 a 16-step window cannot exceed 0.25 -- E141's tuned 1.0 is only
# reachable up to T=4, at ANY rollout. BPTTActorCriticConv_T1_mb32 is the
# control for that: T=1 at num_mini_batch=32, i.e. E141's exact update rate.
# It separates "the window helped" from "fewer updates hurt".
#
# Conv is the primary architecture -- the paper uses a CNN on RGB, and the MLP
# flattens the 9x9x3 aperture and discards spatial structure on a 28x28 world.
# The MLP launches are left commented rather than deleted; uncomment to run the
# architecture control (it doubles the sweep).

for fov in 9; do
    for T in 1 2 4 8 16; do
        python scripts/slurm.py \
            --cluster clusters/vulcan-gpu-vmap-32G.json \
            --tasks 5 --time 12:00:00 --runs 30 --force \
            --entry src/rtu_ppo.py \
            -e experiments/E142-bptt/foragax/ForagaxBig-v5/${fov}/BPTTActorCriticConv_T${T}.json
    done

    # E141-rate control (1.0 grad steps / env step)
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 12:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E142-bptt/foragax/ForagaxBig-v5/${fov}/BPTTActorCriticConv_T1_mb32.json

    # Architecture control -- uncomment to also sweep the MLP.
    # for T in 1 2 4 8 16; do
    #     python scripts/slurm.py \
    #         --cluster clusters/vulcan-gpu-vmap-32G.json \
    #         --tasks 5 --time 12:00:00 --runs 30 --force \
    #         --entry src/rtu_ppo.py \
    #         -e experiments/E142-bptt/foragax/ForagaxBig-v5/${fov}/BPTTActorCriticMLP_T${T}.json
    # done
done
