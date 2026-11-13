#!/bin/bash
# 10M eval for E146, the Timed-Morel Forager, 30 new seeds (seed_offset 0; the
# sweep used 1,000,000). Run AFTER foragax-sweep/process_hypers.sh has written
# the selected configs and their _frozen_5M copies (learning stops at 5M steps)
# into ./<condition>/<fov>/, and only once no eval job is still running:
# scripts/slurm.py counts results on disk, so runs still in flight would be
# submitted again.
#
# The environment is the local copy in src/environments/TimedMorel.py.
#
# Packing follows earlier evals of these agents; none of it has been calibrated
# on this environment:
# - DQN: 15 vmapped runs per GPU (E145). With no save_every a run's 10M steps
#   are one chunk, so every step's logged data lands in host memory at the end.
# - DRQN: one process per run on CPU nodes, 16 per job, no GPU, 12h. A 10M run
#   took the equivalent of ~1.7h locally at FOV 5; set the time from the
#   sweep's measured walltime (x10) before submitting. With no save_every a
#   run that times out restarts from zero.
# - PPO with 2048-step rollouts: 5 runs per job, 6h (R1).
# - Baselines are scripted and jitted: 15 vmapped runs per GPU.
EVAL=experiments/E146-timed-morel/foragax

for cond in ForagaxTimedMorel-v1-no-fog/17 ForagaxTimedMorel-v1-k1/5 ForagaxTimedMorel-v1-k1250/5 ForagaxTimedMorel-v1-fixed/5; do
    for alg in DQN DQN_reward_trace; do
        for variant in "" _frozen_5M; do
            python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 15 --time 12:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/$cond/$alg$variant.json
        done
    done

    for variant in "" _frozen_5M; do
        python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 12:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/$cond/DRQN$variant.json
    done

    for alg in ActorCriticMLP ActorCriticMLP-reward-trace RealTimeActorCriticMLP; do
        for variant in "" _frozen_5M; do
            python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 06:00:00 --runs 30 --force --entry src/rtu_ppo.py -e $EVAL/$cond/$alg$variant.json
        done
    done
done

# Baselines do not depend on the sweep and can be submitted at any time.
for cond in ForagaxTimedMorel-v1-no-fog ForagaxTimedMorel-v1-k1 ForagaxTimedMorel-v1-k1250 ForagaxTimedMorel-v1-fixed; do
    for alg in Random TimedMorel-Oysters-Only TimedMorel-Clock-Oracle TimedMorel-Oracle; do
        python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 15 --time 03:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/$cond/Baselines/$alg.json
    done
done
