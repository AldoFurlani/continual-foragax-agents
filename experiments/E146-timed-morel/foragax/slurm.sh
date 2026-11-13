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
# Packing, from what the sweep measured (the eval itself is not yet run):
# - DQN: 15 vmapped runs per GPU (E145). With no save_every a run's 10M steps
#   are one chunk, so every step's logged data lands in host memory at the end.
# - DRQN: one process per run on CPU nodes, 16 per job, no GPU, 24h. The 1M
#   sweep jobs took 56-72 min at FOV 17, so a 10M run there needs ~10-12h.
#   With no save_every a run that times out restarts from zero.
# - PPO: 15 vmapped runs per GPU, 6h. The sweep ran 54 per GPU in ~5 min per
#   1M steps; 15 keeps the logged data of a 10M run within host memory, as for
#   DQN.
# - RTU-PPO: 6 per GPU, as in the sweep: it needs ~4.4 GiB of GPU memory per
#   run and 18 per GPU died of RESOURCE_EXHAUSTED.
# - Baselines are scripted and jitted: 15 vmapped runs per GPU.
EVAL=experiments/E146-timed-morel/foragax

for cond in ForagaxTimedMorel-v1-no-fog/17 ForagaxTimedMorel-v1-k1/5 ForagaxTimedMorel-v1-k1250/5 ForagaxTimedMorel-v1-fixed/5; do
    for alg in DQN DQN_reward_trace; do
        for variant in "" _frozen_5M; do
            python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 15 --time 12:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/$cond/$alg$variant.json
        done
    done

    for variant in "" _frozen_5M; do
        python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 24:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/$cond/DRQN$variant.json
    done

    for alg in ActorCriticMLP ActorCriticMLP-reward-trace; do
        for variant in "" _frozen_5M; do
            python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 15 --time 06:00:00 --runs 30 --force --entry src/rtu_ppo.py -e $EVAL/$cond/$alg$variant.json
        done
    done

    for variant in "" _frozen_5M; do
        python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 6 --time 06:00:00 --runs 30 --force --entry src/rtu_ppo.py -e $EVAL/$cond/RealTimeActorCriticMLP$variant.json
    done
done

# Baselines do not depend on the sweep and can be submitted at any time.
for cond in ForagaxTimedMorel-v1-no-fog ForagaxTimedMorel-v1-k1 ForagaxTimedMorel-v1-k1250 ForagaxTimedMorel-v1-fixed; do
    for alg in Random TimedMorel-Oysters-Only TimedMorel-Clock-Oracle TimedMorel-Oracle; do
        python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 15 --time 03:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/$cond/Baselines/$alg.json
    done
done
