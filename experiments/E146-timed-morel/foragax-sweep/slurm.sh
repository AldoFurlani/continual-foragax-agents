#!/bin/bash
# Hyperparameter sweep for E146, the Timed-Morel Forager (ForagaxTimedMorel-v1),
# tuned at 1M steps = 10% of the 10M eval, 10 seeds.
#
# Four conditions, one directory each (<condition>/<fov>/<agent>.json):
#   ForagaxTimedMorel-v1-no-fog   no fog, FOV 17; bands move every 1,250 appearances
#   ForagaxTimedMorel-v1-k1       fog, FOV 5; bands move at every appearance
#   ForagaxTimedMorel-v1-k1250    fog, FOV 5; bands move every 1,250 appearances
#   ForagaxTimedMorel-v1-fixed    fog, FOV 5; bands never move
# Each condition is tuned on its own.
#
# Six agents. Every config is XFinal's ForagaxSquareWaveTwoBiome-v11 sweep config
# (the paper's) with only the environment block changed:
#   DQN, DQN_reward_trace, DRQN                      src/continuing_main.py
#   ActorCriticMLP (PPO), ActorCriticMLP-reward-trace,
#   RealTimeActorCriticMLP (RTU-PPO)                 src/rtu_ppo.py
# The reward-trace agents sweep the decay over {0.9, 0.99}. DRQN samples
# sequences of 32 and burns in the first 16, so it learns over 16 steps.
#
# The environment is the local copy in src/environments/TimedMorel.py.
#
# Packing follows the experiments these configs come from; none of it has been
# calibrated on this environment:
# - DQN vmaps all runs of a file on one GPU (E144/E145).
# - PPO with 2048-step rollouts packs 5 runs per job (R1).
# - DRQN runs one process per run on CPU nodes, 16 per job, no GPU. The net is
#   a small MLP: a local run took ~6 s per 10K steps at FOV 5 (~10 min per 1M),
#   and vmapping 16 runs was at least 5x slower per run. FOV 17 and the
#   cluster's single-threaded cores are not timed, hence 4h: a 1M sweep run
#   never reaches a checkpoint, so one that times out restarts from zero.
#
# scripts/slurm.py is idempotent: re-run to fill seeds lost to OOM or timeout.
SWEEP=experiments/E146-timed-morel/foragax-sweep

for cond in ForagaxTimedMorel-v1-no-fog/17 ForagaxTimedMorel-v1-k1/5 ForagaxTimedMorel-v1-k1250/5 ForagaxTimedMorel-v1-fixed/5; do
    for alg in DQN DQN_reward_trace; do
        python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 256 --time 03:00:00 --runs 10 --force --entry src/continuing_main.py -e $SWEEP/$cond/$alg.json
    done

    python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 04:00:00 --runs 10 --force --entry src/continuing_main.py -e $SWEEP/$cond/DRQN.json

    for alg in ActorCriticMLP ActorCriticMLP-reward-trace RealTimeActorCriticMLP; do
        python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 5 --time 03:00:00 --runs 10 --force --entry src/rtu_ppo.py -e $SWEEP/$cond/$alg.json
    done
done
