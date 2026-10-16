#!/bin/bash
# 10M eval, 30 seeds. Run AFTER process_hypers.sh writes the selected configs,
# and only once no eval job is still running: scripts/slurm.py counts results on
# disk, so runs still in flight would be submitted again.
#
# Before launching, check the selected entropy_coef / epsilon in
# foragax-sweep/ForagaxBig-v5/hypers/9/: if no config learned in the 1M sweep,
# selection picks the most random one (E143 got entropy_coef 1.0 everywhere).
#
# Run sizes are E136's / E142's 9x9 evals. A run that hits its time limit saves
# nothing, so if a job times out, raise --time rather than re-running as is.
EVAL=experiments/E144-patch-offsets/foragax/ForagaxBig-v5/9

for alg in PPO_LN_HINT_128 PPO_LN_HINT_OFFSET_128 PPO-RTU_LN_HINT_128 PPO-RTU_LN_HINT_OFFSET_128; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 06:00:00 --runs 30 --force --entry src/rtu_ppo.py -e $EVAL/${alg}.json
done

for alg in BPTTActorCriticConv_T2_HINT BPTTActorCriticConv_T2_HINT_OFFSET; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 20 --time 12:00:00 --runs 30 --force --entry src/rtu_ppo.py -e $EVAL/${alg}.json
done

for alg in DQN_LN_HINT DQN_LN_HINT_OFFSET; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-mps.json --time 03:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/${alg}.json
done

# DRQN: 10x the sweep's steps, which took up to 9h per run, so a 10M run may need
# several 24h jobs. save_every 1M (in the config) checkpoints the full agent
# state; when a job times out, re-run this script -- once no DRQN job is still
# running -- and it resumes each unfinished run from its last checkpoint.
for alg in DRQN_LN_HINT DRQN_LN_HINT_OFFSET; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-mps.json --time 24:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/${alg}.json
done
