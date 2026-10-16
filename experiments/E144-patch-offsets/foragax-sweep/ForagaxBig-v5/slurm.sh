#!/bin/bash
# Hyperparameter sweep for E144 on ForagaxBig-v5, tuned at 1M steps = 10% of
# the 10M eval. Every agent keeps its usual 9x9 rgb aperture and gets the
# constant hint (hint_duration 100); each runs twice:
#
#   <agent>         vanilla: the hint as in E136
#   <agent>_OFFSET  + patch_offsets: the agent-relative (dx, dy) to each of the
#                   four food patches, appended to the hint vector
#                   (src/environments/ForagaxObsExtras.py)
#
# The offsets remove the localisation half of the partial observability -- the
# agent no longer has to work out where hint slot k is from where it stands --
# while keeping the agent-centred view that makes local foraging learnable. E143
# showed that a fixed full-map view does the opposite: no agent learned.
#
# Sweep grids are the source experiments' own: E136 for PPO, RTU-PPO, DQN and
# DRQN (DRQN_LN_1_1: sequence_length 64, burn-in 16, kept fixed -- sweeping it
# let XN34's selection tune DRQN's memory away to 1),
# E142 for T-BPTT (T=2, num_mini_batch 8). Within a pair only patch_offsets
# differs. Run sizes are what those experiments used at 9x9.
#
# scripts/slurm.py is idempotent: re-run to fill seeds lost to OOM or timeout.
SWEEP=experiments/E144-patch-offsets/foragax-sweep/ForagaxBig-v5/9

for alg in PPO_LN_HINT_128 PPO_LN_HINT_OFFSET_128; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 256 --time 03:00:00 --runs 10 --force --entry src/rtu_ppo.py -e $SWEEP/${alg}.json
done

for alg in PPO-RTU_LN_HINT_128 PPO-RTU_LN_HINT_OFFSET_128; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 03:00:00 --runs 10 --force --entry src/rtu_ppo.py -e $SWEEP/${alg}.json
done

for alg in BPTTActorCriticConv_T2_HINT BPTTActorCriticConv_T2_HINT_OFFSET; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 20 --time 03:00:00 --runs 10 --force --entry src/rtu_ppo.py -e $SWEEP/${alg}.json
done

# DQN runs through continuing_main.py; it reads the hint (and so the offsets)
# through representation.scalar_features.
for alg in DQN_LN_HINT DQN_LN_HINT_OFFSET; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 256 --time 03:00:00 --runs 10 --force --entry src/continuing_main.py -e $SWEEP/${alg}.json
done

# DRQN (GRU) is the recurrent counterpart of DQN, as RTU-PPO / T-BPTT are of PPO.
# Packed one run per core on the MPS node, as in E136, whose DRQN sweep used
# 9h jobs. save_every 1M checkpoints the full agent state, so a run that hits
# the time limit resumes when this script is re-run.
for alg in DRQN_LN_HINT DRQN_LN_HINT_OFFSET; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-mps.json --time 09:00:00 --runs 10 --force --entry src/continuing_main.py -e $SWEEP/${alg}.json
done
