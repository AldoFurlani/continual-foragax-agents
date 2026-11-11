#!/bin/bash
# Hyperparameter sweep for E145 on ForagaxBig-v5, tuned at 1M steps = 10% of
# the 10M eval. Replay buffer size is the variable (<agent>_B<size>):
#   DQN_LN_HINT, DQN_LN_HINT_TARGET   10K and 100K
#   DRQN_LN_HINT                      100K, logging delta_q
#
# Every config is E144's with only buffer_size changed (plus delta_q_metric on
# DRQN), so E144's own runs (buffer_size 1000, same grids, seeds and code) are
# the 1K arm -- plot.sh stitches them in rather than re-running them.
# buffer_min_size stays at 50 so every size starts learning at the same step.
#
# What the sizes test, against ForagaxBig-v5's drift (each patch's reward holds
# for 1000 steps; a cycle takes 10K-1M steps; eaten-out patches regrow with a
# new reward function):
#   DQN 1K -> 10K     is DQN's late collapse (E144) a narrow-data effect?
#   10K -> 100K       does stale reward data slow tracking (ewm_biome_rank_5)?
#   DRQN 1K -> 100K   does the agent that did not collapse at 1K change, and do
#                     its stored hidden states go stale (delta_q_first/_last;
#                     the buffer fills over the first 100K steps, so its early
#                     trend is the state-age effect)?
# A 1M buffer is left out: a 1M-step sweep could never fill it.
#
# Packing:
# - DQN vmaps runs on one GPU. Vmapped runs each write their buffer at their
#   own index, which flashbax does with an out-of-place scatter costing
#   O(total buffer) per step (~6x slower steps at 100K than 10K on CPU), so
#   B100K packs 16 runs/job.
# - DRQN runs one process per run under MPS, 16 per GPU on vulcan-gpu-mps.json
#   as in E144's DRQN sweep: each gets ~2% of the GPU, room for the ~130 MB
#   buffer (if not, a run fails at start-up, when the buffer is allocated).
#   The 8-per-GPU vulcan-gpu-mps-c8-16G.json is for the 10M evals, where E144
#   hit host OOM on this profile.
#
# scripts/slurm.py is idempotent: re-run to fill seeds lost to OOM or timeout.
SWEEP=experiments/E145-buffer-size/foragax-sweep/ForagaxBig-v5/9

for alg in DQN_LN_HINT DQN_LN_HINT_TARGET; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 256 --time 03:00:00 --runs 10 --force --entry src/continuing_main.py -e $SWEEP/${alg}_B10K.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 16 --time 03:00:00 --runs 10 --force --entry src/continuing_main.py -e $SWEEP/${alg}_B100K.json
done

# DRQN: 24h as in E144 -- its 1M sweep runs took up to 9h, and a sweep run never
# reaches the 1M checkpoint, so one that times out restarts from zero.
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps.json --time 24:00:00 --runs 10 --force --entry src/continuing_main.py -e $SWEEP/DRQN_LN_HINT_B100K.json
