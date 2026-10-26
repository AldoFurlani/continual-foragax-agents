#!/bin/bash
# Hyperparameter sweep for E145 on ForagaxBig-v5, tuned at 1M steps = 10% of
# the 10M eval. Replay buffer size is the variable: DQN and DRQN, each vanilla
# and with E144's target offset, at buffer_size 1K / 10K / 100K / 1M
# (<agent>_B<size>). Configs are E144's apart from:
#   buffer_size     the variable
#   epsilon         grid extended to 0.5 -- E144's DRQN picked 0.25, the top of
#                   its grid, at every variant
#   delta_q_metric  DRQN only: logs R2D2's Q-value discrepancy, delta_q_first /
#                   delta_q_last (src/algorithms/nn/DRQN.py)
# buffer_min_size stays at 50 so every size starts learning at the same step.
#
# What the sizes test, against ForagaxBig-v5's drift (each patch's reward holds
# for 1000 steps; a cycle takes 10K-1M steps; eaten-out patches regrow with a
# new reward function):
#   1K -> 10K      is DQN's late collapse a narrow-data effect?
#   10K -> 1M      does stale reward data slow tracking (ewm_biome_rank_5)?
#   selected eps   does a larger buffer remove the need for eps 0.25+?
#   DRQN delta_q   do stored hidden states go stale as the buffer grows?
# Caveat: a 1M sweep never fills a 1M buffer, so selection at B1M (and mostly
# at B100K) sees a buffer that holds the whole run so far.
#
# Memory: ~1.0 KB/transition for DQN, ~1.3 KB for DRQN (the stored GRU state),
# so a B1M run holds ~1-1.3 GB of buffer on the GPU.
# - DQN vmaps runs on one GPU; --tasks keeps a job's buffers <= ~16 GB.
# - DRQN runs one process per run under MPS. vulcan-gpu-mps.json gives each
#   ~2% of the GPU, too little for B1M, so B100K/B1M use the 8-per-node,
#   10%-of-GPU vulcan-gpu-mps-c8-16G.json.
#
# scripts/slurm.py is idempotent: re-run to fill seeds lost to OOM or timeout.
SWEEP=experiments/E145-buffer-size/foragax-sweep/ForagaxBig-v5/9

dqn_tasks() {
    case $1 in
        1K | 10K) echo 256 ;;
        100K) echo 64 ;;
        1M) echo 16 ;;
    esac
}

for size in 1K 10K 100K 1M; do
    for alg in DQN_LN_HINT DQN_LN_HINT_TARGET; do
        python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks $(dqn_tasks $size) --time 03:00:00 --runs 10 --force --entry src/continuing_main.py -e $SWEEP/${alg}_B${size}.json
    done
done

# DRQN: 24h as in E144 -- its 1M sweep runs took up to 9h, and a sweep run never
# reaches the 1M checkpoint, so one that times out restarts from zero.
for size in 1K 10K; do
    for alg in DRQN_LN_HINT DRQN_LN_HINT_TARGET; do
        python scripts/slurm.py --cluster clusters/vulcan-gpu-mps.json --time 24:00:00 --runs 10 --force --entry src/continuing_main.py -e $SWEEP/${alg}_B${size}.json
    done
done
for size in 100K 1M; do
    for alg in DRQN_LN_HINT DRQN_LN_HINT_TARGET; do
        python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-c8-16G.json --time 24:00:00 --runs 10 --force --entry src/continuing_main.py -e $SWEEP/${alg}_B${size}.json
    done
done
