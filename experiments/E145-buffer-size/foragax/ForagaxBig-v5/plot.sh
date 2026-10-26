#!/bin/bash
# Figures and tables for E145: DQN / DRQN, vanilla and + target offset, across
# replay buffer sizes 1K / 10K / 100K / 1M on ForagaxBig-v5.
#
# Assumes process_data_job.sh has produced
#   results/E145-buffer-size/foragax/ForagaxBig-v5/data.parquet
#
# METRICS, as in E136/E144:
#   ewm_reward_5      EWM over rewards, alpha 1e-5 (~100k-step horizon)
#   ewm_biome_rank_5  rank of the occupied biome by current value, same EWM;
#                     1 = always in the best biome, LOWER is better
#
# Run from the repo root:
#   bash experiments/E145-buffer-size/foragax/ForagaxBig-v5/plot.sh

set -e

EXP=experiments/E145-buffer-size/foragax/ForagaxBig-v5
PY=.venv/bin/python

series() {  # series <agent> -> the agent at every buffer size
    for size in 1K 10K 100K 1M; do echo -n "$1_B${size}:9 "; done
}

plot() {  # plot <plot-name> <metric> <series...>
    local name=$1 metric=$2
    shift 2
    $PY src/learning_curve.py "$EXP" \
        --metrics "$metric" \
        --filter-alg-apertures "$@" \
        --end-frame 10000000 \
        --xlim 0 10000000 \
        --plot-name "$name" \
        --legend-on-bar \
        --plot-avg \
        --horizontal-bars
}

# One figure per agent, its four buffer sizes adjacent.
for alg in DQN_LN_HINT DQN_LN_HINT_TARGET DRQN_LN_HINT DRQN_LN_HINT_TARGET; do
    plot ForagaxBig-v5_${alg}_reward ewm_reward_5 $(series $alg)
    plot ForagaxBig-v5_${alg}_biome_rank ewm_biome_rank_5 $(series $alg)
done

# Last-1M reward / rank, collapsed seeds, selected hypers and DRQN delta_q.
$PY $EXP/summary.py results/$EXP/data.parquet

echo "Done -> $EXP/plots/"
