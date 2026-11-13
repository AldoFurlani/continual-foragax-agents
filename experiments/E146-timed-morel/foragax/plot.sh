#!/bin/bash
# Learning curves for E146, the Timed-Morel Forager: one panel per condition
# with all agents and baselines, on the same axes, and a second panel with the
# agents frozen at 5M steps (dashed line). Average reward, exponential moving
# average with decay 0.999.
#
# Assumes process_data_job.sh has produced
#   results/E146-timed-morel/foragax/<condition>/data.parquet
#
# Run from the repo root:
#   bash experiments/E146-timed-morel/foragax/plot.sh
set -e

AGENTS="DQN ActorCriticMLP DQN_reward_trace ActorCriticMLP-reward-trace DRQN RealTimeActorCriticMLP"
BASELINES="Random TimedMorel-Oysters-Only TimedMorel-Clock-Oracle TimedMorel-Oracle"
BASELINE_COLORS="Random:tol:muted:pale_grey TimedMorel-Oysters-Only:tol:vibrant:blue TimedMorel-Clock-Oracle:tol:vibrant:orange TimedMorel-Oracle:tol:vibrant:red"

for cond in ForagaxTimedMorel-v1-no-fog:17 ForagaxTimedMorel-v1-k1:5 ForagaxTimedMorel-v1-k1250:5 ForagaxTimedMorel-v1-fixed:5; do
    dir=${cond%:*}
    fov=${cond#*:}
    for variant in "" _frozen_5M; do
        algs=""
        colors="$BASELINE_COLORS"
        for alg in $AGENTS; do
            algs="$algs $alg$variant:$fov"
        done
        colors="$colors DQN$variant:tol:muted:green ActorCriticMLP$variant:tol:muted:indigo"
        colors="$colors DQN_reward_trace$variant:tol:vibrant:teal ActorCriticMLP-reward-trace$variant:tol:vibrant:cyan"
        colors="$colors DRQN$variant:black RealTimeActorCriticMLP$variant:tol:vibrant:magenta"
        lines=""
        if [ -n "$variant" ]; then
            lines="--vertical-lines 5000000"
        fi
        python src/learning_curve.py experiments/E146-timed-morel/foragax/$dir \
            --plot-name "all$variant" \
            --filter-alg-apertures $BASELINES $algs \
            --colors $colors \
            --disable-fov --end-frame 10000000 --ylim '-0.1' 0.7 $lines
    done
done
