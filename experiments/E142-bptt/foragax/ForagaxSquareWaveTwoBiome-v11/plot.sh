#!/bin/bash
# Figures for the E142 T-BPTT window curve on ForagaxSquareWaveTwoBiome-v11.
# Assumes process_data_job.sh has produced results/E142-bptt/.../data.parquet.

set -e

EXP=experiments/E142-bptt/foragax/ForagaxSquareWaveTwoBiome-v11
BASE=experiments/E139-ppo-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11

# Reward switches every 250k steps (square wave: half of the 500k period).
# %.0f so BSD seq emits plain integers rather than 1e+06.
SWITCHES=$(seq -f "%.0f" 250000 250000 9750000)

# 1. Learning curves, all four windows overlaid. T=1 is the control: same
#    architecture and data flow with the recurrent gradient path removed.
python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --filter-alg-apertures \
        BPTTActorCriticMLP_T1:9 BPTTActorCriticMLP_T8:9 \
        BPTTActorCriticMLP_T16:9 BPTTActorCriticMLP_T32:9 \
    --end-frame 10000000 \
    --vertical-lines $SWITCHES \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --ylim 2.2

# 2. The headline figure: performance and learned memory against the window,
#    with E139's real-time (RTRL) agent as the reference line. The shape is the
#    result -- a rise that plateaus says the effect tracks the credit horizon;
#    a flat curve says the deferred-credit gradient was noise.
#
#    The right panel is the more sensitive instrument. The loss can only supply
#    evidence for retention it can see, so tau = -1/ln(r) should climb with T
#    and saturate once T exceeds what the task needs. Where it flattens
#    estimates the task's actual memory requirement -- expected around the void
#    crossing / meal spacing scale of roughly 10-20 steps.
python src/seq_len_curve.py "$EXP" \
    --arch MLP \
    --baseline-path "$BASE" \
    --baseline-alg RealTimeActorCriticMLP

# 3. RTU pole magnitude over training, per window. Confirms the summary in (2)
#    is not an artifact of the late-window average -- e.g. r still drifting at
#    the end of the run means the run was too short, not that memory saturated.
python src/learning_curve.py "$EXP" \
    --metrics rtu_r_mean \
    --filter-alg-apertures \
        BPTTActorCriticMLP_T1:9 BPTTActorCriticMLP_T8:9 \
        BPTTActorCriticMLP_T16:9 BPTTActorCriticMLP_T32:9 \
    --end-frame 10000000 \
    --plot-name rtu_r_mean_by_window \
    --legend
