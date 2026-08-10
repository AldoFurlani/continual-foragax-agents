#!/bin/bash
# Figures for the E142 T-BPTT window curve on ForagaxSquareWaveTwoBiome-v11.
# Assumes process_data_job.sh has produced results/E142-bptt/.../data.parquet.

set -e

EXP=experiments/E142-bptt/foragax/ForagaxSquareWaveTwoBiome-v11
BASE=experiments/E139-ppo-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11
# Matched real-time reference for the STACKED backbone. E140's Stacked2 is the
# right counterpart -- n_blocks=2, d_hidden=512, hidden=64, rollout_steps=2048,
# identical to the E142 stacked configs, with RTRL in place of the window.
E140=experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11

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

# 1b. The stacked backbone's four windows, as their own figure. Eight series on
#     one axis is unreadable and the bar panel would interleave the two
#     backbones; --plot-name keeps this from overwriting (1), which takes the
#     default name. Same --ylim so the two figures can be read side by side.
python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --filter-alg-apertures \
        BPTTActorCriticMLPStacked_T1:9 BPTTActorCriticMLPStacked_T8:9 \
        BPTTActorCriticMLPStacked_T16:9 BPTTActorCriticMLPStacked_T32:9 \
    --end-frame 10000000 \
    --vertical-lines $SWITCHES \
    --plot-name ForagaxSquareWaveTwoBiome-v11_ewm_reward_curve_stacked \
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

# 2b. The same two panels for the stacked backbone, against ITS matched
#     real-time agent. Separate figure, not a second series on (2): --arch takes
#     one architecture, and the two curves have different reference lines --
#     E139's single-layer RealTimeActorCriticMLP for (2), E140's Stacked2 here.
#     seq_len_curve names its output after the arch, so this writes
#     seq_len_curve_MLPStacked.png alongside (2)'s seq_len_curve_MLP.png.
#
#     Read the pair together. If depth substitutes for the window, the stacked
#     curve is flatter in T; if they are additive, the two rise in parallel with
#     the stacked one offset upward.
#
#     --baseline-end-frame 10000000 is REQUIRED here and must not be dropped.
#     E140's runs are 30M while these are 10M, and --tail-fraction is relative
#     to each run's own last frame, so without it the reference line would be
#     E140 averaged over 22.5M-30M against T-BPTT averaged over 7.5M-10M -- a
#     3x horizon advantage read as an architecture result. E139's baseline in
#     (2) is already 10M, which is why it needs no such flag.
python src/seq_len_curve.py "$EXP" \
    --arch MLPStacked \
    --baseline-path "$E140" \
    --baseline-alg RealTimeActorCriticMLPStacked2 \
    --baseline-end-frame 10000000

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

# 3b. Same, stacked backbone. rtu_r_stats aggregates r over EVERY RTU diagonal
#     in the tree, so with n_blocks=2 this is one number averaged across both
#     blocks and both branches -- a per-block breakdown would need a separate
#     metric. Enough to answer the question (3) is for: whether r has settled by
#     the end of the run or is still drifting.
python src/learning_curve.py "$EXP" \
    --metrics rtu_r_mean \
    --filter-alg-apertures \
        BPTTActorCriticMLPStacked_T1:9 BPTTActorCriticMLPStacked_T8:9 \
        BPTTActorCriticMLPStacked_T16:9 BPTTActorCriticMLPStacked_T32:9 \
    --end-frame 10000000 \
    --plot-name rtu_r_mean_by_window_stacked \
    --legend
