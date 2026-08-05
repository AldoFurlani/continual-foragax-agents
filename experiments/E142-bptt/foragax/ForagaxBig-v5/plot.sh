#!/bin/bash
# Figures for the E142 T-BPTT window curve on ForagaxBig-v5, both architectures.
# Assumes process_data_job.sh has produced results/E142-bptt/.../data.parquet.
#
# No --baseline-path here: E141's real-time agent runs at rollout_steps=128,
# whereas these need 512 to fit a 16-step window, so it is not a matched
# comparison. The T=1 control is the reference instead.

set -e

EXP=experiments/E142-bptt/foragax/ForagaxBig-v5

for arch in Conv MLP; do
    python src/learning_curve.py "$EXP" \
        --metrics ewm_reward \
        --filter-alg-apertures \
            BPTTActorCritic${arch}_T1:9 BPTTActorCritic${arch}_T4:9 \
            BPTTActorCritic${arch}_T8:9 BPTTActorCritic${arch}_T16:9 \
        --end-frame 1000000 \
        --plot-name ewm_reward_${arch}_by_window \
        --legend-on-bar --plot-avg --horizontal-bars

    python src/seq_len_curve.py "$EXP" --arch "$arch"
done
