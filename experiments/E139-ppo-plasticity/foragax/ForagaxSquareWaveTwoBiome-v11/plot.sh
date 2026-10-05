#!/bin/bash
# Render plasticity figures for E139 RTU-PPO. Assumes process_data.py has
# already produced results/E139-ppo-plasticity/.../data.parquet.

set -e

EXP=experiments/E139-ppo-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11

# Reward switches every 250k steps (square wave: half of the 500k period).
# Use %.0f so macOS/BSD seq emits plain integers (it defaults to %g, which
# renders e.g. 1000000 as "1e+06" and breaks --vertical-lines int parsing).
SWITCHES=$(seq -f "%.0f" 250000 250000 9750000)

# Reward curve (sanity / baseline), RTU vs PPO overlaid. Top-level cross-alg
# comparison; the per-alg plasticity figures go in plots/<alg>/ below.
python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --filter-alg-apertures Search-Oracle RealTimeActorCriticMLP:9 ActorCriticMLP:9 RealTimeActorCriticMLPReLU:9 ActorCriticMLPReLU:9 \
    --end-frame 10000000 \
    --vertical-lines $SWITCHES \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --ylim 2.2

# Per-layer plasticity plots, one algorithm at a time so the figures land in
# separate plots/<agent>/ folders (an agent with no data yet is skipped):
#   - absolute:     full-run actor-vs-critic curves, switches marked by faint
#                   dotted lines. eff_rank is width-normalized.
#   - fold:         switch-triggered average over a mid-training window, centered
#                   so the pre-switch plateau sits next to tau=0.
#   - fold-overlay: the same fold at early/mid/late windows, exposing how the
#                   transient changes over training (plasticity loss).
# Biomes are symmetric, so switches are folded together on the 250k half-period.
for alg in RealTimeActorCriticMLP ActorCriticMLP RealTimeActorCriticMLPReLU ActorCriticMLPReLU; do
    python src/plasticity_compare.py "$EXP" --alg "$alg"
    python src/plasticity_compare.py "$EXP" --alg "$alg" --mode fold --window 4500000:5500000:500
    python src/plasticity_compare.py "$EXP" --alg "$alg" --mode fold-overlay \
        --windows 1000000:2000000:500 4500000:5500000:500 9000000:10000000:500 \
        --window-labels early mid late

    # Per-layer gradient norms (l0/l1/l2), first-rollout-normalized and
    # parameter-count-weighted, read from the parquet, into plots/<alg>/.
    python src/grad_norm_curve.py "$EXP" --alg "$alg" --log-scale
done
