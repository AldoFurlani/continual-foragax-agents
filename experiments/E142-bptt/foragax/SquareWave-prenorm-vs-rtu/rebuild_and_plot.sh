#!/bin/bash
# Stacked-PreNorm T-BPTT (all four windows) vs vanilla RTU-PPO on
# ForagaxSquareWaveTwoBiome-v11 at 10M.
#
# CROSS-EXPERIMENT view, so the parquet is stitched by hand (same pattern as
# SquareWave-bptt-vs-rtu and E140's SquareWave-single-vs-* dirs):
#   - BPTTActorCriticMLPStackedPreNorm_T{1,8,16,32}  -> from E142
#   - RealTimeActorCriticMLP  (RTU-PPO, real-time RTRL cell, single layer)
#                                                    -> from E139
#
# WHY THIS PAIRING. The stacked-PreNorm agent is E140's block topology (two
# pre-norm residuals per block + rtu_proj) trained with a truncated window
# instead of RTRL. RTU-PPO is the flat single-layer real-time agent. So the
# comparison asks the question the depth line of work exists to answer: does a
# two-block stack, trained with a gradient that actually crosses timesteps, beat
# the flat real-time agent it was meant to improve on?
#
# The answer at 10M is no -- every window sits below RTU-PPO's 1.487:
#
#   RTU-PPO                        1.487   0/30 dead
#   Stacked-PreNorm T=1            1.011   0/30
#   Stacked-PreNorm T=8            0.535  18/30      <- see below
#   Stacked-PreNorm T=16           1.225   0/30
#   Stacked-PreNorm T=32           1.351   0/30
#
# What the pre-norm topology DID fix is the collapse. The post-norm LRU-style
# stack (BPTTActorCriticMLPStacked, plotted in ../ForagaxSquareWaveTwoBiome-v11/
# plot.sh) loses 29/28/27/20 seeds of 30 across the same windows; pre-norm loses
# 0/18/0/0. That is the single largest effect in E142 and it is a topology
# effect, not a gradient-scheme one -- the two agents share seq_len, depth,
# hyperparameter grid and (at 711,365 vs 1,431,493 params) differ only in the
# residual/norm arrangement.
#
# T=8 is the one window that still collapses (18/30), and it is UNEXPLAINED.
# The obvious suspect -- a bad hyperparameter draw, since selection runs at 1M
# where the collapse has not yet appeared -- is ruled out: all four windows
# selected the SAME configuration (entropy_coef 0.1, alpha 3e-4, lr_scale 10.0),
# so T=8 differs from its neighbours only in seq_len. Nor is there a
# discontinuity in the minibatch shape to appeal to; seq_batch runs 64/8/4/2
# across T=1/8/16/32 with T=8 in no way special.
#
# So this is either genuine instability at that window or seed noise, and 18/30
# against 0/30 is too large to wave through as the latter. Do not present the
# pre-norm result as "collapse fixed" without addressing it -- re-running T=8 on
# a fresh seed block is the cheap first check.
#
# Sampling is directly comparable: both sides are 10M configs, so ewm_reward is
# logged on the same cadence. This is NOT true of E140's 30M runs -- do not mix
# those in without truncating first.
#
# Run from the repo root:
#   bash experiments/E142-bptt/foragax/SquareWave-prenorm-vs-rtu/rebuild_and_plot.sh

set -e

CMP=experiments/E142-bptt/foragax/SquareWave-prenorm-vs-rtu
E142_PARQUET=results/E142-bptt/foragax/ForagaxSquareWaveTwoBiome-v11/data.parquet
E139_PARQUET=results/E139-ppo-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/data.parquet
CMP_PARQUET=results/E142-bptt/foragax/SquareWave-prenorm-vs-rtu/data.parquet

PY=.venv/bin/python
mkdir -p "$(dirname "$CMP_PARQUET")"

# 1. Stitch. diagonal_relaxed because the two experiments carry different metric
#    columns (E139 has the plasticity probes, E142 does not); missing columns
#    become null rather than raising.
$PY - "$E142_PARQUET" "$E139_PARQUET" "$CMP_PARQUET" <<'PYEOF'
import sys
import polars as pl

e142_path, e139_path, out_path = sys.argv[1:4]
prenorm = pl.read_parquet(e142_path).filter(
    pl.col("alg").str.starts_with("BPTTActorCriticMLPStackedPreNorm_T")
)
rtu = pl.read_parquet(e139_path).filter(pl.col("alg") == "RealTimeActorCriticMLP")
combined = pl.concat([prenorm, rtu], how="diagonal_relaxed").sort(
    ["env", "group", "alg", "id", "frame"]
)
combined.write_parquet(out_path)
print(f"Wrote {out_path}")
for alg, sub in combined.group_by("alg"):
    print(f"  {alg[0]}: {sub.height} rows, {sub['seed'].n_unique()} seeds")
PYEOF

# 2. Render. No per-switch vertical lines: at 250k spacing they are 40 dots
#    across the axis and the curves' own oscillation already shows the switches.
#    --ylim 2.2 matches the other v11 figures so bars read side by side.
$PY src/learning_curve.py "$CMP" \
    --metrics ewm_reward \
    --filter-alg-apertures \
        RealTimeActorCriticMLP:9 \
        BPTTActorCriticMLPStackedPreNorm_T1:9 BPTTActorCriticMLPStackedPreNorm_T8:9 \
        BPTTActorCriticMLPStackedPreNorm_T16:9 BPTTActorCriticMLPStackedPreNorm_T32:9 \
    --end-frame 10000000 \
    --xlim 0 10000000 \
    --plot-name ForagaxSquareWaveTwoBiome-v11_prenorm_vs_rtu \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --ylim 2.2

echo "Done -> $CMP/plots/"
