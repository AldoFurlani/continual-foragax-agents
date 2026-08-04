#!/bin/bash
# Rebuild the "single RTU-PPO vs depth-1 stacked" reward figure.
#
# The comparison is a CROSS-EXPERIMENT view:
#   - RealTimeActorCriticMLPStacked1 (one [RTU + MLP] residual block) -> from E140
#   - RealTimeActorCriticMLP         (RTU-PPO, single RTU)            -> frozen from E139
# so the combined parquet is stitched by hand; there is no config dir for it.
# This isolates "does wrapping one RTU in a pre-norm residual block with an MLP
# sublayer help at all", before any depth is added on top.
#
# Sibling of SquareWave-single-vs-multi (single vs 2-RTU concat) and
# SquareWave-single-vs-stacked (single vs DEPTH-2 stacked -- note that older
# directory is depth 2 despite the unnumbered name).
#
# Assumes E140's parquet has already been rebuilt from the new npz (process_data_job).
# Run from the repo root:
#   bash experiments/E140-rtu-stacked/foragax/SquareWave-single-vs-stacked1/rebuild_and_plot.sh

set -e

SVS=experiments/E140-rtu-stacked/foragax/SquareWave-single-vs-stacked1
E140_PARQUET=results/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11/data.parquet
E139_PARQUET=results/E139-ppo-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/data.parquet
SVS_PARQUET=results/E140-rtu-stacked/foragax/SquareWave-single-vs-stacked1/data.parquet

PY=.venv/bin/python

mkdir -p "$(dirname "$SVS_PARQUET")"

# 1. Stitch the combined parquet: Stacked1 rows (E140) + frozen E139 RTU-PPO rows.
#    diagonal_relaxed because the two experiments carry different metric columns
#    (E140 is wider); missing columns become null rather than an error.
$PY - "$E140_PARQUET" "$E139_PARQUET" "$SVS_PARQUET" <<'PYEOF'
import sys
import polars as pl

e140_path, e139_path, out_path = sys.argv[1:4]
stacked = pl.read_parquet(e140_path).filter(
    pl.col("alg") == "RealTimeActorCriticMLPStacked1"
)
regular = pl.read_parquet(e139_path).filter(pl.col("alg") == "RealTimeActorCriticMLP")
combined = pl.concat([stacked, regular], how="diagonal_relaxed")
combined = combined.sort(["env", "group", "alg", "id", "frame"])
combined.write_parquet(out_path)
print(f"Wrote {out_path}")
for alg, sub in combined.group_by("alg"):
    print(f"  {alg[0]}: {sub.height} rows, {sub['seed'].n_unique()} seeds")
PYEOF

# 2. Render the overlaid reward curve. No per-switch vertical lines -- at 250k
#    spacing they read as a field of dots; the curve's oscillation shows it.
$PY src/learning_curve.py "$SVS" \
    --metrics ewm_reward \
    --filter-alg-apertures RealTimeActorCriticMLP:9 RealTimeActorCriticMLPStacked1:9 \
    --end-frame 10000000 \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --ylim 2.2

echo "Done -> $SVS/plots/ForagaxSquareWaveTwoBiome-v11_ewm_reward_curve.pdf"
