#!/bin/bash
# Rebuild the "single vs multi" reward figure after new Multi-RTU data is pulled.
#
# The comparison is a CROSS-EXPERIMENT view:
#   - RealTimeActorCriticMLPMulti  (multi RTU)   -> from E140 (this experiment)
#   - RealTimeActorCriticMLP       (RTU-PPO)     -> frozen from E139
# so the combined parquet is stitched by hand; there is no config dir for it.
#
# Assumes E140's parquet has already been rebuilt from the new npz (process_data_job).
# Run from the repo root:
#   bash experiments/E140-rtu-stacked/foragax/SquareWave-single-vs-multi/rebuild_and_plot.sh

set -e

SVM=experiments/E140-rtu-stacked/foragax/SquareWave-single-vs-multi
E140_PARQUET=results/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11/data.parquet
E139_PARQUET=results/E139-ppo-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/data.parquet
SVM_PARQUET=results/E140-rtu-stacked/foragax/SquareWave-single-vs-multi/data.parquet

PY=.venv/bin/python

# 1. Stitch the combined parquet: fresh Multi rows (E140) + frozen E139 RTU-PPO rows.
$PY - "$E140_PARQUET" "$E139_PARQUET" "$SVM_PARQUET" <<'PYEOF'
import sys
import polars as pl

e140_path, e139_path, out_path = sys.argv[1:4]
multi = pl.read_parquet(e140_path).filter(pl.col("alg") == "RealTimeActorCriticMLPMulti")
regular = pl.read_parquet(e139_path).filter(pl.col("alg") == "RealTimeActorCriticMLP")
combined = pl.concat([multi, regular], how="diagonal_relaxed")
combined = combined.sort(["env", "group", "alg", "id", "frame"])
combined.write_parquet(out_path)
print(f"Wrote {out_path}: {dict(zip(*[list(x) for x in [combined['alg'].unique().to_list(), []]])) or ''}")
for alg, sub in combined.group_by('alg'):
    print(f"  {alg[0]}: {sub.height} rows, {sub['seed'].n_unique()} seeds")
PYEOF

# 2. Render the overlaid reward curve. No per-switch vertical lines -- at 250k
#    spacing they read as a field of dots; the curve's oscillation shows it.
$PY src/learning_curve.py "$SVM" \
    --metrics ewm_reward \
    --filter-alg-apertures RealTimeActorCriticMLP:9 RealTimeActorCriticMLPMulti:9 \
    --end-frame 10000000 \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --ylim 2.2

echo "Done -> $SVM/plots/ForagaxSquareWaveTwoBiome-v11_ewm_reward_curve.pdf"
