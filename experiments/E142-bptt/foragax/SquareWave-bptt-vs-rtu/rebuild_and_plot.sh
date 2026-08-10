#!/bin/bash
# T-BPTT (T=16) vs vanilla RTU-PPO on ForagaxSquareWaveTwoBiome-v11 at 10M.
#
# CROSS-EXPERIMENT view, so the parquet is stitched by hand (same pattern as
# E140's SquareWave-single-vs-* dirs; there is no config dir for it):
#   - BPTTActorCriticMLP_T16   (truncated BPTT, 16-step window)  -> from E142
#   - RealTimeActorCriticMLP   (RTU-PPO, real-time RTRL cell)    -> from E139
#
# Same architecture on both sides -- identical dense stack, skip connection,
# heads and RTU cell. The ONLY difference is how the recurrent gradient is
# computed: a 16-step truncated window vs the real-time RTRL correction. That is
# what makes this the comparison the experiment exists to make.
#
# Both sides are tuned, on their own grids:
#   T-BPTT T=16 : alpha 1e-3,  lr_scale 0.1,  entropy_coef 0.01
#   RTU-PPO     : alpha 3e-4,  lr_scale 10.0, entropy_coef 0.1
#
# Sampling is directly comparable here: both are 10M configs, so ewm_reward is
# logged on the same cadence (12051 frames, 20k largest hole). This is NOT true
# of E140's 30M runs -- see the note in E140's plot.sh before mixing those in.
#
# Run from the repo root:
#   bash experiments/E142-bptt/foragax/SquareWave-bptt-vs-rtu/rebuild_and_plot.sh

set -e

CMP=experiments/E142-bptt/foragax/SquareWave-bptt-vs-rtu
E142_PARQUET=results/E142-bptt/foragax/ForagaxSquareWaveTwoBiome-v11/data.parquet
E139_PARQUET=results/E139-ppo-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/data.parquet
CMP_PARQUET=results/E142-bptt/foragax/SquareWave-bptt-vs-rtu/data.parquet

PY=.venv/bin/python
mkdir -p "$(dirname "$CMP_PARQUET")"

# 1. Stitch. diagonal_relaxed because the two experiments carry different metric
#    columns (E139 has the plasticity probes, E142 does not); missing columns
#    become null rather than raising.
$PY - "$E142_PARQUET" "$E139_PARQUET" "$CMP_PARQUET" <<'PYEOF'
import sys
import polars as pl

e142_path, e139_path, out_path = sys.argv[1:4]
bptt = pl.read_parquet(e142_path).filter(pl.col("alg") == "BPTTActorCriticMLP_T16")
e139 = pl.read_parquet(e139_path)
rtu = e139.filter(pl.col("alg") == "RealTimeActorCriticMLP")
# Search-Oracle: scripted greedy planner with a privileged full-world view and
# the true reward map. The performance ceiling, and the only series here that
# does not have to LEARN the switch.
oracle = e139.filter(pl.col("alg") == "Search-Oracle")
combined = pl.concat([bptt, rtu, oracle], how="diagonal_relaxed").sort(
    ["env", "group", "alg", "id", "frame"]
)
combined.write_parquet(out_path)
print(f"Wrote {out_path}")
for alg, sub in combined.group_by("alg"):
    print(f"  {alg[0]}: {sub.height} rows, {sub['seed'].n_unique()} seeds")
PYEOF

# 2. Render. No per-switch vertical lines: at 250k spacing they are 40 dots
#    across the axis and the curve's own oscillation already shows the switches.
#
#    Search-Oracle is filtered WITHOUT an aperture suffix -- it runs with a
#    privileged full-world view, so its `aperture` is null. It does get its own
#    bar here, so --legend-on-bar labels all three series; check that after any
#    change to the filter list, since E139's five-series figure drops the oracle
#    from its bar panel and leaves it as an unlabelled curve.
$PY src/learning_curve.py "$CMP" \
    --metrics ewm_reward \
    --filter-alg-apertures Search-Oracle BPTTActorCriticMLP_T16:9 RealTimeActorCriticMLP:9 \
    --end-frame 10000000 \
    --xlim 0 10000000 \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --ylim 2.2

echo "Done -> $CMP/plots/"
