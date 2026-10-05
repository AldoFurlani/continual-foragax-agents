#!/bin/bash
# T-BPTT (T=2) vs RTU-PPO on ForagaxBig-v5 (the paper's unending forager) at 10M.
#
# THIS IS THE CONTROLLED COMPARISON. Both sides are the SAME network --
# BPTTActorCriticConv is documented as "identical architecture to
# RealTimeActorCriticConv: same conv stack, same skip connection, same heads" --
# and PPO-RTU_LN_128 resolves to RealTimeActorCriticConv via PPORegistry. The
# only difference is how the recurrent gradient is computed: a 2-step truncated
# window vs the real-time RTRL correction. That is the comparison E142 exists to
# make.
#
# CROSS-EXPERIMENT view, so the parquet is stitched by hand (same pattern as
# SquareWave-bptt-vs-rtu and E140's SquareWave-single-vs-* dirs):
#   - BPTTActorCriticConv_T2  (truncated BPTT, 2-step window)   -> from E142
#   - PPO-RTU_LN_128          (RTU-PPO, real-time RTRL cell)    -> from E136
#   - Search-Oracle           (privileged planner, the ceiling) -> from E136
#
# Both sides are tuned, on their own grids:
#   T-BPTT T=2      : alpha 3e-4, lr_scale 1.0, entropy 0.1, rollout 128, mb 8,  ep 4
#   PPO-RTU_LN_128  : alpha 1e-4, lr_scale 0.1, entropy 0.1, rollout 128, mb 32, ep 4
# Architecture settings match exactly: hidden 64, d_hidden 512, tanh,
# use_layernorm true, conv PConv2DConv2D, aperture 9.
#
# METRIC. E136's own figure (ForagaxBig-v5-learning-curve-ppo.pdf) plots
# ewm_reward_5; the rest of E142 plots ewm_reward. Both columns exist in both
# parquets. Default here is ewm_reward for consistency within E142 -- set
# METRIC=ewm_reward_5 to reproduce E136's figure exactly. On E141's oracle the
# two differ materially (0.280 vs 0.218), so do not mix them within one figure.
#
# RESOLVED 2026-08-07: E141 ran what should be this same agent and got ~0.0002,
# not the ~0.115 here. E141's runs are the broken ones -- its own config, run
# unchanged on current code, learns normally (0.094 reward rate by 300k steps).
# Use E136's number; treat E141's single-layer result as invalid until re-run.
#
# Run from the repo root:
#   bash experiments/E142-bptt/foragax/ForagaxBig-v5-bptt-vs-rtu/rebuild_and_plot.sh

set -e

METRIC=${METRIC:-ewm_reward}

CMP=experiments/E142-bptt/foragax/ForagaxBig-v5-bptt-vs-rtu
E142_PARQUET=results/E142-bptt/foragax/ForagaxBig-v5/data.parquet
E136_PARQUET=results/E136-big/foragax/ForagaxBig-v5/data.parquet
CMP_PARQUET=results/E142-bptt/foragax/ForagaxBig-v5-bptt-vs-rtu/data.parquet

PY=.venv/bin/python

if [ ! -f "$E136_PARQUET" ]; then
    echo "ERROR: $E136_PARQUET not found."
    echo "  Pull it from the cluster with:  bash scripts/sync_results.sh"
    echo "  (if the parquet does not exist there either, E136's"
    echo "   process_data_job.sh has to be run on vulcan first)"
    exit 1
fi

mkdir -p "$(dirname "$CMP_PARQUET")"

# 1. Stitch. diagonal_relaxed because the two experiments carry different metric
#    columns; missing columns become null rather than raising.
#
#    The cadence check is not decoration: learning_curve interpolates onto a
#    shared frame axis, so two runs logged on different cadences will silently
#    produce a misaligned figure. E140's 30M runs are the known example.
$PY - "$E142_PARQUET" "$E136_PARQUET" "$CMP_PARQUET" "$METRIC" <<'PYEOF'
import sys
import numpy as np
import polars as pl

e142_path, e136_path, out_path, metric = sys.argv[1:5]
bptt = pl.read_parquet(e142_path).filter(pl.col("alg") == "BPTTActorCriticConv_T2")
e136 = pl.read_parquet(e136_path)
rtu = e136.filter(pl.col("alg") == "PPO-RTU_LN_128")
oracle = e136.filter(pl.col("alg") == "Search-Oracle")

if rtu.height == 0:
    raise SystemExit("PPO-RTU_LN_128 not in E136 parquet; check the alg name")
if metric not in bptt.columns or metric not in rtu.columns:
    raise SystemExit(f"metric {metric!r} missing from one side")

def cadence(df, label):
    s = df.filter(pl.col("seed") == df["seed"].min())
    f = np.sort(s["frame"].unique().to_numpy())
    d = np.diff(f)
    print(f"  {label:26s} n={len(f):6d}  median_gap={int(np.median(d)):6d}  "
          f"max_gap={int(d.max()):6d}  last={int(f[-1]):,}")
    return len(f), int(np.median(d)), int(f[-1])

print("Cadence check:")
a = cadence(bptt, "BPTTActorCriticConv_T2")
b = cadence(rtu, "PPO-RTU_LN_128")
if a != b:
    print("  ! cadences differ -- the interpolated curve may be misaligned. "
          "Inspect before trusting the figure.")

combined = pl.concat([bptt, rtu, oracle], how="diagonal_relaxed").sort(
    ["env", "group", "alg", "id", "frame"]
)
combined.write_parquet(out_path)
print(f"\nWrote {out_path}")
for alg, sub in combined.group_by("alg"):
    m = sub[metric].mean()
    print(f"  {alg[0]:30s} {sub.height:7d} rows  {sub['seed'].n_unique():3d} seeds  "
          f"{metric}={m:.4f}")
PYEOF

# 2. Render. Search-Oracle is filtered WITHOUT an aperture suffix -- it runs with
#    a privileged full-world view, so its `aperture` is null. --xlim pins the
#    axis to the full 10M; without it the tick locator fits the last logged
#    sample and a 10M run reads as a 9M one.
$PY src/learning_curve.py "$CMP" \
    --metrics "$METRIC" \
    --filter-alg-apertures Search-Oracle BPTTActorCriticConv_T2:9 PPO-RTU_LN_128:9 \
    --end-frame 10000000 \
    --xlim 0 10000000 \
    --plot-name "ForagaxBig-v5_bptt_T2_vs_rtu_${METRIC}" \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars

echo "Done -> $CMP/plots/"
