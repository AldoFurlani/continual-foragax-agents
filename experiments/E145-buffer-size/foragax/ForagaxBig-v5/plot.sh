#!/bin/bash
# Figures and tables for E145 on ForagaxBig-v5, across replay buffer sizes:
# DQN, vanilla and + target offset, at 1K / 10K / 100K; DRQN at 1K / 100K.
#
# Assumes process_data_job.sh has produced
#   results/E145-buffer-size/foragax/ForagaxBig-v5/data.parquet
# and that E144's ForagaxBig-v5 parquet is present: its DQN_LN_HINT /
# DQN_LN_HINT_TARGET / DRQN_LN_HINT runs (buffer_size 1000, otherwise the same
# configs) are the 1K arm, relabelled <agent>_B1K. E144's DRQN did not log
# delta_q, so that metric exists only at 100K.
#
# METRICS, as in E136/E144:
#   ewm_reward_5      EWM over rewards, alpha 1e-5 (~100k-step horizon)
#   ewm_biome_rank_5  rank of the occupied biome by current value, same EWM;
#                     1 = always in the best biome, LOWER is better
#
# Run from the repo root:
#   bash experiments/E145-buffer-size/foragax/ForagaxBig-v5/plot.sh

set -e

E145_PARQUET=results/E145-buffer-size/foragax/ForagaxBig-v5/data.parquet
E144_PARQUET=results/E144-patch-offsets/foragax/ForagaxBig-v5/data.parquet
EXP=experiments/E145-buffer-size/foragax/ForagaxBig-v5
CMP=experiments/E145-buffer-size/foragax/ForagaxBig-v5-with-1K
CMP_PARQUET=results/E145-buffer-size/foragax/ForagaxBig-v5-with-1K/data.parquet

PY=.venv/bin/python

# 0. Stitch E144's runs in as the 1K arm so all sizes share an axis.
mkdir -p "$CMP" "$(dirname "$CMP_PARQUET")"
$PY - "$E145_PARQUET" "$E144_PARQUET" "$CMP_PARQUET" <<'PYEOF'
import sys
import polars as pl

e145_path, e144_path, out_path = sys.argv[1:4]
e145 = pl.read_parquet(e145_path)
one_k = (
    pl.scan_parquet(e144_path)
    .filter(pl.col("alg").is_in(["DQN_LN_HINT", "DQN_LN_HINT_TARGET", "DRQN_LN_HINT"]))
    .with_columns(pl.col("alg") + "_B1K")
    .collect()
)
combined = pl.concat([e145, one_k], how="diagonal_relaxed").sort(
    ["env", "group", "alg", "id", "frame"]
)
combined.write_parquet(out_path)
print(f"Wrote {out_path}")
for (alg, aperture), sub in combined.group_by(["alg", "aperture"]):
    print(f"  {alg}:{aperture}: {sub['seed'].n_unique()} seeds, last frame {sub['frame'].max()}")
PYEOF

series() {  # series <agent> <sizes...> -> the agent at each buffer size
    local alg=$1
    shift
    for size in "$@"; do echo -n "${alg}_B${size}:9 "; done
}

plot() {  # plot <plot-name> <metric> <series...>
    local name=$1 metric=$2
    shift 2
    $PY src/learning_curve.py "$CMP" \
        --metrics "$metric" \
        --filter-alg-apertures "$@" \
        --end-frame 10000000 \
        --xlim 0 10000000 \
        --plot-name "$name" \
        --legend-on-bar \
        --plot-avg \
        --horizontal-bars
}

# One figure per agent, its buffer sizes adjacent.
for alg in DQN_LN_HINT DQN_LN_HINT_TARGET; do
    plot ForagaxBig-v5_${alg}_reward ewm_reward_5 $(series $alg 1K 10K 100K)
    plot ForagaxBig-v5_${alg}_biome_rank ewm_biome_rank_5 $(series $alg 1K 10K 100K)
done
plot ForagaxBig-v5_DRQN_LN_HINT_reward ewm_reward_5 $(series DRQN_LN_HINT 1K 100K)
plot ForagaxBig-v5_DRQN_LN_HINT_biome_rank ewm_biome_rank_5 $(series DRQN_LN_HINT 1K 100K)

# Last-1M reward / rank, collapsed seeds, selected hypers and DRQN delta_q.
$PY $EXP/summary.py "$CMP_PARQUET"

echo "Done -> $CMP/plots/"
