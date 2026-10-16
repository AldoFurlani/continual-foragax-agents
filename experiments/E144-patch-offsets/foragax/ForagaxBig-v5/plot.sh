#!/bin/bash
# Figures for E144: each agent on ForagaxBig-v5 with its 9x9 rgb view and the
# constant hint, with and without the agent-relative offsets to the four food
# patches. The gap within a pair is what localisation costs that agent.
#
# Assumes process_data_job.sh has produced
#   results/E144-patch-offsets/foragax/ForagaxBig-v5/data.parquet
# and that E136's ForagaxBig-v5 parquet is present (Search-Oracle, the ceiling).
#
# METRICS, as in E136/E142/E143:
#   ewm_reward_5      EWM over rewards, alpha 1e-5 (~100k-step horizon)
#   ewm_biome_rank_5  rank of the occupied biome by current value, same EWM;
#                     1 = always in the best biome, LOWER is better
#
# Run from the repo root:
#   bash experiments/E144-patch-offsets/foragax/ForagaxBig-v5/plot.sh

set -e

E144_PARQUET=results/E144-patch-offsets/foragax/ForagaxBig-v5/data.parquet
E136_PARQUET=results/E136-big/foragax/ForagaxBig-v5/data.parquet
CMP=experiments/E144-patch-offsets/foragax/ForagaxBig-v5-vs-oracle
CMP_PARQUET=results/E144-patch-offsets/foragax/ForagaxBig-v5-vs-oracle/data.parquet

PY=.venv/bin/python

# 0. Stitch E144 with E136's Search-Oracle so they can share an axis.
mkdir -p "$CMP" "$(dirname "$CMP_PARQUET")"
$PY - "$E144_PARQUET" "$E136_PARQUET" "$CMP_PARQUET" <<'PYEOF'
import sys
import polars as pl

e144_path, e136_path, out_path = sys.argv[1:4]
e144 = pl.read_parquet(e144_path)
oracle = pl.scan_parquet(e136_path).filter(pl.col("alg") == "Search-Oracle").collect()
combined = pl.concat([e144, oracle], how="diagonal_relaxed").sort(
    ["env", "group", "alg", "id", "frame"]
)
combined.write_parquet(out_path)
print(f"Wrote {out_path}")
for (alg, aperture), sub in combined.group_by(["alg", "aperture"]):
    print(f"  {alg}:{aperture}: {sub['seed'].n_unique()} seeds, last frame {sub['frame'].max()}")
PYEOF

# Pairs adjacent so each agent's vanilla and offset bars sit together.
SERIES="Search-Oracle \
    PPO_LN_HINT_128:9 PPO_LN_HINT_OFFSET_128:9 \
    PPO-RTU_LN_HINT_128:9 PPO-RTU_LN_HINT_OFFSET_128:9 \
    BPTTActorCriticConv_T2_HINT:9 BPTTActorCriticConv_T2_HINT_OFFSET:9 \
    DQN_LN_HINT:9 DQN_LN_HINT_OFFSET:9 \
    DRQN_LN_HINT:9 DRQN_LN_HINT_OFFSET:9"

plot() {  # plot <plot-name> <metric>
    $PY src/learning_curve.py "$CMP" \
        --metrics "$2" \
        --filter-alg-apertures $SERIES \
        --end-frame 10000000 \
        --xlim 0 10000000 \
        --plot-name "$1" \
        --legend-on-bar \
        --plot-avg \
        --horizontal-bars
}

plot ForagaxBig-v5_offsets_reward ewm_reward_5
plot ForagaxBig-v5_offsets_biome_rank ewm_biome_rank_5

echo "Done -> $CMP/plots/"
