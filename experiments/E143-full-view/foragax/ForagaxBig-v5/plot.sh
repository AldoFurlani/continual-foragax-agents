#!/bin/bash
# Figures for E143: ForagaxBig-v5 with partial observability mitigated -- the
# whole 28x28 map (rgb + agent-position channel) and a constant or absent hint.
#
# Assumes process_data_job.sh has produced
#   results/E143-full-view/foragax/ForagaxBig-v5/data.parquet
# and that E136 / E142's ForagaxBig-v5 parquets are present (they hold the 9x9
# controls and the Search-Oracle ceiling).
#
# All figures read one stitched parquet so that full-view and 9x9 runs can share
# an axis. E143 and E136 use the same agent names (PPO_LN_HINT_128, ...), so the
# E143 rows are given aperture 28 -- the full map -- and every series is
# filtered as alg:aperture; legends then read "(FOV 28)" vs "(FOV 9)".
#
# METRICS, as in E136/E142:
#   ewm_reward_5      EWM over rewards, alpha 1e-5 (~100k-step horizon)
#   ewm_biome_rank_5  rank of the occupied biome by current value, same EWM;
#                     1 = always in the best biome, LOWER is better. This is the
#                     direct readout of "does the agent go where the hint says".
#
# Run from the repo root:
#   bash experiments/E143-full-view/foragax/ForagaxBig-v5/plot.sh

set -e

E143_PARQUET=results/E143-full-view/foragax/ForagaxBig-v5/data.parquet
E136_PARQUET=results/E136-big/foragax/ForagaxBig-v5/data.parquet
E142_PARQUET=results/E142-bptt/foragax/ForagaxBig-v5/data.parquet
CMP=experiments/E143-full-view/foragax/ForagaxBig-v5-compare
CMP_PARQUET=results/E143-full-view/foragax/ForagaxBig-v5-compare/data.parquet

PY=.venv/bin/python

# 0. Stitch E143 with its references:
#    E136  PPO_LN_HINT_128:9, PPO-RTU_LN_HINT_128:9  9x9 rgb, constant hint --
#          the matched controls, so view size is the only difference
#    E136  Search-Oracle                             hand-coded, full map + true
#          rewards: the ceiling
#    E142  BPTTActorCriticConv_T2:9                  9x9 rgb, INTERMITTENT hint --
#          the nearest T-BPTT control; no 9x9 constant-hint T-BPTT run exists
mkdir -p "$CMP" "$(dirname "$CMP_PARQUET")"
$PY - "$E143_PARQUET" "$E136_PARQUET" "$E142_PARQUET" "$CMP_PARQUET" <<'PYEOF'
import sys
import polars as pl

e143_path, e136_path, e142_path, out_path = sys.argv[1:5]
e143 = pl.read_parquet(e143_path).with_columns(pl.lit(28, dtype=pl.Int64).alias("aperture"))
e136 = (
    pl.scan_parquet(e136_path)
    .filter(pl.col("alg").is_in(["PPO_LN_HINT_128", "PPO-RTU_LN_HINT_128", "Search-Oracle"]))
    .collect()
)
e142 = pl.scan_parquet(e142_path).filter(pl.col("alg") == "BPTTActorCriticConv_T2").collect()
combined = pl.concat([e143, e136, e142], how="diagonal_relaxed").sort(
    ["env", "group", "alg", "id", "frame"]
)
combined.write_parquet(out_path)
print(f"Wrote {out_path}")
for (alg, aperture), sub in combined.group_by(["alg", "aperture"]):
    print(f"  {alg}:{aperture}: {sub['seed'].n_unique()} seeds, last frame {sub['frame'].max()}")
PYEOF

FULL_VIEW="Search-Oracle \
    PPO_LN_HINT_128:28 PPO_LN_NOHINT_128:28 \
    PPO-RTU_LN_HINT_128:28 PPO-RTU_LN_NOHINT_128:28 \
    BPTTActorCriticConv_T2_HINT:28 BPTTActorCriticConv_T2_NOHINT:28"

# Pairs adjacent so each agent's 9x9 and full-view bars sit together.
VIEW_EFFECT="Search-Oracle \
    PPO_LN_HINT_128:9 PPO_LN_HINT_128:28 \
    PPO-RTU_LN_HINT_128:9 PPO-RTU_LN_HINT_128:28 \
    BPTTActorCriticConv_T2:9 BPTTActorCriticConv_T2_HINT:28"

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

# 1. Full view: hint vs no hint for each agent. With a constant hint there is
#    almost nothing left to remember, so an RTU / T-BPTT gain over feedforward
#    PPO here is not memory (E139: RTUs keep plasticity vanilla PPO loses).
#    Without the hint, memory of what each patch tasted like is what helps.
plot ForagaxBig-v5_full_view_reward ewm_reward_5 $FULL_VIEW
plot ForagaxBig-v5_full_view_biome_rank ewm_biome_rank_5 $FULL_VIEW

# 2. View size with the hint held constant: each agent at 9x9 vs the full map.
#    If the full-view agents reach a rank near 1 and the 9x9 ones do not,
#    localisation from a 9x9 view was the bottleneck in E136. The T-BPTT pair is
#    not matched on the hint (intermittent at 9x9), so read it with that caveat.
plot ForagaxBig-v5_view_effect_reward ewm_reward_5 $VIEW_EFFECT
plot ForagaxBig-v5_view_effect_biome_rank ewm_biome_rank_5 $VIEW_EFFECT

echo "Done -> $CMP/plots/"
