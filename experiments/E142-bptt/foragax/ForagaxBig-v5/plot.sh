#!/bin/bash
# Figures for the E142 T-BPTT window curve on ForagaxBig-v5 (the paper's
# unending-forager). Assumes process_data_job.sh has produced
# results/E142-bptt/foragax/ForagaxBig-v5/data.parquet.
#
# METRIC: ewm_reward_5 (EWM over rewards with alpha 1e-5, ~100k-step horizon),
# matching E136's ForagaxBig-v5 figures so bars are readable side by side. NOT
# ewm_reward, which is alpha 1e-3 -- a 100x shorter horizon, and identical to
# the ewm_reward_3 column (see src/utils/metrics.py). The learning agents are
# nearly flat in alpha but Search-Oracle is not (0.280 at 1e-3 vs 0.218 at
# 1e-5), so never mix the two within one figure.
#
# --bar-tick-step 0.1 pins the bar axis to E136's tick step; the default
# MaxNLocator lands on 0.04/0.08/0.12 here, which reads incomparably.
#
# Run from the repo root:
#   bash experiments/E142-bptt/foragax/ForagaxBig-v5/plot.sh

set -e

EXP=experiments/E142-bptt/foragax/ForagaxBig-v5
E142_PARQUET=results/E142-bptt/foragax/ForagaxBig-v5/data.parquet
E141_PARQUET=results/E141-unending-forager/foragax/ForagaxBig-v5/data.parquet
CMP=experiments/E142-bptt/foragax/ForagaxBig-v5-vs-oracle
CMP_PARQUET=results/E142-bptt/foragax/ForagaxBig-v5-vs-oracle/data.parquet

PY=.venv/bin/python

# 1. PRIMARY: one reward curve per window, plus the average-reward bars.
#    All five windows share rollout_steps 128, num_mini_batch 8 and epochs 4, so
#    they sit in an identical optimisation regime and only the gradient path
#    differs -- the comparison is internally controlled and needs no baseline.
#    --xlim pins the axis to the full 10M; without it the tick locator fits the
#    last logged sample and a 10M run reads as a 9M one.
$PY src/learning_curve.py "$EXP" \
    --metrics ewm_reward_5 \
    --filter-alg-apertures \
        BPTTActorCriticConv_T1:9 BPTTActorCriticConv_T2:9 \
        BPTTActorCriticConv_T4:9 BPTTActorCriticConv_T8:9 \
        BPTTActorCriticConv_T16:9 \
    --end-frame 10000000 \
    --xlim 0 10000000 \
    --plot-name ForagaxBig-v5_ewm_reward_by_window \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --bar-tick-step 0.1

# 1b. The stacked backbone's five windows, as their own figure. Ten series on
#     one axis is unreadable and the bar panel would interleave the two
#     backbones. Same metric, --xlim and --bar-tick-step as (1) so the two read
#     side by side.
$PY src/learning_curve.py "$EXP" \
    --metrics ewm_reward_5 \
    --filter-alg-apertures \
        BPTTActorCriticConvStacked_T1:9 BPTTActorCriticConvStacked_T2:9 \
        BPTTActorCriticConvStacked_T4:9 BPTTActorCriticConvStacked_T8:9 \
        BPTTActorCriticConvStacked_T16:9 \
    --end-frame 10000000 \
    --xlim 0 10000000 \
    --plot-name ForagaxBig-v5_ewm_reward_by_window_stacked \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --bar-tick-step 0.1

# 2. Performance and learned memory against the window. The right panel is the
#    more sensitive readout: the loss can only supply evidence for retention it
#    can see, so tau = -1/ln(r) should climb with T and then saturate once the
#    window exceeds what the task needs. The ~90-step gap between cue bursts is
#    the horizon to compare it against -- if tau is still rising at T=16, the
#    window is the binding constraint.
$PY src/seq_len_curve.py "$EXP" --arch Conv --metric ewm_reward_5

# 2b. Same two panels for the stacked backbone; writes seq_len_curve_ConvStacked
#     rather than overwriting (2), which is named for its arch.
#
#     No --baseline-path, deliberately. There is no matched real-time stacked
#     CONV agent anywhere in the repo: E140's stacked runs are MLP on v11, and
#     E141's RealTimeActorCriticConvMulti is a different topology (parallel RTUs,
#     not [RTU -> MLP] residual blocks). A mismatched reference line here would
#     be worse than none. Compare this panel against (2) instead -- same
#     environment, same windows, same budget, depth the only difference.
$PY src/seq_len_curve.py "$EXP" --arch ConvStacked --metric ewm_reward_5

# 3. Same curves against the Search-Oracle reference, which is the paper's
#    framing for this environment ("all learning agents were unable to reach the
#    performance level of the Oracle Search baseline").
#
#    The oracle lives in E141's parquet, not E142's, so the two are stitched --
#    same pattern as E140's SquareWave-single-vs-* dirs. Both are 10M runs on
#    this environment, so the comparison is like-for-like.
#
#    Measured from E141's 30 oracle seeds on ewm_reward_5: mean 0.218. For
#    scale, E141's stored RealTimeActorCriticConv on this environment averages
#    0.0003, so it is NOT included as a baseline here. Its Conv-Multi variant
#    reaches 0.071; uncomment below to include it.
#
#    !! E141's single-layer RealTimeActorCriticConv results are INVALID. Do not
#    read that 0.0003 as "the architecture cannot learn this environment".
#    Verified 2026-08-07: E141's own config, run unchanged on current code,
#    reaches a 0.094 reward rate by 300k steps (entropy 1.386 -> 0.90,
#    value_loss healthy, RTU grad norm stable). The stored 10M runs instead show
#    entropy frozen near uniform, value_loss collapsed to 1e-8 and
#    grad_l2_actor_rtu diverging 10 -> 140 across all 30 seeds: an early
#    numerical failure those runs never recovered from. The network and
#    optimizer are byte-identical to E136, and E136's config reproduces the same
#    healthy trajectory, so the cause is in how E141's runs executed, not in the
#    code or the config. Re-run before that agent is used as a baseline.
mkdir -p "$(dirname "$CMP_PARQUET")"
$PY - "$E142_PARQUET" "$E141_PARQUET" "$CMP_PARQUET" <<'PYEOF'
import sys
import polars as pl

e142_path, e141_path, out_path = sys.argv[1:4]
# "BPTTActorCriticConv" without the _T, so BPTTActorCriticConvStacked_T* is
# carried into the stitched parquet too -- the two vs-oracle figures below read
# from this one file and each filters down to its own backbone.
bptt = pl.read_parquet(e142_path).filter(pl.col("alg").str.starts_with("BPTTActorCriticConv"))
e141 = pl.read_parquet(e141_path)
refs = e141.filter(pl.col("alg") == "Search-Oracle")
# refs = e141.filter(pl.col("alg").is_in(["Search-Oracle", "RealTimeActorCriticConvMulti"]))
combined = pl.concat([bptt, refs], how="diagonal_relaxed").sort(
    ["env", "group", "alg", "id", "frame"]
)
combined.write_parquet(out_path)
print(f"Wrote {out_path}")
for alg, sub in combined.group_by("alg"):
    print(f"  {alg[0]}: {sub.height} rows, {sub['seed'].n_unique()} seeds")
PYEOF

#    Search-Oracle is filtered WITHOUT an aperture suffix -- it runs with a
#    privileged full-world view, so its `aperture` is null. Check it gets a bar
#    as well as a curve; where it does not, --legend-on-bar leaves it as an
#    unlabelled line (the defect in E139's own reward figure).
$PY src/learning_curve.py "$CMP" \
    --metrics ewm_reward_5 \
    --filter-alg-apertures \
        Search-Oracle \
        BPTTActorCriticConv_T1:9 BPTTActorCriticConv_T2:9 \
        BPTTActorCriticConv_T4:9 BPTTActorCriticConv_T8:9 \
        BPTTActorCriticConv_T16:9 \
    --end-frame 10000000 \
    --xlim 0 10000000 \
    --plot-name ForagaxBig-v5_ewm_reward_vs_oracle \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --bar-tick-step 0.1

# 3b. Stacked backbone against the same oracle, from the same stitched parquet.
$PY src/learning_curve.py "$CMP" \
    --metrics ewm_reward_5 \
    --filter-alg-apertures \
        Search-Oracle \
        BPTTActorCriticConvStacked_T1:9 BPTTActorCriticConvStacked_T2:9 \
        BPTTActorCriticConvStacked_T4:9 BPTTActorCriticConvStacked_T8:9 \
        BPTTActorCriticConvStacked_T16:9 \
    --end-frame 10000000 \
    --xlim 0 10000000 \
    --plot-name ForagaxBig-v5_ewm_reward_vs_oracle_stacked \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --bar-tick-step 0.1

echo "Done -> $EXP/plots/ and $CMP/plots/"
