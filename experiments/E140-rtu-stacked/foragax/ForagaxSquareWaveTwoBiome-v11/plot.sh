#!/bin/bash
# Render E140 stacked-RTU figures. Assumes process_data.py has already produced
# results/E140-rtu-stacked/.../data.parquet.

set -e

EXP=experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11

# Reward curve at 30M: the depth ablation (L1 vs L2 vs L4) against plain
# RTU-PPO (single non-residual RTU). Primary E140 figure -- "does the residual
# block help, and does depth help".
# Multi is deliberately excluded: it is still a 10M run and would stop a third
# of the way across the x-axis.
#
# No --vertical-lines. The square wave switches every 250k steps, which is 119
# lines across a 30M axis -- dense enough to grey out the curves they were
# meant to annotate. The sawtooth in the data marks every switch anyway. To
# bring them back at a readable density, mark one switch in ten:
#   SWITCHES=$(seq -f "%.0f" 2500000 2500000 27500000)   # %.0f: BSD seq emits
#   ... --vertical-lines $SWITCHES                       # 1e+06 under %g
#
# --xlim pins the axis to the full 30M. Without it the tick locator fits the
# last logged sample -- the collector subsamples every 30k steps, so the run
# ends at ~29.97M and the axis tops out at a 24 tick, making a 30M run look
# like a 29M one.
python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --filter-alg-apertures RealTimeActorCriticMLP:9 RealTimeActorCriticMLPStacked1:9 RealTimeActorCriticMLPStacked2:9 RealTimeActorCriticMLPStacked4:9 \
    --end-frame 30000000 \
    --xlim 0 30000000 \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --ylim 2.2

# Same figure truncated to 10M: the first 10M is where the depth ordering is
# established, and the 30M axis compresses it into the left third.
#
# NOT comparable to E139 or the SquareWave-single-vs-* figures, even though they
# share an x-axis. The collector subsamples ewm_reward at
#   Subsample(max(total_steps // 1000, 1))                    [rtu_ppo.py main()]
# so this experiment's 30M configs log reward 3x more coarsely than a 10M run:
# largest sampling hole 60k steps here vs 20k there. The post-switch dip is a
# sharp transient, so holes swallow it -- 16 of 39 half-cycles show no dip at all
# in these curves versus 5 of 39 in E139's, which reads as a jagged curve with
# shallow dips. Where a dip IS sampled the depth matches (median -0.041 vs
# -0.042): same dynamics, coarser observation. Truncating a 30M run does NOT
# reproduce a native 10M run.
# Within THIS figure the comparison is sound -- all four algs are 30M configs and
# share the cadence. (Multi is excluded for the same reason it always was: it is
# a 10M run, so it would differ in both length and sampling.)
#
# The bar summary is over the plotted window, so its averages are 0-10M here and
# 0-30M above -- they are not interchangeable.
#
# Search-Oracle is drawn as a horizontal reference, not a curve: it is a fixed
# scripted policy (privileged full-world view + the true reward map), so its
# reward is stationary and a line costs no compute. It is also not in E140's
# parquet -- the value is measured from E139's 30 oracle seeds over this same
# 0-10M window: mean 1.619, per-seed p10 1.45 / p90 1.73. Kept on this panel
# only; on the 30M panel above there is no oracle run to justify extending it.
python src/learning_curve.py "$EXP" \
    --metrics ewm_reward \
    --filter-alg-apertures RealTimeActorCriticMLP:9 RealTimeActorCriticMLPStacked1:9 RealTimeActorCriticMLPStacked2:9 RealTimeActorCriticMLPStacked4:9 \
    --end-frame 10000000 \
    --xlim 0 10000000 \
    --horizontal-lines 1.61:Search-Oracle \
    --plot-name ForagaxSquareWaveTwoBiome-v11_ewm_reward_curve_10M \
    --legend-on-bar \
    --plot-avg \
    --horizontal-bars \
    --ylim 2.2

# Learned RTU memory horizon, vanilla RTU-PPO only.
#
# rtu_r_mean / rtu_r_max are the mean and max spectral radius r over every RTU
# diagonal, logged every rollout via experiment.r_stats_freq (2048 = one
# rollout, the finest cadence available since metrics are emitted at update
# boundaries). r sets the per-unit memory horizon, tau = -1/ln(r), so r -> 1 is
# long memory. For this agent the pool is 1024 poles: 512 in actor_rtu + 512 in
# critic_rtu. At init r ~ sqrt(U(0,1)) by construction (initialize_exp_exp_r),
# giving E[r] = 2/3 -- measured r_mean 0.661 (tau 2.4 steps) and r_max 0.99985
# (tau ~6.7k steps). The question is whether training moves either.
#
# The stacked variants are deliberately excluded. rtu_r_stats pools every
# r_param leaf in the tree into ONE vector before reducing, so depth changes the
# pool size (Stacked1 1024, Stacked2 2048, Stacked4 4096 poles) and r_max is an
# extreme order statistic: at init E[max] = 1 - 1/(2N+1), i.e. 0.99951 / 0.99976
# / 0.99988 for those three. A depth ordering on the r_max panel is therefore
# confounded with pool size, and the mean is averaged over both branches and all
# blocks at once. Restricting to the single-RTU agent removes that confound.
# Resolving depth properly needs rtu_r_stats to return per-cell values rather
# than a pooled pair.
python src/learning_curve.py "$EXP" \
    --metrics rtu_r_mean rtu_r_max \
    --filter-alg-apertures RealTimeActorCriticMLP:9 \
    --end-frame 30000000 \
    --xlim 0 30000000 \
    --plot-name ForagaxSquareWaveTwoBiome-v11_rtu_spectral_radius \
    --legend

# Plasticity-vs-depth figures. DISABLED until the stacked class carries per-block
# probes (add it to _PROBED_CLASSES and generalize the metric sites in
# rtu_ppo.py, then set compute_plasticity=true in the configs). Kept here so the
# E140 format matches E139 once probing lands.
# for alg in RealTimeActorCriticMLPStacked1 RealTimeActorCriticMLPStacked2 RealTimeActorCriticMLPStacked4; do
#     python src/plasticity_compare.py "$EXP" --alg "$alg"
#     python src/plasticity_compare.py "$EXP" --alg "$alg" --mode fold --window 4500000:5500000:500
#     python src/plasticity_compare.py "$EXP" --alg "$alg" --mode fold-overlay \
#         --windows 1000000:2000000:500 4500000:5500000:500 9000000:10000000:500 \
#         --window-labels early mid late
#     python src/grad_norm_curve.py "$EXP" --alg "$alg" --log-scale
# done
# python src/plasticity_retention.py "$EXP"
