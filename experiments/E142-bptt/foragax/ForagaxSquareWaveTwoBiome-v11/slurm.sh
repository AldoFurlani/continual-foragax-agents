#!/bin/bash
# Submit the T-BPTT window curve at 30 seeds x 10M steps on
# ForagaxSquareWaveTwoBiome-v11.
#
# Run AFTER ../../foragax-sweep/.../process_hypers.sh, which writes each
# window's own selected (alpha, lr_scale, entropy_coef) into these configs.
# Until then they carry E139's real-time winner as a placeholder, which runs but
# is not the tuned setting for any of these windows.
#
# All four windows are run, not just the best one. The result is the shape of
# performance-vs-T, not a single winning window: a monotone rise that plateaus
# is a mechanism claim (the effect tracks the credit-assignment horizon),
# whereas one window beating the baseline is a single number beating another.
# Three points above the T=1 control is the minimum needed to tell "rises then
# plateaus" from "rises linearly".
#
# The matched baseline is E139's RealTimeActorCriticMLP on this same
# environment at 10M -- same architecture, same budget, RTRL instead of a
# truncated window. It needs no new runs; plot.sh reads it via --baseline-path.
#
# 10M steps at rollout_steps=2048 is ~4900 updates. Measured per-update cost
# rises with the window (T=1 614ms, T=16 761ms, T=32 883ms on CPU), so budget
# T=32 at roughly 1.5x the T=1 walltime.
#
# --tasks 20, raised from the 5 inherited from E139 (whose limit came from the
# RTRL sensitivity carry that T-BPTT does not have -- see the sweep script for
# that derivation).
#
# THE EVAL IS THE BINDING CONSTRAINT, NOT THE SWEEP, and a width that passes at
# 1M does not automatically pass here. The training loop is a lax.scan over
# num_updates that stacks per-env-step arrays (rewards, pos, biome_id,
# object_collected_id, biome_regret, biome_rank), so that buffer is sized by
# total_steps: ~28MB per run at 1M, ~280MB at 10M. It dominates everything else
# -- the 32MB rollout carry and 16MB of params/Adam are noise beside it.
#
# Modelled per-run cost here is ~332MB. The same model accounts for only ~2.5GB
# of E139's measured ~8.4GB/run, so it undercounts ~3.3x; corrected, ~1.1GB/run
# puts the ceiling near 40 and 20 inside it with margin. ESTIMATED, NOT
# MEASURED. XLA allocates the scan output up front, so if a width is too high it
# fails on the first executed step with RESOURCE_EXHAUSTED in
# $SCRATCH/job_output_<jobid>.txt rather than hours in.
#
# Sweep and eval need not share a width: the sweep's logging buffer is 10x
# smaller, so it can be packed considerably wider than this.

for fov in 9; do
    for T in 1 8 16 32; do
        python scripts/slurm.py \
            --cluster clusters/vulcan-gpu-vmap-32G.json \
            --tasks 20 --time 06:00:00 --runs 30 --force \
            --entry src/rtu_ppo.py \
            -e experiments/E142-bptt/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/BPTTActorCriticMLP_T${T}.json
    done
done

# The stacked backbone over the same four windows. Two curves rather than one
# is the point: the single-layer curve says what the window buys, and the pair
# says whether depth and window are additive or whether depth substitutes for
# the window. E140's RealTimeActorCriticMLPStacked is the matched real-time
# reference here, exactly as E139's RealTimeActorCriticMLP is for the single
# layer -- same architecture, same budget, RTRL instead of a truncated window.
#
# --tasks 20 as for the single layer (see its note above for the sizing and the
# estimate's caveats); only --time is raised, to 12:00:00.
#
# Depth is not what makes this tight: it adds ~13MB per run (params + Adam
# moments go from ~3.8MB to ~17MB), and the window adds nothing at all --
# create_seq_minibatches makes transitions per minibatch = rollout_steps /
# num_mini_batch = 64 regardless of seq_len. The ~280MB/run lax.scan logging
# buffer is what sets the ceiling, and it is identical for both backbones.
#
# Walltime is the constraint. Two RTU cells per branch instead of one roughly
# doubles the per-update cost, on top of the window's own scaling (measured
# single-layer: T=1 614ms, T=32 883ms), and 10M at rollout_steps=2048 is ~4900
# updates. 12h is sized for T=32; shorter windows finish well inside it.
for fov in 9; do
    for T in 1 8 16 32; do
        python scripts/slurm.py \
            --cluster clusters/vulcan-gpu-vmap-32G.json \
            --tasks 20 --time 12:00:00 --runs 30 --force \
            --entry src/rtu_ppo.py \
            -e experiments/E142-bptt/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/BPTTActorCriticMLPStacked_T${T}.json
    done
done

# The pre-norm / two-residual / rtu_proj stack (E140's block topology under
# T-BPTT) over the same four windows. See the sweep script for why this variant
# exists; in short it completes the gradient-scheme x topology 2x2 and is
# parameter-matched to E140's Stacked2 at 711,365.
#
# Its matched real-time reference is E140's RealTimeActorCriticMLPStacked2 --
# and unlike BPTTActorCriticMLPStacked, that comparison is now topology- AND
# parameter-matched, so the only difference is RTRL vs a truncated window.
#
# --tasks 20 and --time 12:00:00 as for the other stacked arm: same depth, same
# two RTU cells per branch, and slightly fewer parameters.
for fov in 9; do
    for T in 1 8 16 32; do
        python scripts/slurm.py \
            --cluster clusters/vulcan-gpu-vmap-32G.json \
            --tasks 20 --time 12:00:00 --runs 30 --force \
            --entry src/rtu_ppo.py \
            -e experiments/E142-bptt/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/BPTTActorCriticMLPStackedPreNorm_T${T}.json
    done
done
