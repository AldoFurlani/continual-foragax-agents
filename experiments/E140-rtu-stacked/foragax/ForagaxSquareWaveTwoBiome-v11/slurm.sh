#!/bin/bash
# E140: stacked RTU-PPO (pre-norm [RTU + MLP] residual blocks) at depths
# 1/2/4 on ForagaxSquareWaveTwoBiome-v11, 30 seeds x 30M steps, plus the
# Multi (2-RTU concat-idiom) non-residual baseline and the Search-Oracle
# reward reference. The three Stacked depths are the depth ablation (same code
# path, only n_blocks varies; L1 is the within-family baseline); Multi vs
# Stacked-2 isolates "more recurrence" from "the residual redesign" (all at
# d_hidden=512). scripts/slurm.py is idempotent -- re-run to fill missing seeds.
#
# To run ONLY the Multi baseline, use slurm_multi.sh instead.
#
# Hypers for all four arms come from the 1M sweep in ../../foragax-sweep/ (run its
# slurm.sh, then process_data_job.sh, then process_hypers.sh, which writes the
# selected cell into the 9/*.json configs here). Re-run this script after any
# re-selection so the 30M results match the configs. The sweep stays at the 1M
# tune-short horizon -- selection uses ~3% of the eval run, well inside the
# early window (do NOT re-tune at 30M; late plasticity is the measurement).
#
# HORIZON: Stacked 1/2/4, plain RTU-PPO and Search-Oracle all run to 30M.
# Multi is deliberately left at 10M and is excluded from the 30M figure
# (see plot.sh) -- run slurm_multi.sh separately if you want it back.
#
# r_stats_freq=2048 in the configs logs the mean/max RTU spectral radius r every
# rollout (tau = -1/ln(r) is the learned memory horizon in steps). Depth-agnostic:
# it reduces over every RTU diagonal in both branches.
#
# NOTE: compute_plasticity is OFF in the configs -- the stacked class is not yet
# in _PROBED_CLASSES, so plasticity metrics would be a silent no-op. This run is
# reward-curves only. Flip compute_plasticity to true once the per-block probes
# are wired (then the plasticity block in plot.sh applies).
#
# 30M is 3x the 10M walltime (~6h -> ~18h); --time 12:00:00 leaves headroom.
#
# --tasks is scaled inversely with depth: the stored RTRL traces are ~1 GiB per
# RTU per vmapped run (4 tensors of 2048 x 64 x 512 f32, stacked over the
# rollout by lax.scan), and an agent has 2*n_blocks RTUs. --tasks 5 OOM'd
# Stacked4 on a 40 GiB card, asking for 82 GiB. Memory is horizon-independent,
# so the same widths hold at 1M and 30M.
#
# If the partition caps below 24h, rely on rtu_ppo's Checkpoint -- re-running
# this script is idempotent and resumes unfinished seeds.

for fov in 9; do
    # Stacked RTU-PPO, depth 1 (within-family baseline)
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 4 --time 12:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLPStacked1.json

    # Stacked RTU-PPO, depth 2
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 2 --time 12:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLPStacked2.json

    # Stacked RTU-PPO, depth 4 (matches Lu et al. S5-in-PPO depth)
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 1 --time 12:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLPStacked4.json

    # Plain RTU-PPO (single non-residual RTU) -- the within-experiment
    # reference for "does the residual block help at all". Hypers are E139's
    # swept cell (entropy 0.1, actor alpha 3e-4, critic lr_scale 10.0);
    # compute_plasticity off, so this is a fresh reward-only 30M run rather
    # than the E139 10M one.
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 4 --time 12:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLP.json

    # Multi RTU-PPO (2 stacked RTUs, concat idiom -- non-residual baseline).
    # Deliberately still 10M and excluded from the 30M figure.
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLPMulti.json
done

# Search-Oracle is NOT run here. It is a fixed policy, so its reward is
# stationary (mean ewm_reward 1.61 over 30 seeds, p10 1.37 / p90 1.87) and a
# 30M -- or even a repeat 10M -- run adds nothing. plot.sh draws it as a
# horizontal reference line instead, which also spans the full 30M x-axis
# rather than stopping a third of the way across.
#
# If you ever do re-run it, note that raising total_steps above
# experiment.save_every (default 10_001_000) crashes: the periodic checkpoint
# pickles the glue state and JAX's typed PRNGKeyArray is not picklable. Set
# save_every above total_steps first.
