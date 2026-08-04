#!/bin/bash
# E140: run ONLY the depth-1 Stacked agent (RealTimeActorCriticMLPStacked1 -- a
# single [RTU + MLP] residual block) on ForagaxSquareWaveTwoBiome-v11,
# 30 seeds x 30M steps, d_hidden=512. This is the within-family baseline for the
# depth ablation; the full four-arm run is in slurm.sh.
#
# Hypers come from the 1M sweep in ../../foragax-sweep/ (stacked_1_slurm.sh
# there, then process_data_job.sh, then process_hypers.sh, which writes the
# selected cell into 9/RealTimeActorCriticMLPStacked1.json). Re-run this after
# any re-selection so the 30M results match the config. The sweep stays at 1M --
# selection uses ~3% of the eval run, which is the intended tune-short protocol.
#
# NOTE: compute_plasticity is OFF in the config -- the stacked class is not yet
# in _PROBED_CLASSES, so plasticity metrics would be a silent no-op.
#
# 30M is 3x the 10M walltime (~6h -> ~18h at --tasks 5); --time 24:00:00 leaves
# headroom. Per-step GPU memory is horizon-independent, so --tasks 5 still fits.
# scripts/slurm.py is idempotent -- re-run after timeouts to fill missing seeds
# (rtu_ppo checkpoints on cancel, so a timed-out seed resumes rather than restarts).

EXP=experiments/E140-rtu-stacked/foragax/ForagaxSquareWaveTwoBiome-v11

for fov in 9; do
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 4 --time 24:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e "$EXP/${fov}/RealTimeActorCriticMLPStacked1.json"
done
