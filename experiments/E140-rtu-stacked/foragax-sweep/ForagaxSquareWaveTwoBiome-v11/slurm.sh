#!/bin/bash
# Hyperparameter sweep for every E140 RTU-PPO variant on
# ForagaxSquareWaveTwoBiome-v11: the Multi (2-RTU, concat-idiom, non-residual)
# agent and the three Stacked ([RTU + MLP] residual block) depths 1/2/4. Tuned at
# the short 1M horizon (tune-short / eval-long: the selected configs run at 10M in
# ../../foragax/). Matches the paper's never-ending-relearning grid (Table 8) so
# every arm is tuned on the same grid/protocol as the Real-Time PPO baseline they
# are compared against -- without this the depth ablation pits a tuned Multi
# against hand-set Stacked configs.
#
# Grid: actor alpha {1e-3, 3e-4, 1e-4} x critic lr_scale {0.1, 1.0, 10} x
# entropy_coef {0.01, 0.1, 1.0} = 3*3*3 = 27 cells, x --runs seeds. beta2 fixed
# at 0.999 (paper default; the E139 beta2 sweep selected 0.999 anyway). n_blocks
# and use_gating stay fixed per config: depth is the ablation axis (one config per
# depth, tuned independently), and gating is a separate architectural ablation.
# seed_offset=1M keeps the selection seeds disjoint from the 30 eval seeds.
#
# --tasks == vmap width per GPU (memory-bound). The 2-RTU Multi (d_hidden=512 +
# LayerNorm) OOMs a single L40S above --tasks ~2 (the 10M eval OOM'd at --tasks 5,
# needing ~44 GiB; per-step GPU footprint is horizon-independent, so the 1M sweep
# is the same). Stacked is lighter per RTU: every block's RTU reads the width-W
# (64) stream, so its (d_input x d_hidden) gradient traces are 64x512 per block
# versus Multi's second RTU at 1024x512. --tasks 5 is the width the 10M stacked
# eval already ran at; bump it if the assigned GPU has headroom.
# scripts/slurm.py is idempotent -- re-run after timeouts to fill missing seeds.

SW=experiments/E140-rtu-stacked/foragax-sweep/ForagaxSquareWaveTwoBiome-v11

for fov in 9; do
    # Multi RTU-PPO (2 RTUs, concat idiom -- non-residual reference)
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 2 --time 02:00:00 --runs 10 --force \
        --entry src/rtu_ppo.py \
        -e "$SW/${fov}/RealTimeActorCriticMLPMulti.json"

    # Stacked RTU-PPO, depth 1 (within-family baseline)
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 02:00:00 --runs 10 --force \
        --entry src/rtu_ppo.py \
        -e "$SW/${fov}/RealTimeActorCriticMLPStacked1.json"

    # Stacked RTU-PPO, depth 2 (RTU count matched to Multi)
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 02:00:00 --runs 10 --force \
        --entry src/rtu_ppo.py \
        -e "$SW/${fov}/RealTimeActorCriticMLPStacked2.json"

    # Stacked RTU-PPO, depth 4 (matches Lu et al. S5-in-PPO depth)
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 03:00:00 --runs 10 --force \
        --entry src/rtu_ppo.py \
        -e "$SW/${fov}/RealTimeActorCriticMLPStacked4.json"
done
