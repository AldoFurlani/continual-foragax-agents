#!/bin/bash
# Hyperparameter sweep for the Multi (2-RTU, concat-idiom, non-residual) agent on
# ForagaxSquareWaveTwoBiome-v11, tuned at the short 1M horizon (tune-short /
# eval-long: the selected config runs at 10M in ../../foragax/). Matches the
# paper's never-ending-relearning grid (Table 8) so Multi is tuned on the same
# grid/protocol as the Real-Time PPO baseline it is compared against.
#
# Grid: actor alpha {1e-3, 3e-4, 1e-4} x critic lr_scale {0.1, 1.0, 10} x
# entropy_coef {0.01, 0.1, 1.0} = 3*3*3 = 27 cells, x --runs seeds. beta2 fixed
# at 0.999 (paper default; the E139 beta2 sweep selected 0.999 anyway).
# seed_offset=1M keeps the selection seeds disjoint from the 30 eval seeds.
#
# --tasks 2 vmaps 2 runs/GPU: the 2-RTU Multi (d_hidden=512 + LayerNorm) OOMs a
# single L40S above --tasks ~2 (the 10M eval OOM'd at --tasks 5, needing ~44 GiB;
# per-step GPU footprint is horizon-independent, so the 1M sweep is the same).
# Bump to --tasks 3 only if the assigned GPU has >=40 GB. scripts/slurm.py is
# idempotent -- re-run after timeouts to fill missing seeds.

for fov in 9; do
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 2 --time 02:00:00 --runs 10 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E140-rtu-stacked/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLPMulti.json
done
