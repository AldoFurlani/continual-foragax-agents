#!/bin/bash
# Submit RTU-PPO and the vanilla PPO baseline at 30 seeds × 10M steps on
# ForagaxSquareWaveTwoBiome-v11.
# --tasks 5 vmaps 5 seeds/GPU -> 6 jobs (d_hidden=512 + LayerNorm + the
# plasticity/grad-norm probes OOM a single L40S at --tasks 10/30).
# scripts/slurm.py is idempotent — re-run after timeouts to fill missing seeds,
# and an agent whose seeds are all done schedules nothing.

EXP=experiments/R4-exp_rtus/foragax/ForagaxSquareWaveTwoBiome-v11

for fov in 9; do
    # PPO
    python scripts/slurm.py \
       --cluster clusters/vulcan-gpu-vmap-32G.json \
       --tasks 5 --time 06:00:00 --runs 30 --force \
       --entry src/rtu_ppo.py \
       -e ${EXP}/${fov}/ActorCriticMLP.json
done
