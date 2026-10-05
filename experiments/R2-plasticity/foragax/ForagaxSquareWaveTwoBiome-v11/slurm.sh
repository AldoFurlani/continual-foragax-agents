#!/bin/bash
# Submit RTU-PPO and the vanilla PPO baseline at 30 seeds × 10M steps on
# ForagaxSquareWaveTwoBiome-v11.
# --tasks 5 vmaps 5 seeds/GPU -> 6 jobs (d_hidden=512 + LayerNorm + the
# plasticity/grad-norm probes OOM a single L40S at --tasks 10/30).
# scripts/slurm.py is idempotent — re-run after timeouts to fill missing seeds,
# and an agent whose seeds are all done schedules nothing.

EXP=experiments/R2-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11

for fov in 9; do
    # Vanilla PPO with ReLU everywhere (incl. the wide mid layer) -- the LOP baseline.
    python scripts/slurm.py \
       --cluster clusters/vulcan-gpu-vmap-32G.json \
       --tasks 5 --time 06:00:00 --runs 30 --force \
       --entry src/rtu_ppo.py \
       -e ${EXP}/${fov}/ActorCriticMLPReLU.json

    python scripts/slurm.py \
       --cluster clusters/vulcan-gpu-vmap-32G.json \
       --tasks 5 --time 06:00:00 --runs 30 --force \
       --entry src/rtu_ppo.py \
       -e ${EXP}/${fov}/ActorCriticMLPReLU_20m.json

    # PPO with tanh from original paper
    python scripts/slurm.py \
       --cluster clusters/vulcan-gpu-vmap-32G.json \
       --tasks 5 --time 06:00:00 --runs 30 --force \
       --entry src/rtu_ppo.py \
       -e ${EXP}/${fov}/ActorCriticMLP_tanh.json

    # RTU-PPO (ReLU) -- the agent that should resist LOP; the comparison arm.
    python scripts/slurm.py \
       --cluster clusters/vulcan-gpu-vmap-32G.json \
       --tasks 5 --time 06:00:00 --runs 30 --force \
       --entry src/rtu_ppo.py \
       -e ${EXP}/${fov}/RealTimeActorCriticMLPReLU.json


    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e ${EXP}/${fov}/ActorCriticMLP-l2-init_relu.json

    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e ${EXP}/${fov}/ActorCriticMLP_tanh_2.json

    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e ${EXP}/${fov}/RealTimeActorCriticMLP-l2-init.json

    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e ${EXP}/${fov}/RealTimeActorCriticMLP-l2-init_relu.json

    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 2 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e ${EXP}/${fov}/RealTimeActorCriticMLP_crelu.json


done
