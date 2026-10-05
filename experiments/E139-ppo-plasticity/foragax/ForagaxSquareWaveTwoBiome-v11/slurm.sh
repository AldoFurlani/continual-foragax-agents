#!/bin/bash
# Submit RTU-PPO and the vanilla PPO baseline at 30 seeds × 10M steps on
# ForagaxSquareWaveTwoBiome-v11.
# --tasks 5 vmaps 5 seeds/GPU -> 6 jobs (d_hidden=512 + LayerNorm + the
# plasticity/grad-norm probes OOM a single L40S at --tasks 10/30).
# scripts/slurm.py is idempotent — re-run after timeouts to fill missing seeds,
# and an agent whose seeds are all done schedules nothing.

for fov in 9; do
    # RTU-PPO.
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E139-ppo-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLP.json

    # Vanilla PPO baseline
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E139-ppo-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLP.json

    # RTU-PPO with ReLU activations at every layer
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E139-ppo-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/RealTimeActorCriticMLPReLU.json

    # Vanilla PPO with ReLU everywhere (incl. the wide mid layer)
    python scripts/slurm.py \
        --cluster clusters/vulcan-gpu-vmap-32G.json \
        --tasks 5 --time 06:00:00 --runs 30 --force \
        --entry src/rtu_ppo.py \
        -e experiments/E139-ppo-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/${fov}/ActorCriticMLPReLU.json
done

# Search-Oracle reference line: scripted greedy planner with privileged full-world
# observation (aperture -1) and the true per-cell reward map (reward_prioritization).
# It adapts to each square-wave switch for free, so it is the performance ceiling,
# not a learner. Runs on CPU via continuing_main.py (no network, no plasticity probes),
# aperture-independent so it sits outside the fov loop.
python scripts/slurm.py \
    --cluster clusters/vulcan-cpu.json \
    --time 01:00:00 --runs 30 --force \
    --entry src/continuing_main.py \
    -e experiments/E139-ppo-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11/Baselines/Search-Oracle.json
