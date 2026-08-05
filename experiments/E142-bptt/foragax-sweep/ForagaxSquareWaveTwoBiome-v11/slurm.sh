#!/bin/bash
# Hyperparameter sweep for the T-BPTT RTU-PPO agents on
# ForagaxSquareWaveTwoBiome-v11, tuned at the short 1M horizon (tune-short /
# eval-long convention; the selected configs run at 10M in ../../foragax/).
#
# ONE FILE PER seq_len, deliberately. seq_len is the treatment variable, not a
# nuisance hyperparameter: the selection pipeline collapses each config file to
# a single winning setting, so a shared file sweeping seq_len would discard
# every window but one. That is exactly what happened to DRQN's sequence_length
# in XN34 (hypers/9/DRQN.json selected sequence_length=1). Separate files also
# keep the agent name -- and therefore the results path -- distinct per T, which
# is what makes the performance-vs-T curve plottable.
#
# Grid per file: actor alpha x critic lr_scale x entropy_coef = 3*3*3 = 27
# cells, x --runs seeds. 4 windows x 27 = 108 permutations total, the same
# budget as a combined sweep -- it just yields four tuned agents instead of one.
#
# Learning rates are selected WITHIN each seq_len. Longer windows accumulate
# gradient over more steps and use more correlated minibatches, so the optimum
# genuinely moves with T; a globally-selected alpha would confound "this window
# is worse" with "this window ran at another window's learning rate".
#
# T=1 is the control: identical architecture and data flow to the other windows
# with the recurrent gradient path removed, so it isolates what the window buys
# over the real-time RTRL cell in ../../../E139-ppo-plasticity/.
#
# --tasks 5 mirrors E139: d_hidden=512 + LayerNorm has the same per-run
# footprint, which OOMs a single L40S above 5 vmapped runs. T-BPTT adds
# O(seq_len) stored activations per sequence but drops the RTRL sensitivity
# carry (which was O(d_input * d_hidden) per step), so peak memory is comparable
# -- drop --tasks for T=32 if you see OOM. scripts/slurm.py is idempotent:
# re-run after timeouts to fill missing seeds.

for fov in 9; do
    for T in 1 8 16 32; do
        python scripts/slurm.py \
            --cluster clusters/vulcan-gpu-vmap-32G.json \
            --tasks 5 --time 03:00:00 --runs 10 --force \
            --entry src/rtu_ppo.py \
            -e experiments/E142-bptt/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/BPTTActorCriticMLP_T${T}.json
    done
done
