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
# --tasks 5 mirrors E139, and is CONSERVATIVE here rather than tight. E139's
# limit came from the RTRL sensitivity carry: four (batch, d_input, d_hidden)
# trace tensors per RTU, 1128KB per step, so 2.2GB over a 2048-step rollout and
# ~11GB at 5 vmapped runs -- that is what OOMs a 48GB L40S at 6. T-BPTT drops
# that carry entirely and stores only (h_c1, h_c2): 8KB per step, 16MB per
# rollout, ~140x less.
#
# The window adds nothing either. create_seq_minibatches sets
#     n_seq = rollout_steps // seq_len,  seq_batch = n_seq // num_mini_batch
# so transitions per minibatch = seq_len * seq_batch = rollout_steps /
# num_mini_batch = 64 here, INDEPENDENT of seq_len. T=32 stores the same 64
# transition activations as T=1, shaped (32,2) instead of (1,64).
#
# scripts/slurm.py is idempotent: re-run after timeouts to fill missing seeds.

for fov in 9; do
    for T in 1 8 16 32; do
        python scripts/slurm.py \
            --cluster clusters/vulcan-gpu-vmap-32G.json \
            --tasks 5 --time 03:00:00 --runs 10 --force \
            --entry src/rtu_ppo.py \
            -e experiments/E142-bptt/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/BPTTActorCriticMLP_T${T}.json
    done
done

# The stacked backbone (n_blocks=2 [RTU -> MLP] residual blocks per branch),
# swept over the same windows so the depth effect and the window effect are
# separable rather than confounded. Its real-time counterpart is E140's
# RealTimeActorCriticMLPStacked on this same environment.
#
# --tasks 5, matching the single-layer arm above. Depth costs far less memory
# than the parameter ratio suggests. At d_hidden=512 / hidden=64 / n_blocks=2
# the stacked MLP holds 1,431,493 params against the single layer's 315,461, so
# params + Adam moments go from ~3.8MB to ~17MB per run -- 13MB more, on a card
# with 48GB.
#
# The window costs nothing at all. create_seq_minibatches sets
#     n_seq = rollout_steps // seq_len,  seq_batch = n_seq // num_mini_batch
# so transitions per minibatch = seq_len * seq_batch = rollout_steps /
# num_mini_batch = 64 here, INDEPENDENT of seq_len. T=32 stores the same 64
# transition activations as T=1, just shaped (32,2) instead of (1,64). T-BPTT
# trades the RTRL sensitivity carry for a reshape, not for O(seq_len) memory.
#
# If T=32 does OOM, drop --tasks for that window alone: it only changes how runs
# are packed into jobs, never the results, and scripts/slurm.py is idempotent so
# a re-run fills exactly the missing seeds.
for fov in 9; do
    for T in 1 8 16 32; do
        python scripts/slurm.py \
            --cluster clusters/vulcan-gpu-vmap-32G.json \
            --tasks 5 --time 03:00:00 --runs 10 --force \
            --entry src/rtu_ppo.py \
            -e experiments/E142-bptt/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/BPTTActorCriticMLPStacked_T${T}.json
    done
done
