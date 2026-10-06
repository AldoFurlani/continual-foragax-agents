#!/bin/bash
# 10M eval, 30 seeds. Run AFTER process_hypers.sh writes the selected configs,
# and only once no eval job is still running: scripts/slurm.py counts results
# on disk, so runs still in flight would be submitted again.
#
# Sized from the 1M sweep's measured job times x10 steps. A run that hits the
# time limit saves nothing (only the collector is checkpointed), so leave margin.
# Compare against E136 PPO_LN_HINT_128 / PPO-RTU_LN_HINT_128 (9x9 rgb, constant
# hint) and E136 Search-Oracle (hand-coded, full map + true rewards) as the ceiling.
for alg in PPO_LN_HINT_128 PPO_LN_NOHINT_128; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 15 --time 24:00:00 --runs 30 --force --entry src/rtu_ppo.py -e experiments/E143-full-view/foragax/ForagaxBig-v5/world/${alg}.json
done

# RTU sweep: ~1h42 per 1M-step job of ~23 runs, so up to ~17h at 10M;
# 10 runs/job under a 24h limit.
for alg in PPO-RTU_LN_HINT_128 PPO-RTU_LN_NOHINT_128; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 10 --time 24:00:00 --runs 30 --force --entry src/rtu_ppo.py -e experiments/E143-full-view/foragax/ForagaxBig-v5/world/${alg}.json
done

# T-BPTT sweep: ~42min per 1M-step job, so ~7h at 10M; 24h limit.
for alg in BPTTActorCriticConv_T2_HINT BPTTActorCriticConv_T2_NOHINT; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 20 --time 24:00:00 --runs 30 --force --entry src/rtu_ppo.py -e experiments/E143-full-view/foragax/ForagaxBig-v5/world/${alg}.json
done
