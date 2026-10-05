#!/bin/bash
# 10M eval, 30 seeds. Run AFTER process_hypers.sh writes the selected configs.
# Compare against E136 PPO_LN_HINT_128 / PPO-RTU_LN_HINT_128 (9x9 rgb, constant
# hint) and E136 Search-Oracle (hand-coded, full map + true rewards) as the ceiling.
for alg in PPO_LN_HINT_128 PPO_LN_NOHINT_128; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 30 --time 12:00:00 --runs 30 --force --entry src/rtu_ppo.py -e experiments/E143-full-view/foragax/ForagaxBig-v5/world/${alg}.json
done

# E136's 9x9 RTU eval used 24 runs / 6h; doubled for the 28x28 conv. Check the
# sweep's elapsed time with sacct before launching and adjust.
for alg in PPO-RTU_LN_HINT_128 PPO-RTU_LN_NOHINT_128; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 12:00:00 --runs 30 --force --entry src/rtu_ppo.py -e experiments/E143-full-view/foragax/ForagaxBig-v5/world/${alg}.json
done

# E142's 9x9 T-BPTT eval used 20 runs / 12h; doubled for the 28x28 conv.
for alg in BPTTActorCriticConv_T2_HINT BPTTActorCriticConv_T2_NOHINT; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 20 --time 24:00:00 --runs 30 --force --entry src/rtu_ppo.py -e experiments/E143-full-view/foragax/ForagaxBig-v5/world/${alg}.json
done
