#!/bin/bash
# Hyperparameter sweep for the full-information PPO baseline on ForagaxBig-v5,
# tuned at 1M steps = 10% of the 10M eval. Same grid, network and PPO settings
# as E136's PPO_LN_HINT_128 (alpha x lr_scale x entropy_coef, 27 configs).
#
# world/PPO_LN_HINT_128   full 28x28 map, constant hint
# world/PPO_LN_NOHINT_128 full 28x28 map, no hint (does the hint do anything?)
# world/PPO-RTU_LN_*      the same two with E136's RTU-PPO. With the full map and
#                         a constant hint there is almost nothing to remember, so
#                         an RTU-vs-feedforward gap here is not memory (E139 found
#                         RTUs keep plasticity that vanilla PPO loses). Without the
#                         hint, the RTU can remember what each patch tasted like,
#                         so that pair measures memory for values without the
#                         localisation problem.
# world/BPTTActorCriticConv_T2_*  the same two with E142's T-BPTT agent (T=2),
#                         sweep config copied from E142 (num_mini_batch=8).
#                         E142 ran 9x9 with the intermittent hint, so there is
#                         no 9x9 constant-hint T-BPTT control for the HINT run.
#
# Observation is E136's rgb view, widened to the whole map, plus one channel
# marking the agent's cell (agent_channel, src/environments/ForagaxAgentChannel.py):
# the full-world rgb view is in fixed coordinates and does not otherwise show
# where the agent is. The control is E136's PPO_LN_HINT_128 (9x9 rgb, constant
# hint), so view size is the only difference (E136's PPO-RTU_LN_HINT_128
# for the RTU pair).
#
# --tasks 128 is half of E136's 256: the 28x28 input makes each conv pass ~10x
# larger. Estimated, not measured; scripts/slurm.py is idempotent, so re-run to
# fill seeds lost to OOM or timeout.
for alg in PPO_LN_HINT_128 PPO_LN_NOHINT_128; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 128 --time 03:00:00 --runs 10 --force --entry src/rtu_ppo.py -e experiments/E143-full-view/foragax-sweep/ForagaxBig-v5/world/${alg}.json
done

# RTU-PPO packs fewer runs per job (E136's 9x9 sweep used 24 / 3h); the extra
# hours are for the 28x28 conv. The RTU sits after the 64-unit dense layer, so
# its real-time update costs the same as at 9x9. Estimated, not measured.
for alg in PPO-RTU_LN_HINT_128 PPO-RTU_LN_NOHINT_128; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 24 --time 06:00:00 --runs 10 --force --entry src/rtu_ppo.py -e experiments/E143-full-view/foragax-sweep/ForagaxBig-v5/world/${alg}.json
done

# T-BPTT packs 20 runs per job as in E142 (9x9: 20 / 3h); doubled hours for the
# 28x28 conv. Estimated, not measured.
for alg in BPTTActorCriticConv_T2_HINT BPTTActorCriticConv_T2_NOHINT; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 20 --time 06:00:00 --runs 10 --force --entry src/rtu_ppo.py -e experiments/E143-full-view/foragax-sweep/ForagaxBig-v5/world/${alg}.json
done
