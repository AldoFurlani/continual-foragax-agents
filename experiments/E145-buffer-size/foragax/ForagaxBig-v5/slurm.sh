#!/bin/bash
# 10M eval, 30 seeds. Run AFTER process_hypers.sh writes the selected configs
# to ./9/, and only once no eval job is still running: scripts/slurm.py counts
# results on disk, so runs still in flight would be submitted again.
#
# The 1K arm is E144's DQN_LN_HINT / DQN_LN_HINT_TARGET / DRQN_LN_HINT eval,
# not re-run here.
#
# One process per run, sharing the node's GPU through MPS, on
# vulcan-gpu-mps-c8-16G.json (8 runs/node, 16G host and 10% of the GPU each),
# as E144's evals: vulcan-gpu-mps.json OOMs on 10M DQN/DRQN runs.
EVAL=experiments/E145-buffer-size/foragax/ForagaxBig-v5/9

for size in 10K 100K; do
    for alg in DQN_LN_HINT DQN_LN_HINT_TARGET; do
        python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-c8-16G.json --time 24:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/${alg}_B${size}.json
    done
done

# DRQN: a 10M run needs several 24h jobs. save_every 1M (in the config)
# checkpoints the full agent state, buffer included (~130 MB per run). When a
# job times out, re-run this script once no DRQN job is still running, and it
# resumes each unfinished run from its last checkpoint.
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-c8-16G.json --time 24:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/DRQN_LN_HINT_B100K.json
