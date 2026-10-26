#!/bin/bash
# 10M eval, 30 seeds. Run AFTER process_hypers.sh writes the selected configs
# to ./9/, and only once no eval job is still running: scripts/slurm.py counts
# results on disk, so runs still in flight would be submitted again.
#
# Before launching, check the selected epsilon in
# foragax-sweep/ForagaxBig-v5/hypers/9/: if it is 0.5 (the new top of the
# grid) for an agent, its selection is again at the edge.
#
# Everything runs one process per run, sharing the node's GPU through MPS, on
# vulcan-gpu-mps-c8-16G.json (8 runs/node, 16G host and 10% of the GPU each):
# E144 found vulcan-gpu-mps.json OOMs on 10M DQN/DRQN runs, and B1M's ~1-1.3 GB
# buffer needs the larger GPU share anyway.
EVAL=experiments/E145-buffer-size/foragax/ForagaxBig-v5/9

for size in 1K 10K 100K 1M; do
    for alg in DQN_LN_HINT DQN_LN_HINT_TARGET; do
        python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-c8-16G.json --time 24:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/${alg}_B${size}.json
    done
done

# DRQN: a 10M run may need several 24h jobs. save_every 1M (in the config)
# checkpoints the full agent state, buffer included -- ~1.3 GB per B1M run, so
# 60 B1M runs hold ~80 GB of checkpoints on scratch while running. When a job
# times out, re-run this script once no DRQN job is still running, and it
# resumes each unfinished run from its last checkpoint.
for size in 1K 10K 100K 1M; do
    for alg in DRQN_LN_HINT DRQN_LN_HINT_TARGET; do
        python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-c8-16G.json --time 24:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/${alg}_B${size}.json
    done
done
