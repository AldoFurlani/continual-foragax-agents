#!/bin/bash
# 10M eval, 30 seeds. Run AFTER process_hypers.sh has filled the selected
# epsilon / alpha into ./9/, and only once no eval job is still running:
# scripts/slurm.py counts results on disk, so runs still in flight would be
# submitted again.
#
# The 1K arm is E144's DQN_LN_HINT / DQN_LN_HINT_TARGET / DRQN_LN_HINT eval,
# not re-run here.
#
# DQN: the 30 seeds run vmapped in one process on one GPU, as in the sweep,
# rather than E144's one process per run (vulcan-gpu-mps-c8-16G, 4 jobs per
# config) -- ~16x less by scripts/slurm.py's estimate. The sweep's DQN jobs
# took 43 min (B10K, 150 runs) and ~22 min (B100K, 15 runs) per 1M steps, and
# vmapped step time barely depends on the number of runs, so a 10M eval job
# should take ~7h / ~4h. B100K keeps the sweep's 15 runs per job: vmapped runs
# pay an O(total buffer) cost per step for the buffer write.
#
# Before launching, add save_every 1M to the selected DQN eval configs
# (process_hypers.sh does not write it):
#   for f in $EVAL/DQN_*.json; do .venv/bin/python -c "import json,sys; p=sys.argv[1]; c=json.load(open(p)); c['metaParameters']['experiment']['save_every']=1000000; json.dump(c,open(p,'w'),indent=4)" $f; done
# It chunks the per-step data a run holds on the GPU (one 10M chunk for 30
# runs would be ~11 GB) and checkpoints, so a timed-out job resumes when re-run.
EVAL=experiments/E145-buffer-size/foragax/ForagaxBig-v5/9

for alg in DQN_LN_HINT DQN_LN_HINT_TARGET; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 30 --time 24:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/${alg}_B10K.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 15 --time 24:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/${alg}_B100K.json
done

# DRQN: one process per run, 8 per GPU on vulcan-gpu-mps-c8-16G.json (16G host
# and 10% of the GPU each), as E144's eval: vulcan-gpu-mps.json hit host OOM
# on 10M runs. A 10M run needs several 24h jobs; save_every 1M (from the sweep
# config) checkpoints the full agent state, buffer included (~130 MB per run).
# When a job times out, re-run this script once no DRQN job is still running,
# and it resumes each unfinished run from its last checkpoint.
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-c8-16G.json --time 24:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/DRQN_LN_HINT_B100K.json
