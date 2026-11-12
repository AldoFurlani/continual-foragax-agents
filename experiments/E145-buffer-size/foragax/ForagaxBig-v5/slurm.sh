#!/bin/bash
# 10M eval, 30 seeds. Run AFTER process_hypers.sh has filled the selected
# epsilon / alpha into ./9/, and only once no eval job is still running:
# scripts/slurm.py counts results on disk, so runs still in flight would be
# submitted again.
#
# The 1K arm is E144's DQN_LN_HINT / DQN_LN_HINT_TARGET / DRQN_LN_HINT eval,
# not re-run here.
#
# DQN: seeds run vmapped, 15 per process on one GPU, as in the sweep, rather
# than E144's one process per run (vulcan-gpu-mps-c8-16G, 4 jobs per config)
# -- ~8x less by scripts/slurm.py's estimate. The sweep's DQN jobs took 43 min
# (B10K) and ~22 min (B100K) per 1M steps, and vmapped step time barely depends
# on the number of runs, so a 10M eval job should take ~7h / ~4h.
#
# Why 15 runs per job: with no save_every, a run's 10M steps are one chunk, so
# every step's logged data stays on the GPU and lands in host memory at once
# at the end -- ~5.5 GB for 15 runs, the scale of the sweep's 150 x 1M jobs.
# 30 runs would be ~11 GB on each side, close to the job's 32 GB. 15 also
# keeps B100K's O(total buffer) per-step buffer-write cost at the sweep's.
EVAL=experiments/E145-buffer-size/foragax/ForagaxBig-v5/9

for alg in DQN_LN_HINT DQN_LN_HINT_TARGET; do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 15 --time 24:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/${alg}_B10K.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --tasks 15 --time 24:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/${alg}_B100K.json
done

# DRQN: one process per run, 8 per GPU on vulcan-gpu-mps-c8-16G.json (16G host
# and 10% of the GPU each), as E144's eval: vulcan-gpu-mps.json hit host OOM
# on 10M runs. A 10M run needs several 24h jobs; save_every 1M (from the sweep
# config) checkpoints the full agent state, buffer included (~130 MB per run).
# When a job times out, re-run this script once no DRQN job is still running,
# and it resumes each unfinished run from its last checkpoint.
python scripts/slurm.py --cluster clusters/vulcan-gpu-mps-c8-16G.json --time 24:00:00 --runs 30 --force --entry src/continuing_main.py -e $EVAL/DRQN_LN_HINT_B100K.json
