#!/bin/bash
#SBATCH --account=aip-amw8
#SBATCH --job-name=ppo_cluster_diag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/ppo_cluster_diag_%j.out
#
# Feedforward PPO (ActorCriticConv) learns on a laptop CPU with the current code
# but stayed at random-play level in every E144 cluster run, while T-BPTT
# learned on the cluster. This runs E136's PPO_LN_HINT_128 config (which learned
# in March) for ~150k steps, 2 seeds, three ways, to separate GPU vs code:
#   new-gpu  current code, GPU      (what E144 did)
#   new-cpu  current code, CPU      (what learned on the laptop)
#   old-gpu  E136's code (e34cfc69), GPU, same .venv
# Submit from the repo root:  sbatch scripts/diag/ppo_cluster_diag.sh
set -e
REPO=$PWD
D=$SCRATCH/ppo_diag
rm -rf $D; mkdir -p $D/experiments/DIAG/foragax/ForagaxBig-v5/9
cp -R .venv $SLURM_TMPDIR
PY=$SLURM_TMPDIR/.venv/bin/python
CFG=$D/experiments/DIAG/foragax/ForagaxBig-v5/9/PPO_LN_HINT_128.json
$PY - $CFG <<'PYEOF'
import json, sys
c = json.load(open("experiments/E136-big/foragax/ForagaxBig-v5/9/PPO_LN_HINT_128.json"))
c["metaParameters"]["experiment"] = {"seed_offset": 0}
json.dump(c, open(sys.argv[1], "w"))
PYEOF

echo "== versions"; $PY -c "
import importlib.metadata as m, sys, jax
print(sys.version.split()[0], jax.devices())
for p in ['jax','jaxlib','jax-cuda12-plugin','flax','distrax','optax','continual-foragax','gymnax','numpy','chex']:
    try: print(' ', p, m.version(p))
    except Exception: print(' ', p, '-')"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

git -C $REPO worktree add -f $D/old e34cfc69 > /dev/null

run() {  # run <label> <code dir> <extra flags>
    (cd $2 && $PY src/rtu_ppo.py -e $CFG -i 0 1 --max_steps 1200 --silent $3 \
        --save_path $D/$1 --checkpoint_path $D/ckpt_$1) > $D/$1.log 2>&1 || echo "$1 FAILED (see $D/$1.log)"
}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
run new-gpu $REPO --gpu
run old-gpu $D/old --gpu
JAX_PLATFORMS=cpu run new-cpu $REPO ""

echo "== reward by fifth of the run (random play ~0.001; E136-like learning reaches ~0.05+)"
for label in new-gpu old-gpu new-cpu; do
    for f in $(find $D/$label -name "*.npz" -path "*data*" 2>/dev/null); do
        $PY -c "
import numpy as np; r = np.load('$f')['rewards'].astype(float); n = len(r)
print('$label', '$f'.split('/')[-1], [round(r[i*n//5:(i+1)*n//5].mean(), 4) for i in range(5)])"
    done
done
git -C $REPO worktree remove --force $D/old
