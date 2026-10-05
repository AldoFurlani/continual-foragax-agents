#!/usr/bin/env bash
# Poster figure: state construction mitigates partial observability tracking failure (ForagaxBig-v5).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

EXP=experiments/XN37/foragax/ForagaxBig-v5

# Learning curve: Oracle Search, Local Search, RTU-PPO, RTU-PPO Simple Memory
uv run python src/learning_curve.py "$EXP" \
  --plot-name poster-big-state-construction \
  --filter-alg-apertures Search-Oracle Search-9 PPO_LN_128_1:9 PPO-RTU_LN_128_1:9 PPO-RTU_LN_RT_128_1:9 \
  --colors 'Search-Oracle:tol:vibrant:red' 'Search-9:tol:vibrant:orange' 'PPO_LN_128_1:tol:muted:indigo' 'PPO-RTU_LN_128_1:tol:vibrant:magenta' 'PPO-RTU_LN_RT_128_1:#1771a4' \
  --rename 'Search-Oracle:Oracle Search' 'Search-9:Local Search' 'PPO_LN_128_1:PPO' 'PPO-RTU_LN_128_1:RTU-PPO' 'PPO-RTU_LN_RT_128_1:RTU-PPO Simple Memory' \
  --metric ewm_reward_5 \
  --disable-fov --no-legend --end-frame 10000000 --ylim 0 0.4 \
  --aspect-ratio 2.33 --font-size 48 --num-xticks 2 --num-yticks 3
