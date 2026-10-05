#!/usr/bin/env bash
# Poster figures: state construction mitigates partial observability tracking failure.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

EXP=experiments/X33-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11

uv run python src/learning_curve.py "$EXP" \
  --plot-name poster-state-construction-ppo \
  --filter-alg-apertures Search-Oracle ActorCriticMLP:9 ActorCriticMLP-l2-init:9 RealTimeActorCriticMLP:9 \
  --colors 'Search-Oracle:tol:vibrant:red' 'ActorCriticMLP:tol:muted:indigo' 'ActorCriticMLP-l2-init:tol:muted:olive' 'RealTimeActorCriticMLP:tol:vibrant:magenta' \
  --disable-fov --no-legend --end-frame 10000000 --ylim '-0.25' 1.7 \
  --aspect-ratio 2.33 --font-size 48 --num-xticks 2 --num-yticks 3
