"""
Roll out the Search-9 scripted agent in a Foragax environment and write the
rollout to a GIF.

Usage:
    python scripts/render_search9_frames.py [--steps 500] [-o out.gif] \\
        [--env-id ForagaxSquareWaveTwoBiome-v11] [--aperture-size 9] \\
        [--render-mode world_reward] [--fps 20] [--seed 0] \\
        [--biome-consumption-threshold 20]
"""

import argparse
import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import imageio.v2 as imageio
import jax
import numpy as np
from foragax.registry import make as foragax_make  # noqa: E402
from ml_instrumentation.Collector import Collector

from algorithms.SearchAgent import SearchAgent  # noqa: E402


def rollout_and_render(
    env_id: str,
    aperture_size: int,
    n_steps: int,
    render_mode: str,
    seed: int,
    output: str,
    fps: float,
    env_kwargs: dict,
):
    env = foragax_make(
        env_id=env_id,
        aperture_size=aperture_size,
        observation_type="object",
        **env_kwargs,
    )

    agent_params = {
        "mode": "aperture",
        "nowrap": False,
        "reward_prioritization": True,
        "channel_priorities": {"2": -1},
    }
    agent = SearchAgent(
        observations=(aperture_size, aperture_size, 6),
        actions=4,
        params=agent_params,
        collector=Collector(),
        seed=seed,
    )

    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    action = agent.start(obs)

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    is_mp4 = output.lower().endswith(".mp4")
    writer_kwargs = (
        dict(
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=2,
        )
        if is_mp4
        else dict(mode="I", fps=fps, loop=0)
    )

    with imageio.get_writer(output, **writer_kwargs) as writer:
        for step in range(n_steps):
            key, step_key = jax.random.split(key)
            obs, state, reward, done, info = env.step(step_key, state, action)

            frame = env.render(state, None, render_mode=render_mode)
            writer.append_data(np.asarray(frame).astype(np.uint8))

            action = agent.step(reward, obs, info)

            if step % 50 == 0:
                print(f"  step {step}/{n_steps}")

    print(f"Saved {n_steps} frames -> {output} ({fps} fps)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "-o",
        "--output",
        default="experiments/X33-ForagaxSquareWaveTwoBiome-v11/foragax/"
        "ForagaxSquareWaveTwoBiome-v11/plots/search9.gif",
        help="Output .gif or .mp4 path",
    )
    parser.add_argument("--env-id", default="ForagaxSquareWaveTwoBiome-v11")
    parser.add_argument("--aperture-size", type=int, default=9)
    parser.add_argument("--render-mode", default="world_reward")
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--biome-consumption-threshold",
        type=int,
        default=None,
        help="Override the env's biome_consumption_threshold (item count before "
        "a biome respawns with new random parameters; must stay an int to hit "
        "the count-based branch rather than the fractional-rate branch).",
    )
    args = parser.parse_args()

    env_kwargs = {}
    if args.biome_consumption_threshold is not None:
        env_kwargs["biome_consumption_threshold"] = args.biome_consumption_threshold

    rollout_and_render(
        env_id=args.env_id,
        aperture_size=args.aperture_size,
        n_steps=args.steps,
        render_mode=args.render_mode,
        seed=args.seed,
        output=args.output,
        fps=args.fps,
        env_kwargs=env_kwargs,
    )


if __name__ == "__main__":
    main()
