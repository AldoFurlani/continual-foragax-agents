# Optional extra observation features for Foragax, layered on registry.make.
#
# agent_channel  With aperture_size=-1 the "rgb" and "color" views render the
#                world in fixed coordinates and nothing marks the agent's cell
#                ("object" view does, but drops the per-generation food colours).
#                Appends one image channel holding a single 1 at the agent.
#
# patch_offsets  Appends to the hint vector the agent-relative offset (dx, dy) to
#                the centre of each food biome, in hint-slot order, taking the
#                shortest way around the wrapped world and scaled by half the
#                world size into [-1, 1). The agent keeps its agent-centred
#                aperture, so local foraging is unchanged; the offsets tell it
#                where each patch is from where it stands, so acting on the hint
#                needs no localisation. The hint is extended rather than given a
#                new key because both rtu_ppo and NNAgent already feed the hint
#                vector to the network as scalar features.
#
# Lives here rather than in continual-foragax so runs do not wait on an
# upstream release; it overrides get_obs, which both reset and step call.
from typing import cast

import jax.numpy as jnp
import numpy as np
from foragax.env import ForagaxEnv
from foragax.registry import make as foragax_make
from gymnax.environments import spaces


class ExtendedObsForagaxEnv(ForagaxEnv):
    agent_channel: bool
    patch_offsets: bool
    # Set while rendering, so the video's hint bar shows only the one-hot hint.
    _render_plain_hint: bool = False

    def _setup_extras(self, agent_channel: bool, patch_offsets: bool):
        self.agent_channel = agent_channel
        self.patch_offsets = patch_offsets
        food = np.asarray(self.food_biome_indices)
        starts, stops = self.biome_starts[food], self.biome_stops[food]
        # (num_food_biomes, 2) as (x, y), matching state.pos.
        self._patch_centres = jnp.asarray((starts + stops - 1) / 2.0, jnp.float32)
        self._world_size = jnp.asarray(self.size, jnp.float32)

    def get_obs(self, state, params, key=None):
        obs = super().get_obs(state, params, key)
        if self.agent_channel:
            img = obs["image"] if isinstance(obs, dict) else obs
            agent = jnp.zeros((*img.shape[:2], 1), img.dtype)
            agent = agent.at[state.pos[1], state.pos[0], 0].set(1.0)
            img = jnp.concatenate([img, agent], axis=-1)
            obs = {**obs, "image": img} if isinstance(obs, dict) else img
        if self.patch_offsets and not self._render_plain_hint and isinstance(obs, dict):
            half = self._world_size / 2.0
            delta = self._patch_centres - state.pos.astype(jnp.float32)
            offsets = (jnp.mod(delta + half, self._world_size) - half) / half
            hint = jnp.concatenate([obs["hint"], offsets.reshape(-1)])
            obs = {**obs, "hint": hint.astype(obs["hint"].dtype)}
        return obs

    # ForagaxEnv annotates this as Box but returns a Dict when return_hint is on.
    def observation_space(self, params):  # type: ignore
        space = super().observation_space(params)
        image = space.spaces["image"] if isinstance(space, spaces.Dict) else space
        if self.agent_channel:
            image = spaces.Box(0, 1, (*image.shape[:-1], image.shape[-1] + 1))
        if not isinstance(space, spaces.Dict):
            return image
        out = {**space.spaces, "image": image}
        if self.patch_offsets:
            n = space.spaces["hint"].shape[0] + 2 * self.num_food_biomes
            out["hint"] = spaces.Box(-1, 1, (n,))
        return spaces.Dict(out)

    # Overrides the jitted ForagaxEnv.render; super().render is still the jitted one.
    def render(self, *args, **kwargs):  # type: ignore
        self._render_plain_hint = True
        try:
            return super().render(*args, **kwargs)
        finally:
            self._render_plain_hint = False


def make(env_id, agent_channel=False, patch_offsets=False, **kwargs):
    env = foragax_make(env_id, **kwargs)
    if not (agent_channel or patch_offsets):
        return env
    if agent_channel and (not env.full_world or env.observation_type == "object"):
        raise ValueError(
            "agent_channel needs aperture_size=-1 with an rgb or color view; "
            "apertures are agent-centred and the object view already marks the agent"
        )
    if patch_offsets and not env.return_hint:
        raise ValueError("patch_offsets extends the hint vector, so needs return_hint")
    # registry.make builds a plain ForagaxEnv with env-specific objects; retype
    # it rather than duplicating that construction.
    env.__class__ = ExtendedObsForagaxEnv
    ext = cast(ExtendedObsForagaxEnv, env)
    ext._setup_extras(agent_channel, patch_offsets)
    return ext
