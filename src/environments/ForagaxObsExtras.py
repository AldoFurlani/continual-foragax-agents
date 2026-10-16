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
# target_offset  Appends just two numbers: the same offset, to the patch the hint
#                currently names (zeros while the hint is blank). Given the fixed
#                layout this carries the same information as patch_offsets plus
#                the hint, but the agent no longer has to pick out the hinted
#                patch's offset itself -- it is told which way to go.
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
    target_offset: bool
    # Set while rendering, so the video's hint bar shows only the one-hot hint.
    _render_plain_hint: bool = False

    def _setup_extras(self, agent_channel: bool, patch_offsets: bool, target_offset: bool):
        self.agent_channel = agent_channel
        self.patch_offsets = patch_offsets
        self.target_offset = target_offset
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
        extend = self.patch_offsets or self.target_offset
        if extend and not self._render_plain_hint and isinstance(obs, dict):
            half = self._world_size / 2.0
            delta = self._patch_centres - state.pos.astype(jnp.float32)
            # (num_food_biomes, 2), each in [-1, 1).
            offsets = (jnp.mod(delta + half, self._world_size) - half) / half
            hint = obs["hint"]
            parts = [hint]
            if self.patch_offsets:
                parts.append(offsets.reshape(-1))
            if self.target_offset:
                # The hint is one-hot (or all zeros), so this selects one row.
                parts.append(hint[: self.num_food_biomes] @ offsets)
            obs = {**obs, "hint": jnp.concatenate(parts).astype(hint.dtype)}
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
        n = space.spaces["hint"].shape[0]
        n += 2 * self.num_food_biomes if self.patch_offsets else 0
        n += 2 if self.target_offset else 0
        if n != space.spaces["hint"].shape[0]:
            out["hint"] = spaces.Box(-1, 1, (n,))
        return spaces.Dict(out)

    # Overrides the jitted ForagaxEnv.render; super().render is still the jitted one.
    def render(self, *args, **kwargs):  # type: ignore
        self._render_plain_hint = True
        try:
            return super().render(*args, **kwargs)
        finally:
            self._render_plain_hint = False


def make(env_id, agent_channel=False, patch_offsets=False, target_offset=False, **kwargs):
    env = foragax_make(env_id, **kwargs)
    if not (agent_channel or patch_offsets or target_offset):
        return env
    if agent_channel and (not env.full_world or env.observation_type == "object"):
        raise ValueError(
            "agent_channel needs aperture_size=-1 with an rgb or color view; "
            "apertures are agent-centred and the object view already marks the agent"
        )
    if (patch_offsets or target_offset) and not env.return_hint:
        raise ValueError("patch_offsets / target_offset extend the hint vector, so need return_hint")
    # registry.make builds a plain ForagaxEnv with env-specific objects; retype
    # it rather than duplicating that construction.
    env.__class__ = ExtendedObsForagaxEnv
    ext = cast(ExtendedObsForagaxEnv, env)
    ext._setup_extras(agent_channel, patch_offsets, target_offset)
    return ext
