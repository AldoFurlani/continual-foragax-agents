# Full-world Foragax observation with an agent-position channel.
#
# With aperture_size=-1 the "rgb" and "color" views render the world in fixed
# coordinates and nothing marks the agent's cell ("object" view does mark it,
# but drops the per-generation food colours). This appends one channel holding
# a single 1 at the agent's position, so the full-world rgb view carries the
# same information as E136's 9x9 rgb aperture plus the agent's location.
#
# Lives here rather than in continual-foragax so runs do not wait on an
# upstream release; it overrides get_obs, which both reset and step call.
import jax.numpy as jnp
from foragax.env import ForagaxEnv
from foragax.registry import make as foragax_make
from gymnax.environments import spaces


class AgentChannelForagaxEnv(ForagaxEnv):
    def get_obs(self, state, params, key=None):
        obs = super().get_obs(state, params, key)
        img = obs["image"] if isinstance(obs, dict) else obs
        agent = jnp.zeros((*img.shape[:2], 1), img.dtype)
        agent = agent.at[state.pos[1], state.pos[0], 0].set(1.0)
        img = jnp.concatenate([img, agent], axis=-1)
        return {**obs, "image": img} if isinstance(obs, dict) else img

    # ForagaxEnv annotates this as Box but returns a Dict when return_hint is on.
    def observation_space(self, params):  # type: ignore
        space = super().observation_space(params)
        image = space.spaces["image"] if isinstance(space, spaces.Dict) else space
        image = spaces.Box(0, 1, (*image.shape[:-1], image.shape[-1] + 1))
        if isinstance(space, spaces.Dict):
            return spaces.Dict({**space.spaces, "image": image})
        return image


def make(env_id, agent_channel=False, **kwargs):
    env = foragax_make(env_id, **kwargs)
    if not agent_channel:
        return env
    if not env.full_world or env.observation_type == "object":
        raise ValueError(
            "agent_channel needs aperture_size=-1 with an rgb or color view; "
            "apertures are agent-centred and the object view already marks the agent"
        )
    # registry.make builds a plain ForagaxEnv with env-specific objects; retype
    # it rather than duplicating that construction.
    env.__class__ = AgentChannelForagaxEnv
    return env
