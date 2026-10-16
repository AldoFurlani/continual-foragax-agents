from functools import partial

import jax
from foragax.env import EnvState as ForagaxEnvState
from environments.ForagaxObsExtras import make

import utils.chex as cxu
from utils.rlglue import BaseEnvironment


@cxu.dataclass
class EnvState:
    state: ForagaxEnvState
    key: jax.Array


class Foragax(BaseEnvironment):
    def __init__(self, seed: int, **env_params):
        self.env = make(**env_params)
        self.seed = seed
        self.state = EnvState(state=None, key=jax.random.key(seed))  # type: ignore

    def start(self) -> jax.Array:
        self.state, obs = self._start(self.state)
        return obs

    @partial(jax.jit, static_argnums=0)
    def _start(self, state: EnvState) -> tuple[EnvState, jax.Array]:
        state.key, env_reset_key = jax.random.split(state.key)
        obs, state.state = self.env.reset(env_reset_key)
        return state, obs

    def step(self, action: jax.Array):
        self.state, (obs, reward, done, done, info) = self._step(
            self.state,
            action,
        )
        return (obs, reward, done, done, info)

    @partial(jax.jit, static_argnums=0)
    def _step(self, state: EnvState, action: jax.Array):
        state.key, env_step_key = jax.random.split(state.key)
        obs, state.state, reward, done, info = self.env.step(
            env_step_key, state.state, action
        )
        return state, (obs, reward, done, done, info)
