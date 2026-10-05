from functools import partial
from typing import Dict, Tuple

import jax
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.BaseAgent import BaseAgent


@cxu.dataclass
class AgentState:
    key: jax.Array
    steps: int = 0


class RandomAgent(BaseAgent):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        self.state = AgentState(
            key=self.key,
        )

    @partial(jax.jit, static_argnums=0)
    def act(
        self,
        state: AgentState,
        obs: jax.Array,
    ) -> tuple[AgentState, jax.Array]:
        state.key, sample_key = jax.random.split(state.key)
        a = jax.random.choice(sample_key, self.actions)
        return state, a

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, obs: jax.Array):
        self.state, a = self._start(self.state, obs)
        return a

    @partial(jax.jit, static_argnums=0)
    def _start(self, state: AgentState, obs: jax.Array):
        return self.act(state, obs)

    def step(self, reward: jax.Array, obs: jax.Array, extra: Dict[str, jax.Array]):
        self.state, a = self._step(self.state, reward, obs, extra)
        return a

    @partial(jax.jit, static_argnums=0)
    def _step(
        self,
        state: AgentState,
        reward: jax.Array,
        obs: jax.Array,
        extra: Dict[str, jax.Array],
    ):
        return self.act(state, obs)

    def end(self, reward: jax.Array, extra: Dict[str, jax.Array]):
        self.state = self._end(self.state, reward, extra)

    @partial(jax.jit, static_argnums=0)
    def _end(self, state, reward: jax.Array, extra: Dict[str, jax.Array]):
        return state
