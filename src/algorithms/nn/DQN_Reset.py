from dataclasses import replace
from functools import partial
from typing import Dict

import jax
import jax.lax
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.nn.DQN import DQN
from algorithms.nn.DQN import AgentState as BaseAgentState
from algorithms.nn.DQN import Hypers as BaseHypers


@cxu.dataclass
class Hypers(BaseHypers):
    reset_steps: int
    reset_head_only: bool


@cxu.dataclass
class AgentState(BaseAgentState):
    hypers: Hypers


class DQN_Reset(DQN):
    def __init__(
        self,
        observations: tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)

        hypers = Hypers(
            **self.state.hypers.__dict__,
            reset_steps=params["reset_steps"],
            reset_head_only=params["reset_head_only"],
        )

        self.state = AgentState(
            **{k: v for k, v in self.state.__dict__.items() if k != "hypers"},
            hypers=hypers,
        )

        self.periodic_freq = int(params["reset_steps"])

    def _periodic_step(self, state: AgentState) -> AgentState:
        return self._reset(state)

    @partial(jax.jit, static_argnums=0)
    def _reset(self, state: AgentState):
        key, q_key, body_key = jax.random.split(state.key, 3)
        optimizer = self._build_optimizer(state.hypers.optimizer, state.hypers.swr)

        # Reinitialize q-network parameters
        q_params = self.q_net.init(q_key, self.builder._sample_phi)

        # Conditionally reset body parameters
        def _reset_body():
            return self.builder.reset(body_key)["phi"]

        def _no_reset_body():
            return state.params["phi"]

        new_phi = jax.lax.cond(
            state.hypers.reset_head_only,
            _no_reset_body,
            _reset_body,
        )

        # Update params with new q and phi
        new_params = {
            **state.params,
            "q": q_params,
            "phi": new_phi,
        }
        new_params["phi"] = new_phi

        # Reinitialize optimizer state for the updated parameters
        # Create new optimizer state by initializing with the new params
        new_optim = optimizer.init(new_params)

        # Use tree_map_with_path to selectively replace only the parts we want to reset
        def should_reset(path, _):
            key_path = tuple(k.key if hasattr(k, "key") else str(k) for k in path)
            if len(key_path) > 0:
                # Always reset 'q'
                if key_path[0] == "q":
                    return True
                # Reset 'phi' only if not head_only
                if key_path[0] == "phi" and not state.hypers.reset_head_only:
                    return True
            return False

        # Blend old and new optimizer states
        final_optim = jax.tree_util.tree_map_with_path(
            lambda path, new_val, old_val: (
                new_val if should_reset(path, new_val) else old_val
            ),
            new_optim,
            state.optim,
        )

        return replace(state, key=key, params=new_params, optim=final_optim)
