from functools import partial
from typing import Any, Dict, Tuple
from dataclasses import replace

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from ml_instrumentation.Collector import Collector

import flashbax as fbx
import utils.chex as cxu
from algorithms.nn.NNAgent import AgentState as BaseAgentState
from algorithms.nn.NNAgent import Hypers as BaseHypers
from algorithms.nn.NNAgent import NNAgent
from representations.networks import NetworkBuilder, reluLayers
from utils.jax import huber_loss
from utils.policies import egreedy_probabilities


@cxu.dataclass
class Hypers(BaseHypers):
    target_refresh: int
    sequence_length: int
    burn_in_steps: int


@cxu.dataclass
class AgentState(BaseAgentState):
    target_params: Any
    hypers: Hypers
    carry: Any


def q_loss(q, a, r, gamma, qp):
    vp = qp.max()
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)
    delta = target - q[a]

    return huber_loss(1.0, q[a], target), {
        "delta": delta,
    }


class AADRQN(NNAgent):
    def __init__(
        self,
        observations: Tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        # set up the target network parameters
        self.target_refresh = params["target_refresh"]

        self.num_actions = actions

        dummy_timestep = {
            "x": jnp.zeros(self.observations),
            "carry": jnp.zeros(self.hidden_size),
            "reset": jnp.bool(True),
            "last_a": jnp.int32(-1),
            "last_a_encoded": jnp.zeros((1, actions)),
            "a": jnp.int32(0),
            "r": jnp.float32(0),
            "gamma": jnp.float32(0),
        }
        buffer_state = self.buffer.init(dummy_timestep)

        hypers = Hypers(
            **self.state.hypers.__dict__,
            target_refresh=params["target_refresh"],
            sequence_length=self.sequence_length,
            burn_in_steps=params.get("burn_in_steps", 0),
        )

        self.state = AgentState(
            **{
                k: v
                for k, v in self.state.__dict__.items()
                if k != "hypers" and k != "buffer_state"
            },
            target_params=self.state.params,
            buffer_state=buffer_state,
            carry=None,
            hypers=hypers,
        )

        self.burn_in_steps = self.state.hypers.burn_in_steps

    def get_feature_function(self, builder: NetworkBuilder):
        return builder.getMultiplicativeActionRecurrentFeatureFunction()

    def encode_action(self, a):
        return jax.nn.one_hot(a, self.num_actions)

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        name = self.rep_params["type"]
        hidden = self.rep_params["hidden"]
        num_layers = self.rep_params.get("num_layers", 2)
        head_layers = [hidden] * num_layers
        layer_norm = "LayerNorm" in name

        def q_net_builder():
            layers = reluLayers(head_layers, name="q", layer_norm=layer_norm)
            layers += [hk.Linear(self.actions, name="q")]
            return hk.Sequential(layers)

        self.q_net, _, self.q = builder.addHead(q_net_builder, name="q")

    # internal compiled version of the value function
    @partial(jax.jit, static_argnums=0)
    def _values(
        self,
        state: AgentState,
        x: jax.Array,
        last_a: jax.Array,
        carry: jax.Array = None,
    ):  # type: ignore
        phi = self.phi(state.params, x, a=last_a, carry=carry)
        return self.q(state.params, phi[0][:, -1]), phi[1][:, -1], phi[2]

    @partial(jax.jit, static_argnums=0)
    def _policy(
        self,
        state: AgentState,
        obs: jax.Array,
        last_a: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        obs = jnp.expand_dims(obs, 0)
        q, carry, _ = self._values(state, obs, last_a, carry=state.carry)
        pi = egreedy_probabilities(q, self.actions, state.hypers.epsilon)[0]
        return pi, carry

    @partial(jax.jit, static_argnums=0)
    def act(
        self, state: AgentState, obs: jax.Array, last_a: jax.Array
    ) -> tuple[AgentState, jax.Array]:
        last_a = self.encode_action(last_a)
        pi, state.carry = self._policy(state, obs, last_a)
        state.key, sample_key = jax.random.split(state.key)
        a = jax.random.choice(sample_key, self.actions, p=pi)
        return state, a

    @partial(jax.jit, static_argnums=0)
    def _update(self, state: AgentState):
        updates = state.updates + 1

        state.key, buffer_sample_key = jax.random.split(state.key)
        batch = self.buffer.sample(state.buffer_state, buffer_sample_key)

        state, metrics = self._computeUpdate(
            state, batch.experience, batch.probabilities
        )

        # Not doing Prioritized Buffer for now
        # priorities = metrics["delta"]
        # buffer_state = self.buffer.set_priorities(
        #     state.buffer_state, batch.indices, priorities
        # )

        target_params = jax.lax.cond(
            updates % state.hypers.target_refresh == 0,
            lambda: state.params,
            lambda: state.target_params,
        )

        return replace(
            state,
            updates=updates,
            # buffer_state=buffer_state,
            target_params=target_params,
        ), metrics

    # -------------
    # -- Updates --
    # -------------
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Dict, weights: jax.Array):
        grad_fn = jax.grad(self._loss, has_aux=True)
        grad, metrics = grad_fn(state.params, state.target_params, batch, weights)
        optimizer = self._build_optimizer(state.hypers.optimizer, state.hypers.swr)

        new_params = {}
        new_optim = {}
        weight_change = 0
        for name, p in state.params.items():
            updates, optim = optimizer.update(grad[name], state.optim[name], p)
            new_params[name] = optax.apply_updates(p, updates)
            new_optim[name] = optim
            flat_updates, _ = ravel_pytree(updates)
            weight_change += jnp.linalg.norm(flat_updates, ord=1)
        metrics["weight_change"] = weight_change

        return replace(state, params=new_params, optim=new_optim), metrics

    # Of shape: <batch, sequence, *feat>
    # TODO: Important, this assumes that no truncation happens, that is end() is properly called before calling start()
    def _loss(
        self, params: hk.Params, target: hk.Params, batch: Dict, weights: jax.Array
    ):
        B, T = batch["a"][:, :-1].shape
        weights = jnp.broadcast_to(weights[:, None], (B, T))
        x = batch["x"][:, :-1]
        xp = batch["x"][:, 1:]
        a = batch["a"][:, :-1]
        last_a_encoded = batch["last_a_encoded"][:, :-1]
        last_a_encodedp = batch["last_a_encoded"][:, 1:]
        r = batch["r"][:, :-1]
        g = batch["gamma"][:, :-1]
        carry = batch["carry"][:, :-1]
        carryp = batch["carry"][:, 1:]
        # reset[t] marks x[t] as an episode-start; reset_p[t] marks xp[t] (== x[t+1]) as one.
        reset = batch["reset"][:, :-1]
        reset_p = batch["reset"][:, 1:]

        # Perform burn-in
        if self.burn_in_steps > 0:
            b_x, x = jnp.hsplit(x, [self.burn_in_steps])
            b_xp, xp = jnp.hsplit(xp, [self.burn_in_steps])
            b_reset, reset = jnp.hsplit(reset, [self.burn_in_steps])
            b_reset_p, reset_p = jnp.hsplit(reset_p, [self.burn_in_steps])
            b_carry, carry = jnp.hsplit(carry, [self.burn_in_steps])
            b_carryp, carryp = jnp.hsplit(carryp, [self.burn_in_steps])
            b_last_a_encoded, last_a_encoded = jnp.hsplit(
                last_a_encoded, [self.burn_in_steps]
            )
            b_last_a_encodedp, last_a_encodedp = jnp.hsplit(
                last_a_encodedp, [self.burn_in_steps]
            )
            _, a = jnp.hsplit(a, [self.burn_in_steps])
            _, r = jnp.hsplit(r, [self.burn_in_steps])
            _, g = jnp.hsplit(g, [self.burn_in_steps])
            _, weights = jnp.hsplit(weights, [self.burn_in_steps])

            carry = carry.at[:, 0].set(
                jax.lax.stop_gradient(
                    self.phi(
                        params,
                        b_x,
                        a=b_last_a_encoded,
                        carry=b_carry,
                        reset=b_reset,
                        is_target=False,
                    )[1][:, -1, ...]
                )
            )
            carryp = carryp.at[:, 0].set(
                jax.lax.stop_gradient(
                    self.phi(
                        target,
                        b_xp,
                        a=b_last_a_encodedp,
                        carry=b_carryp,
                        reset=b_reset_p,
                        is_target=True,
                    )[1][:, -1, ...]
                )
            )

        phi = self.phi(
            params, x, a=last_a_encoded, carry=carry, reset=reset, is_target=False
        )[0]
        phi_p = self.phi(
            target, xp, a=last_a_encodedp, carry=carryp, reset=reset_p, is_target=True
        )[0]

        qs = self.q(params, phi)
        qsp = self.q(target, phi_p)

        qs = qs.reshape(-1, qs.shape[-1])
        qsp = qsp.reshape(-1, qsp.shape[-1])
        a = a.ravel()
        r = r.ravel()
        g = g.ravel()
        # weights = weights.ravel()

        batch_loss = jax.vmap(q_loss, in_axes=0)
        losses, batch_metrics = batch_loss(qs, a, r, g, qsp)

        # chex.assert_equal_shape((weights, losses))
        loss = jnp.mean(losses)  # jnp.mean(weights * losses)

        # aggregate metrics
        metrics = {
            "loss": loss,
            "abs_td_error": jnp.mean(jnp.abs(batch_metrics["delta"])),
            "squared_td_error": jnp.mean(jnp.square(batch_metrics["delta"])),
        }

        return loss, metrics

    @partial(jax.jit, static_argnums=0)
    def _start(self, state: AgentState, obs: jax.Array):
        state.carry = None
        state.last_timestep.update(
            {
                "last_a": jnp.int32(-1),
                "last_a_encoded": self.encode_action(jnp.int32(-1)),
            }
        )
        state, a = self.act(state, obs, state.last_timestep["last_a"])
        state.last_timestep.update(
            {
                "x": obs,
                "a": a,
                "carry": jnp.zeros(
                    self.hidden_size
                ),  # Replaced with learnt init within alg
                "reset": jnp.bool(True),
            }
        )
        state = replace(state, steps=state.steps + 1)
        state = self._decay_epsilon(state)
        state = self._maybe_update(state)
        return state, a

    @partial(jax.jit, static_argnums=0)
    def _step(
        self,
        state: AgentState,
        reward: jax.Array,
        obs: jax.Array,
        extra: Dict[str, jax.Array],
    ):
        # see if the problem specified a discount term
        gamma = extra.get("gamma", 1.0)

        # possibly process the reward
        if self.reward_clip > 0:
            reward = jnp.clip(reward, -self.reward_clip, self.reward_clip)

        state.last_timestep.update(
            {
                "r": reward,
                "gamma": jnp.float32(state.hypers.gamma * gamma),
            }
        )
        batch_sequence = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (1, 1, *x.shape)), state.last_timestep
        )
        buffer_state = self.buffer.add(state.buffer_state, batch_sequence)
        state = replace(state, buffer_state=buffer_state)
        state.last_timestep.update(
            {
                "last_a": state.last_timestep["a"],
                "last_a_encoded": self.encode_action(state.last_timestep["a"]),
            }
        )
        last_carry = state.carry[0]
        state, a = self.act(state, obs, jnp.array(state.last_timestep["last_a"]))
        state.last_timestep.update(
            {"x": obs, "a": a, "carry": last_carry, "reset": jnp.bool(False)}
        )
        state = self._maybe_update(state)
        state = replace(state, steps=state.steps + 1)
        state = self._decay_epsilon(state)
        return state, a

    @partial(jax.jit, static_argnums=0)
    def _end(self, state, reward: jax.Array, extra: Dict[str, jax.Array]):
        # possibly process the reward
        if self.reward_clip > 0:
            reward = jnp.clip(reward, -self.reward_clip, self.reward_clip)

        state.last_timestep.update({"r": reward, "gamma": jnp.float32(0)})
        batch_sequence = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (1, 1, *x.shape)), state.last_timestep
        )
        buffer_state = self.buffer.add(state.buffer_state, batch_sequence)
        state = replace(state, buffer_state=buffer_state)
        state = self._maybe_update(state)
        state = replace(state, steps=state.steps + 1)
        state = self._decay_epsilon(state)
        return state
