from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.nn.NNAgent import AgentState as BaseAgentState
from algorithms.nn.NNAgent import Hypers as BaseHypers
from algorithms.nn.NNAgent import NNAgent
from representations.networks import NetworkBuilder
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


class DRQN(NNAgent):
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
        if isinstance(observations, Mapping):
            image_shape = observations["image"]
        else:
            image_shape = observations

        dummy_hint = None
        dummy_hint_trace = None
        if isinstance(observations, Mapping) and "hint" in observations and "hint" in self.scalar_features:
            dummy_hint = jnp.zeros(observations["hint"])
        if (
            isinstance(observations, Mapping)
            and "hint" in observations
            and "hint_trace" in self.scalar_features
        ):
            dummy_hint_trace = jnp.zeros(observations["hint"])

        dummy_scalars = self.encode_scalar_features(
            jnp.int32(0),
            jnp.float32(0),
            jnp.float32(0),
            dummy_hint,
            dummy_hint_trace,
        )
        dummy_network_carry = self.phi(
            self.state.params,
            jnp.expand_dims(jnp.zeros(image_shape), 0),
            scalars=jnp.expand_dims(dummy_scalars, 0),
            carry=None,
        )[2]
        dummy_carry = self._carry_for_buffer(dummy_network_carry)
        dummy_timestep = {
            "x": jnp.zeros(image_shape),
            "carry": dummy_carry,
            "reset": jnp.bool(True),
            "scalars": dummy_scalars,
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

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        def q_net_builder():
            return hk.Linear(self.actions, name="q")

        self.q_net, _, self.q = builder.addHead(q_net_builder, name="q")

    def _extract_carry(self, carry: Any):
        if isinstance(carry, tuple):
            return carry
        return carry[:, -1]

    def _carry_for_buffer(self, carry: Any):
        if isinstance(carry, tuple):
            return carry
        return carry[0]

    def _slice_carry(self, carry: Any, start: int, end: int | None):
        if isinstance(carry, tuple):
            return jax.tree.map(lambda x: x[:, start:end], carry)
        return carry[:, start:end]

    def _split_carry(self, carry: Any, split: int):
        if isinstance(carry, tuple):
            def _split(x):
                left, right = jnp.hsplit(x, [split])
                return left, right

            split_tree = jax.tree.map(_split, carry)
            left = jax.tree.map(lambda x: x[0], split_tree)
            right = jax.tree.map(lambda x: x[1], split_tree)
            return left, right

        left, right = jnp.hsplit(carry, [split])
        return left, right

    def _set_carry_start(self, carry: Any, new_carry: Any):
        if isinstance(carry, tuple):
            return jax.tree.map(lambda base, new: base.at[:, 0].set(new), carry, new_carry)
        return carry.at[:, 0].set(new_carry)

    # internal compiled version of the value function
    @partial(jax.jit, static_argnums=0)
    def _values(
        self,
        state: AgentState,
        x: jax.Array,
        scalars: jax.Array,
        carry: Optional[Any] = None,
    ):  # type: ignore
        scalars_seq = jnp.expand_dims(scalars, 1)
        phi = self.phi(state.params, x, scalars=scalars_seq, carry=carry)

        return self.q(state.params, phi[0][:, -1]), self._extract_carry(phi[1]), phi[2]

    @partial(jax.jit, static_argnums=0)
    def _policy(
        self, state: AgentState, obs: jax.Array, scalars: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        obs = jnp.expand_dims(obs, 0)
        scalars = jnp.expand_dims(scalars, 0)
        q, carry, _ = self._values(state, obs, scalars, carry=state.carry)
        pi = egreedy_probabilities(q, self.actions, state.hypers.epsilon)[0]
        return pi, carry

    @partial(jax.jit, static_argnums=0)
    def act(
        self, state: AgentState, obs: jax.Array, scalars: jax.Array
    ) -> tuple[AgentState, jax.Array]:
        pi, state.carry = self._policy(state, obs, scalars)
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

        updates, new_optim = optimizer.update(grad, state.optim, state.params)
        new_params = optax.apply_updates(state.params, updates)
        weight_change = jax.tree.reduce(
            lambda total, leaf: total + jnp.sum(jnp.abs(leaf)),
            updates,
            initializer=jnp.array(0.0, dtype=jnp.float32),
        )
        metrics["weight_change"] = weight_change

        return replace(state, params=new_params, optim=new_optim), metrics

    # Of shape: <batch, sequence, *feat>
    # TODO: Important, this assumes that no truncation happens, that is end() is properly called before calling start()
    def _loss(
        self, params: hk.Params, target: hk.Params, batch: Dict, _weights: jax.Array
    ):
        x = batch["x"][:, :-1]
        xp = batch["x"][:, 1:]
        a = batch["a"][:, :-1]
        r = batch["r"][:, :-1]
        g = batch["gamma"][:, :-1]
        # batch["carry"] is expected to have shape [B, T+1, *carry_shape], where carry at time t corresponds to the transition from x[t-1] to x[t]
        carry = self._slice_carry(batch["carry"], 0, -1)
        carryp = self._slice_carry(batch["carry"], 1, None)
        # reset[t] marks x[t] as an episode-start; reset_p[t] marks xp[t] (== x[t+1]) as one.
        reset = batch["reset"][:, :-1]
        reset_p = batch["reset"][:, 1:]

        scalars = batch["scalars"][:, :-1]
        scalars_p = batch["scalars"][:, 1:]

        # Perform burn-in
        if self.burn_in_steps > 0:
            b_x, x = jnp.hsplit(x, [self.burn_in_steps])
            b_xp, xp = jnp.hsplit(xp, [self.burn_in_steps])
            b_reset, reset = jnp.hsplit(reset, [self.burn_in_steps])
            b_reset_p, reset_p = jnp.hsplit(reset_p, [self.burn_in_steps])
            b_carry, carry = self._split_carry(carry, self.burn_in_steps)
            b_carryp, carryp = self._split_carry(carryp, self.burn_in_steps)
            b_scalars, scalars = jnp.hsplit(scalars, [self.burn_in_steps])
            b_scalars_p, scalars_p = jnp.hsplit(scalars_p, [self.burn_in_steps])
            _, a = jnp.hsplit(a, [self.burn_in_steps])
            _, r = jnp.hsplit(r, [self.burn_in_steps])
            _, g = jnp.hsplit(g, [self.burn_in_steps])

            carry = self._set_carry_start(
                carry,
                jax.lax.stop_gradient(
                    self._extract_carry(
                        self.phi(
                            params,
                            b_x,
                            scalars=b_scalars,
                            carry=b_carry,
                            reset=b_reset,
                            is_target=False,
                        )[1]
                    )
                ),
            )
            carryp = self._set_carry_start(
                carryp,
                jax.lax.stop_gradient(
                    self._extract_carry(
                        self.phi(
                            target,
                            b_xp,
                            scalars=b_scalars_p,
                            carry=b_carryp,
                            reset=b_reset_p,
                            is_target=True,
                        )[1]
                    )
                ),
            )

        phi = self.phi(
            params, x, scalars=scalars, carry=carry, reset=reset, is_target=False
        )[0]
        phi_p = self.phi(
            target, xp, scalars=scalars_p, carry=carryp, reset=reset_p, is_target=True
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

        loss = jnp.mean(losses)

        # aggregate metrics
        metrics = {
            "loss": loss,
            "abs_td_error": jnp.mean(jnp.abs(batch_metrics["delta"])),
            "squared_td_error": jnp.mean(jnp.square(batch_metrics["delta"])),
        }

        return loss, metrics

    @partial(jax.jit, static_argnums=0)
    def _start(self, state: AgentState, obs: Union[jax.Array, Dict[str, jax.Array]]):
        if isinstance(obs, Mapping):
            obs_img = obs["image"]
            hint = obs["hint"]
        else:
            obs_img = obs
            hint = None

        scalars = self.encode_scalar_features(
            jnp.int32(-1), jnp.float32(0), jnp.float32(0), hint
        )
        state, a = self.act(state, obs_img, scalars)
        carry_for_buffer = self._carry_for_buffer(state.carry)  # Reset within alg.
        state.last_timestep.update(
            {
                "x": obs_img,
                "a": a,
                "scalars": scalars,
                "carry": carry_for_buffer,
                "reset": jnp.bool(True),
            }
        )
        state = self._decay_epsilon(state)
        state = self._maybe_update_if_not_frozen(state)
        return state, a

    @partial(jax.jit, static_argnums=0)
    def _step(
        self,
        state: AgentState,
        reward: jax.Array,
        obs: Union[jax.Array, Dict[str, jax.Array]],
        extra: Dict[str, jax.Array],
    ):
        return self._step_impl(state, reward, obs, extra, True)

    @partial(jax.jit, static_argnums=(0, 5))
    def _step_impl(
        self,
        state: AgentState,
        reward: jax.Array,
        obs: Union[jax.Array, Dict[str, jax.Array]],
        extra: Dict[str, jax.Array],
        update: bool,
    ):
        if isinstance(obs, Mapping):
            obs_img = obs["image"]
            hint = obs["hint"]
        else:
            obs_img = obs
            hint = None
        gamma = extra.get("gamma", 1.0)

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

        state, unbiased_reward_trace = self._compute_reward_trace(state, reward)

        scalars = self.encode_scalar_features(
            state.last_timestep["a"], reward, unbiased_reward_trace, hint
        )
        last_carry = self._carry_for_buffer(state.carry)
        state, a = self.act(state, obs_img, scalars)

        state.last_timestep.update(
            {
                "x": obs_img,
                "a": a,
                "scalars": scalars,
                "carry": last_carry,
                "reset": jnp.bool(False),
            }
        )
        if update:
            state = self._maybe_update_if_not_frozen(state)
            state = self._decay_epsilon(state)
        return state, a

    @partial(jax.jit, static_argnums=0)
    def _step_without_update(
        self,
        state: AgentState,
        reward: jax.Array,
        obs: Union[jax.Array, Dict[str, jax.Array]],
        extra: Dict[str, jax.Array],
    ):
        return self._step_impl(state, reward, obs, extra, False)

    @partial(jax.jit, static_argnums=0)
    def _end(self, state, reward: jax.Array, extra: Dict[str, jax.Array]):
        return self._end_impl(state, reward, extra, True)

    @partial(jax.jit, static_argnums=(0, 4))
    def _end_impl(
        self,
        state: AgentState,
        reward: jax.Array,
        extra: Dict[str, jax.Array],
        update: bool,
    ):
        # possibly process the reward
        if self.reward_clip > 0:
            reward = jnp.clip(reward, -self.reward_clip, self.reward_clip)

        state.last_timestep.update({"r": reward, "gamma": jnp.float32(0)})
        batch_sequence = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (1, 1, *x.shape)), state.last_timestep
        )
        buffer_state = self.buffer.add(state.buffer_state, batch_sequence)
        state = replace(state, buffer_state=buffer_state)

        state, _ = self._compute_reward_trace(state, reward)

        if update:
            state = self._maybe_update_if_not_frozen(state)
            state = self._decay_epsilon(state)
        return state

    @partial(jax.jit, static_argnums=0)
    def _end_without_update(
        self, state: AgentState, reward: jax.Array, extra: Dict[str, jax.Array]
    ):
        return self._end_impl(state, reward, extra, False)
