from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import fields, replace
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import flashbax as fbx
import jax
import jax.numpy as jnp
import optax
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.BaseAgent import AgentState as BaseAgentState
from algorithms.BaseAgent import BaseAgent
from algorithms.BaseAgent import Hypers as BaseHypers
from optimizers import selective_weight_reinitialization
from representations.networks import NetworkBuilder
from utils.checkpoint import checkpointable
from utils.policies import egreedy_probabilities


@cxu.dataclass
class OptimizerHypers:
    learning_rate: float
    b1: float
    b2: float
    eps: float


@cxu.dataclass
class SWRHypers:
    reinit_freq: int
    reinit_factor: float
    decay_rate: float
    seed: int


@cxu.dataclass
class Metrics:
    weight_change: jax.Array
    abs_td_error: jax.Array
    squared_td_error: jax.Array
    loss: jax.Array


@cxu.dataclass
class Hypers(BaseHypers):
    epsilon: jax.Array
    optimizer: OptimizerHypers
    swr: Optional[SWRHypers]
    total_steps: int
    update_freq: int
    epsilon_linear_decay: Optional[float]
    initial_epsilon: Optional[float]
    final_epsilon: Optional[float]
    freeze_steps: float
    greedy_when_frozen: bool
    reward_trace_decay: float
    hint_trace_decay: float


@cxu.dataclass
class AgentState(BaseAgentState):
    params: Any
    optim: optax.OptState
    buffer_state: Any
    key: jax.Array
    last_timestep: Dict[str, jax.Array]
    steps: int
    updates: int
    hypers: Hypers
    metrics: Metrics
    reward_trace: jax.Array
    hint_trace: jax.Array


@checkpointable(("buffer", "steps", "state", "updates"))
class NNAgent(BaseAgent):
    # Env steps between `_periodic_step` firings; None = no periodic op.
    periodic_freq: Optional[int] = None

    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)

        # ------------------------------
        # -- Configuration Parameters --
        # ------------------------------
        self.rep_params: Dict = params["representation"]
        self.optimizer_params: Dict = params["optimizer"]

        total_steps = params["total_steps"]
        freeze_steps = params.get("freeze_steps", jnp.inf)
        epsilon_linear_decay = params.get("epsilon_linear_decay")
        initial_epsilon = params.get("initial_epsilon")
        final_epsilon = params.get("final_epsilon")
        greedy_when_frozen = params.get("greedy_when_frozen", False)
        if epsilon_linear_decay is not None:
            epsilon = initial_epsilon
        else:
            epsilon = params["epsilon"]

        assert epsilon is not None or (
            epsilon_linear_decay is not None
            and initial_epsilon is not None
            and final_epsilon is not None
        )
        self.reward_clip = params.get("reward_clip", 0)
        self.reward_scale = params.get("reward_scale")
        self.reward_trace_decay = params.get("reward_trace_decay", 0.9)
        self.hint_trace_decay = params.get("hint_trace_decay", 0.9)
        scalar_features_str = params.get("representation", {}).get(
            "scalar_features", "hint,last_action,last_reward"
        )
        self.scalar_features = scalar_features_str.split(",")

        self.scalars_size = 0
        self.hint_size = 0
        if "last_action" in self.scalar_features:
            self.scalars_size += actions
        if "last_reward" in self.scalar_features:
            self.scalars_size += 1
        if "reward_trace" in self.scalar_features:
            self.scalars_size += 1

        if isinstance(observations, Mapping):
            image_shape = observations["image"]
            if "hint" in observations and "hint" in self.scalar_features:
                self.hint_size = observations["hint"][0]
                self.scalars_size += self.hint_size
            if "hint" in observations and "hint_trace" in self.scalar_features:
                self.hint_size = self.hint_size or observations["hint"][0]
                self.scalars_size += observations["hint"][0]
        else:
            image_shape = observations

        self.rep_params["scalars"] = self.scalars_size
        self.rep_params["hint_size"] = self.hint_size
        self.hidden_size = self.rep_params["hidden"]

        # ---------------------
        # -- NN Architecture --
        # ---------------------
        self.builder = NetworkBuilder(image_shape, self.rep_params, self.key)
        self._build_heads(self.builder)
        self.phi = self.get_feature_function(self.builder)
        net_params = self.builder.getParams()

        # ---------------
        # -- Optimizer --
        # ---------------
        optimizer_hypers = OptimizerHypers(
            learning_rate=self.optimizer_params["alpha"],
            b1=self.optimizer_params["beta1"],
            b2=self.optimizer_params["beta2"],
            eps=self.optimizer_params["eps"],
        )

        # Check for SWR configuration
        swr_params = params.get("swr")
        swr_hypers = None
        if swr_params is not None:
            self.swr_utility_function = swr_params["utility_function"]
            self.swr_pruning_method = swr_params["pruning_method"]
            swr_hypers = SWRHypers(
                reinit_freq=swr_params["reinit_freq"],
                reinit_factor=swr_params["reinit_factor"],
                decay_rate=swr_params.get("decay_rate", 0.0),
                seed=seed,
            )
        self.initializers = self.builder.getInitializers()
        optimizer = self._build_optimizer(optimizer_hypers, swr_hypers)
        opt_state = optimizer.init(net_params)

        # ------------------
        # -- Data ingress --
        # ------------------
        self.buffer_size = params["buffer_size"]
        self.batch_size = params["batch"]
        self.sequence_length = params.get("sequence_length", 1)
        self.buffer_min_size = params.get("buffer_min_size", self.batch_size)
        if self.buffer_min_size == "buffer_size":
            self.buffer_min_size = self.buffer_size
        elif self.buffer_min_size == "batch_size":
            self.buffer_min_size = self.batch_size
        elif self.buffer_min_size == "batch_size*sequence_length":
            self.buffer_min_size = self.batch_size * self.sequence_length
        self.priority_exponent = params.get("priority_exponent", 0.0)

        buffer = fbx.make_prioritised_trajectory_buffer(
            max_length_time_axis=self.buffer_size,
            min_length_time_axis=self.buffer_min_size,
            sample_batch_size=self.batch_size,
            add_batch_size=1,
            sample_sequence_length=self.n_step + self.sequence_length,
            period=1,
            priority_exponent=self.priority_exponent,
        )
        self.buffer = replace(
            buffer,
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=(0,)),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
            set_priorities=jax.jit(buffer.set_priorities, donate_argnums=(0,)),
        )

        dummy_hint = None
        dummy_hint_trace = None
        if (
            isinstance(observations, Mapping)
            and "hint" in observations
            and "hint" in self.scalar_features
        ):
            dummy_hint = jnp.zeros(observations["hint"])
        if (
            isinstance(observations, Mapping)
            and "hint" in observations
            and "hint_trace" in self.scalar_features
        ):
            dummy_hint_trace = jnp.zeros(observations["hint"])

        dummy_timestep = {
            "x": jnp.zeros(image_shape),
            "scalars": self.encode_scalar_features(
                jnp.int32(0),
                jnp.float32(0),
                jnp.float32(0),
                dummy_hint,
                dummy_hint_trace,
            ),
            "a": jnp.int32(0),
            "r": jnp.float32(0),
            "gamma": jnp.float32(0),
        }
        buffer_state = self.buffer.init(dummy_timestep)

        # --------------------------
        # -- Stateful information --
        # --------------------------
        update_freq = params.get("update_freq", 1)
        hypers = Hypers(
            **self.state.hypers.__dict__,
            epsilon=epsilon,
            optimizer=optimizer_hypers,
            swr=swr_hypers,
            total_steps=total_steps,
            update_freq=update_freq,
            epsilon_linear_decay=epsilon_linear_decay,
            initial_epsilon=initial_epsilon,
            final_epsilon=final_epsilon,
            freeze_steps=freeze_steps,
            greedy_when_frozen=greedy_when_frozen,
            reward_trace_decay=self.reward_trace_decay,
            hint_trace_decay=self.hint_trace_decay,
        )
        self.state = AgentState(
            **{k: v for k, v in self.state.__dict__.items() if k != "hypers"},
            params=net_params,
            optim=opt_state,
            buffer_state=buffer_state,
            key=self.key,
            last_timestep=dummy_timestep,
            steps=0,
            updates=0,
            hypers=hypers,
            metrics=Metrics(
                weight_change=jnp.float32(0.0),
                abs_td_error=jnp.float32(0.0),
                squared_td_error=jnp.float32(0.0),
                loss=jnp.float32(0.0),
            ),
            reward_trace=jnp.float32(0.0),
            hint_trace=jnp.zeros(max(self.hint_size, 1), dtype=jnp.float32),
        )

    def get_feature_function(self, builder: NetworkBuilder):
        return builder.getFeatureFunction()

    def _build_optimizer(
        self,
        optimizer_hypers: OptimizerHypers,
        swr_hypers: Optional[SWRHypers],
    ) -> optax.GradientTransformationExtraArgs:
        """Build optimizer with optional SWR."""

        # Start with Adam optimizer
        optimizer = optax.adam(**optimizer_hypers.__dict__)

        # If SWR is configured, chain it after Adam
        if swr_hypers is not None:
            swr_optimizer = selective_weight_reinitialization(
                utility_function=self.swr_utility_function,
                pruning_method=self.swr_pruning_method,
                initializers=self.initializers,
                reinit_freq=swr_hypers.reinit_freq,
                reinit_factor=swr_hypers.reinit_factor,
                decay_rate=swr_hypers.decay_rate,
                seed=swr_hypers.seed,
            )

            # Chain Adam and SWR
            optimizer = optax.chain(optimizer, swr_optimizer)

        return optimizer

    @partial(jax.jit, static_argnums=0)
    def _compute_reward_trace(
        self, state: AgentState, reward: jax.Array
    ) -> Tuple[AgentState, jax.Array]:
        beta = state.hypers.reward_trace_decay
        new_reward_trace = beta * state.reward_trace + (1 - beta) * reward
        correction = 1.0 - jnp.power(beta, state.steps.astype(jnp.float32))
        unbiased_reward_trace = new_reward_trace / correction
        state = replace(state, reward_trace=new_reward_trace)
        return state, unbiased_reward_trace

    @partial(jax.jit, static_argnums=0)
    def _compute_hint_trace(
        self, state: AgentState, hint: jax.Array
    ) -> Tuple[AgentState, jax.Array]:
        beta = state.hypers.hint_trace_decay
        new_hint_trace = beta * state.hint_trace + (1 - beta) * hint
        correction = 1.0 - jnp.power(beta, state.steps.astype(jnp.float32))
        unbiased_hint_trace = new_hint_trace / correction
        state = replace(state, hint_trace=new_hint_trace)
        return state, unbiased_hint_trace

    @partial(jax.jit, static_argnums=0)
    def encode_scalar_features(
        self,
        last_action: jax.Array,
        last_reward: jax.Array,
        reward_trace: jax.Array,
        hint: Optional[jax.Array] = None,
        hint_trace: Optional[jax.Array] = None,
    ) -> jax.Array:
        parts = []
        if "last_action" in self.scalar_features:
            parts.append(jax.nn.one_hot(last_action, self.actions))
        if "last_reward" in self.scalar_features:
            parts.append(jnp.atleast_1d(last_reward))
        if "reward_trace" in self.scalar_features:
            parts.append(jnp.atleast_1d(reward_trace))
        if "hint" in self.scalar_features and hint is not None:
            parts.append(jnp.atleast_1d(hint))
        if "hint_trace" in self.scalar_features and hint_trace is not None:
            parts.append(jnp.atleast_1d(hint_trace))

        if len(parts) == 0:
            return jnp.zeros((0,))
        return jnp.concatenate(parts, axis=-1)

    # ------------------------
    # -- NN agent interface --
    # ------------------------

    @abstractmethod
    def _build_heads(self, builder: NetworkBuilder) -> None: ...

    @abstractmethod
    def _values(self, state: AgentState, x: jax.Array, z: jax.Array) -> jax.Array: ...

    @abstractmethod
    def _update(self, state: AgentState) -> Tuple[AgentState, Dict[str, jax.Array]]: ...

    def _update_state_with_metrics(self, state: AgentState) -> AgentState:
        new_state, updates = self._update(state)
        field_names = {f.name for f in fields(state.metrics)}
        applicable = {k: v for k, v in updates.items() if k in field_names}
        return replace(new_state, metrics=replace(state.metrics, **applicable))

    def _update_state_with_metrics_if_can_sample(self, state: AgentState) -> AgentState:
        return jax.lax.cond(
            self.buffer.can_sample(state.buffer_state),
            self._update_state_with_metrics,
            lambda s: s,
            state,
        )

    def _periodic_step(self, state: AgentState) -> AgentState:
        # Override here rather than gating with a per-step `jax.lax.cond`:
        # under vmap, cond is rewritten to select and both branches execute
        # every step. The training loop dispatches this from an outer scan.
        return state

    def _advance_update_clock(self, state: AgentState) -> AgentState:
        state = replace(state, steps=state.steps + 1)
        return self._decay_epsilon(state)

    @partial(jax.jit, static_argnums=0)
    def _maybe_update_if_not_frozen(self, state: AgentState) -> AgentState:
        return jax.lax.cond(
            state.steps < state.hypers.freeze_steps,
            self._maybe_update,
            lambda s: s,
            state,
        )

    @partial(jax.jit, static_argnums=0)
    def _maybe_update(self, state: AgentState) -> AgentState:
        state = jax.lax.cond(
            (state.steps % state.hypers.update_freq == 0)
            & self.buffer.can_sample(state.buffer_state),
            self._update_state_with_metrics,
            lambda s: s,
            state,
        )
        return self._advance_update_clock(state)

    @partial(jax.jit, static_argnums=0)
    def _decay_epsilon(self, state: AgentState):
        epsilon = state.hypers.epsilon
        if state.hypers.epsilon_linear_decay is not None:
            assert state.hypers.initial_epsilon is not None
            assert state.hypers.final_epsilon is not None
            decay_steps = state.hypers.epsilon_linear_decay
            progress = state.steps / decay_steps
            calculated_epsilon = (
                state.hypers.initial_epsilon
                + (state.hypers.final_epsilon - state.hypers.initial_epsilon) * progress
            )
            epsilon = jnp.maximum(calculated_epsilon, state.hypers.final_epsilon)

        hypers = replace(state.hypers, epsilon=epsilon)
        return replace(state, hypers=hypers)

    @partial(jax.jit, static_argnums=0)
    def _policy(
        self, state: AgentState, obs: jax.Array, scalars: jax.Array
    ) -> jax.Array:
        obs = jnp.expand_dims(obs, 0)
        z = jnp.expand_dims(scalars, 0)
        q = self._values(state, obs, z)[0]
        epsilon = jax.lax.cond(
            state.hypers.greedy_when_frozen
            & (state.steps >= state.hypers.freeze_steps),
            lambda: jnp.array(0.0),
            lambda: state.hypers.epsilon,
        )
        pi = egreedy_probabilities(q, self.actions, epsilon)
        return pi

    @partial(jax.jit, static_argnums=0)
    def act(
        self,
        state: AgentState,
        obs: jax.Array,
        scalars: jax.Array,
    ) -> tuple[AgentState, jax.Array]:
        pi = self._policy(state, obs, scalars)
        state.key, sample_key = jax.random.split(state.key)
        a = jax.random.choice(sample_key, self.actions, p=pi)
        return state, a

    # --------------------------
    # -- Base agent interface --
    # --------------------------
    def values(self, x: jax.Array):
        # if x is a vector, then jax handles a lack of "batch" dimension gracefully
        #   at a 5x speedup
        # if x is a tensor, jax does not handle lack of "batch" dim gracefully
        if len(x.shape) > 1:
            x = jnp.expand_dims(x, 0)
            q = self._values(self.state, x)[0]

        else:
            q = self._values(self.state, x)

        return q

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, obs: Union[jax.Array, Dict[str, jax.Array]]):
        self.state, a = self._start(self.state, obs)
        return a

    @partial(jax.jit, static_argnums=0)
    def _start(self, state: AgentState, obs: Union[jax.Array, Dict[str, jax.Array]]):
        if isinstance(obs, Mapping):
            obs_img = obs["image"]
            hint = obs["hint"]
        else:
            obs_img = obs
            hint = None

        scalars = self.encode_scalar_features(
            jnp.int32(-1),
            jnp.float32(0),
            jnp.float32(0),
            hint,
            jnp.zeros_like(state.hint_trace) if hint is not None else None,
        )
        state, a = self.act(state, obs_img, scalars)
        state.last_timestep.update(
            {
                "x": obs_img,
                "a": a,
                "scalars": scalars,
            }
        )
        state = self._maybe_update_if_not_frozen(state)
        return state, a

    def step(
        self,
        reward: jax.Array,
        obs: Union[jax.Array, Dict[str, jax.Array]],
        extra: Dict[str, jax.Array],
    ):
        self.state, a = self._step(self.state, reward, obs, extra)
        return a

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

        state, unbiased_reward_trace = self._compute_reward_trace(state, reward)

        unbiased_hint_trace = None
        if hint is not None:
            state, unbiased_hint_trace = self._compute_hint_trace(state, hint)

        scalars = self.encode_scalar_features(
            state.last_timestep["a"],
            reward,
            unbiased_reward_trace,
            hint,
            unbiased_hint_trace,
        )
        state, a = self.act(state, obs_img, scalars)

        gamma = extra.get("gamma", 1.0)

        if self.reward_clip > 0:
            reward = jnp.clip(reward, -self.reward_clip, self.reward_clip)
        if self.reward_scale is not None:
            reward = reward / self.reward_scale

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
                "x": obs_img,
                "scalars": scalars,
                "a": a,
            }
        )
        if update:
            state = self._maybe_update_if_not_frozen(state)
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

    def end(self, reward: jax.Array, extra: Dict[str, jax.Array]):
        self.state = self._end(self.state, reward, extra)

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
        if self.reward_scale is not None:
            reward = reward / self.reward_scale

        state, _ = self._compute_reward_trace(state, reward)

        state.last_timestep.update(
            {
                "r": reward,
                "gamma": jnp.float32(0),
            }
        )
        batch_sequence = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (1, 1, *x.shape)), state.last_timestep
        )
        buffer_state = self.buffer.add(state.buffer_state, batch_sequence)
        state = replace(state, buffer_state=buffer_state)
        if update:
            state = self._maybe_update_if_not_frozen(state)
        return state

    @partial(jax.jit, static_argnums=0)
    def _end_without_update(
        self, state: AgentState, reward: jax.Array, extra: Dict[str, jax.Array]
    ):
        return self._end_impl(state, reward, extra, False)
