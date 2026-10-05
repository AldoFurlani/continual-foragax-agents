"""
PT-DQN: Permanent-Transient DQN (Separate Networks)

From "Prediction and Control in Continual Reinforcement Learning"
(Anand & Precup, NeurIPS 2023)

Algorithm 4: PT-DQN Pseudocode (Continual Reinforcement Learning)

Decomposes Q-function: Q^(PT)(s,a) = Q_P(s,a) + Q_T(s,a)
  - Permanent network (Q_P): slowly accumulates general knowledge via SGD
  - Transient network (Q_T): quickly adapts via Adam, periodically decayed by λ

Every k gradient updates:
  1. Update permanent network via U = len(PM_buffer) // batch_size gradient
     steps of regression towards Q^(PT), using a separate PM replay buffer
     storing (s, a, old_Q_P(s,a))
  2. Decay ALL transient network weights by λ

Architecture (matching reference github.com/NishanthVAnand):
  - T_Net: Independent transient backbone + head (state.params)
  - P_Net: Independent permanent backbone + head (state.perm_params)
  - Target_net: Copy of T_Net only (state.target_params)
  - Two replay buffers: main buffer for TD updates, PM buffer for permanent
    updates storing historical Q_P values at collection time.
  - Default permanent optimizer is SGD; configurable via pt_optimizer.name.
  - MSE loss for both transient and permanent updates (matching reference).
"""

from dataclasses import replace
from functools import partial
from typing import Any, Dict, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.nn.DQN import DQN, q_loss
from algorithms.nn.DQN import AgentState as DQNAgentState
from algorithms.nn.DQN import Hypers as DQNHypers
from algorithms.nn.NNAgent import OptimizerHypers
from representations.networks import NetworkBuilder


# ---------------------------------------------------------------------------
# MSE-based q_loss matching the reference PT-DQN (nn.MSELoss)
# ---------------------------------------------------------------------------
def mse_q_loss(q, a, r, gamma, qp):
    vp = qp.max()
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)
    delta = target - q[a]
    return optax.squared_error(q[a], target), {"delta": delta}


@cxu.dataclass
class Hypers(DQNHypers):
    pt_update_freq: int  # k: gradient updates between permanent updates
    pt_decay: float  # λ: decay factor for transient weights
    pt_optimizer: OptimizerHypers  # optimizer config for permanent network
    pm_buffer_size: int  # size of PM replay buffer (reference: 10000)


@cxu.dataclass
class PMBufferState:
    """Simple circular buffer for PM data, supporting sequential iteration.

    Reference uses itertools.islice for sequential (no-replacement) reads:
      curr_batch = list(islice(memory, i*batch, (i+1)*batch))
    """

    x: jax.Array  # (max_size, *obs_shape)
    scalars: jax.Array  # (max_size, *scalar_shape)
    a: jax.Array  # (max_size,)
    q_p_a: jax.Array  # (max_size,)
    write_index: jnp.int32
    size: jnp.int32


@cxu.dataclass
class AgentState(DQNAgentState):
    perm_params: Any  # permanent network params {"phi": ..., "q": ...}
    pt_optim: Any  # optimizer state for permanent network
    pm_buffer_state: PMBufferState  # PM circular buffer state
    hypers: Hypers


class PT_DQN(DQN):
    def __init__(
        self,
        observations: Tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)

        # -----------------------------------------------
        # -- Permanent network (separate backbone+head) --
        # -----------------------------------------------
        self.key, perm_key = jax.random.split(self.key)
        image_shape = self.builder._input_shape
        perm_builder = NetworkBuilder(image_shape, self.rep_params, perm_key)

        def q_net_builder():
            return hk.Linear(self.actions, name="q")

        self.perm_q_net, _, self.perm_q = perm_builder.addHead(q_net_builder, name="q")
        self.perm_phi = perm_builder.getFeatureFunction()
        perm_params = perm_builder.getParams()

        # ----------------------
        # -- PT optimizer     --
        # ----------------------
        pt_optimizer_params = params.get("pt_optimizer", {})
        pt_optimizer_name = pt_optimizer_params.get("name", "SGD").upper()
        self._pt_optimizer_is_sgd = pt_optimizer_name == "SGD"

        transient_alpha = self.optimizer_params["alpha"]
        pt_alpha_ratio = pt_optimizer_params.get("alpha_ratio", 1.0)
        pt_optimizer_hypers = OptimizerHypers(
            learning_rate=transient_alpha * pt_alpha_ratio,
            b1=pt_optimizer_params.get("beta1", 0.9),
            b2=pt_optimizer_params.get("beta2", 0.999),
            eps=pt_optimizer_params.get("eps", 1e-8),
        )
        pt_optimizer = self._build_pt_optimizer(pt_optimizer_hypers)
        pt_optim = pt_optimizer.init(perm_params)

        # -------------------------------------------------------
        # -- PM replay buffer (simple circular, sequential read) --
        # -------------------------------------------------------
        pm_buffer_size = params.get("pm_buffer_size", 10000)
        self.pm_buffer_size = pm_buffer_size
        pm_buffer_state = PMBufferState(
            x=jnp.zeros((pm_buffer_size, *self.state.last_timestep["x"].shape)),
            scalars=jnp.zeros(
                (pm_buffer_size, *self.state.last_timestep["scalars"].shape)
            ),
            a=jnp.zeros(pm_buffer_size, dtype=jnp.int32),
            q_p_a=jnp.zeros(pm_buffer_size, dtype=jnp.float32),
            write_index=jnp.int32(0),
            size=jnp.int32(0),
        )

        # ----------------------
        # -- Hypers + state   --
        # ----------------------
        hypers = Hypers(
            **self.state.hypers.__dict__,
            pt_update_freq=params["pt_update_freq"],
            pt_decay=params["pt_decay"],
            pt_optimizer=pt_optimizer_hypers,
            pm_buffer_size=pm_buffer_size,
        )

        self.state = AgentState(
            **{k: v for k, v in self.state.__dict__.items() if k != "hypers"},
            perm_params=perm_params,
            pt_optim=pt_optim,
            pm_buffer_state=pm_buffer_state,
            hypers=hypers,
        )

        # pt_update_freq is in agent updates; periodic_freq is in env steps.
        self.periodic_freq = int(params["pt_update_freq"]) * int(
            self.state.hypers.update_freq
        )

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        """Build transient Q-head only. Permanent network built in __init__."""

        def q_net_builder():
            return hk.Linear(self.actions, name="q")

        self.q_net, _, self.q = builder.addHead(q_net_builder, name="q")

    def _build_pt_optimizer(
        self,
        pt_optimizer_hypers: OptimizerHypers,
    ) -> optax.GradientTransformation:
        """Build optimizer for permanent network updates.

        Default: SGD (matching reference implementation).
        Set pt_optimizer.name = "ADAM" in config for Adam.
        """
        if self._pt_optimizer_is_sgd:
            return optax.sgd(pt_optimizer_hypers.learning_rate)
        return optax.adam(**pt_optimizer_hypers.__dict__)

    # ---------------------------
    # -- Values: Q_T + Q_P     --
    # ---------------------------
    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array, scalars: jax.Array):
        """Return combined Q^(PT) = Q_T + Q_P from separate networks."""
        phi_t = self.phi(state.params, x, scalars=scalars).out
        q_t = self.q(state.params, phi_t)
        phi_p = self.perm_phi(state.perm_params, x, scalars=scalars).out
        q_p = self.perm_q(state.perm_params, phi_p)
        return q_t + q_p

    # ---------------------------
    # -- PM buffer management  --
    # ---------------------------
    @partial(jax.jit, static_argnums=0)
    def _add_to_pm_buffer(self, state: AgentState) -> AgentState:
        """Compute Q_P(s,a) for current last_timestep and store in PM buffer.

        Reference: exp_replay_PM.store(cs, c_action, val_p)
        """
        x = state.last_timestep["x"]
        scalars = state.last_timestep["scalars"]
        a = state.last_timestep["a"]

        # Compute Q_P(s, a) with current permanent network
        phi_p = self.perm_phi(
            state.perm_params,
            jnp.expand_dims(x, 0),
            scalars=jnp.expand_dims(scalars, 0),
        ).out
        q_p = self.perm_q(state.perm_params, phi_p)[0]
        q_p_a = q_p[a]

        pm = state.pm_buffer_state
        idx = pm.write_index
        new_pm = PMBufferState(
            x=pm.x.at[idx].set(x),
            scalars=pm.scalars.at[idx].set(scalars),
            a=pm.a.at[idx].set(a),
            q_p_a=pm.q_p_a.at[idx].set(q_p_a),
            write_index=(idx + 1) % self.pm_buffer_size,
            size=jnp.minimum(pm.size + 1, self.pm_buffer_size),
        )
        return replace(state, pm_buffer_state=new_pm)

    # ------------------------------------------
    # -- Override _step_impl/_end_impl for PM buffer add --
    # ------------------------------------------
    @partial(jax.jit, static_argnums=(0, 5))
    def _step_impl(
        self,
        state: AgentState,
        reward: jax.Array,
        obs: Union[jax.Array, Dict[str, jax.Array]],
        extra: Dict[str, jax.Array],
        update: bool,
    ):
        # Add (s, a, Q_P(s,a)) to PM buffer before main buffer add + update
        state = self._add_to_pm_buffer(state)
        return super()._step_impl(state, reward, obs, extra, update)

    @partial(jax.jit, static_argnums=(0, 4))
    def _end_impl(
        self,
        state: AgentState,
        reward: jax.Array,
        extra: Dict[str, jax.Array],
        update: bool,
    ):
        state = self._add_to_pm_buffer(state)
        return super()._end_impl(state, reward, extra, update)

    # ---------------------------
    # -- Update logic          --
    # ---------------------------
    @partial(jax.jit, static_argnums=0)
    def _update(self, state: AgentState):
        updates = state.updates + 1

        # --- Transient network TD update ---
        state.key, buffer_sample_key = jax.random.split(state.key)
        batch = self.buffer.sample(state.buffer_state, buffer_sample_key)

        state, metrics = self._computeUpdate(state, batch.experience)

        state = replace(state, updates=updates)

        target_params = self._update_target_network(state, updates)
        state = replace(state, target_params=target_params)

        return state, metrics

    @partial(jax.jit, static_argnums=0)
    def _periodic_step(self, state: AgentState) -> AgentState:
        state = self._pt_gradient_update(state)
        state = replace(
            state,
            params=jax.tree_util.tree_map(
                lambda p: state.hypers.pt_decay * p, state.params
            ),
        )
        return state

    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Dict):
        """Override to pass perm_params to _loss."""
        grad_fn = jax.grad(self._loss, has_aux=True)
        grad, metrics = grad_fn(
            state.params, state.target_params, state.perm_params, batch
        )
        optimizer = self._build_optimizer(state.hypers.optimizer, state.hypers.swr)

        updates, new_optim = optimizer.update(
            grad, state.optim, state.params, grad=grad
        )
        new_params = optax.apply_updates(state.params, updates)
        flat_updates, _ = ravel_pytree(updates)
        weight_change = jnp.linalg.norm(flat_updates, ord=1)
        metrics["weight_change"] = weight_change

        return replace(state, params=new_params, optim=new_optim), metrics

    # -------------------------------------------------------
    # -- PT update: U gradient steps on PM buffer + decay  --
    # -------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def _pt_gradient_update(self, state: AgentState) -> AgentState:
        """Update permanent network via U sequential gradient steps on PM buffer.

        Reference train_P_Net(): iterates sequentially over PM buffer:
          u_steps = (exp_replay_PM.size() // batch_size) - 1
          for p_update in range(u_steps):
            curr_batch = islice(memory, p*batch, (p+1)*batch)
            ...
        Decay is handled separately in _update() to match reference ordering.
        """
        # Compute U from current PM buffer occupancy (matching reference)
        # Reference: u_steps = (exp_replay_PM.size() // args.batch_size) - 1
        pm_size = state.pm_buffer_state.size
        pt_update_steps = (pm_size // self.batch_size) - 1

        pt_optimizer = self._build_pt_optimizer(state.hypers.pt_optimizer)
        pm = state.pm_buffer_state

        # Roll arrays so index 0 = oldest element (matching deque iteration).
        # When full, write_index points to the oldest slot; when not full,
        # data starts at 0 already. Reference: islice reads oldest → newest.
        oldest = jax.lax.cond(
            pm.size >= self.pm_buffer_size,
            lambda: pm.write_index,
            lambda: jnp.int32(0),
        )
        rolled_x = jnp.roll(pm.x, -oldest, axis=0)
        rolled_scalars = jnp.roll(pm.scalars, -oldest, axis=0)
        rolled_a = jnp.roll(pm.a, -oldest, axis=0)
        rolled_q_p_a = jnp.roll(pm.q_p_a, -oldest, axis=0)

        def body_fn(i, carry):
            perm_params, pt_optim = carry
            # Sequential slice: [i*batch_size : (i+1)*batch_size]
            # Reference: islice(memory, p*batch, (p+1)*batch)
            start = i * self.batch_size
            batch = {
                "x": jax.lax.dynamic_slice_in_dim(
                    rolled_x, start, self.batch_size, axis=0
                ),
                "scalars": jax.lax.dynamic_slice_in_dim(
                    rolled_scalars, start, self.batch_size, axis=0
                ),
                "a": jax.lax.dynamic_slice_in_dim(
                    rolled_a, start, self.batch_size, axis=0
                ),
                "q_p_a": jax.lax.dynamic_slice_in_dim(
                    rolled_q_p_a, start, self.batch_size, axis=0
                ),
            }

            grad_fn = jax.grad(self._permanent_loss, has_aux=True)
            perm_grad, _ = grad_fn(perm_params, state.params, batch)

            perm_updates, new_pt_optim = pt_optimizer.update(
                perm_grad, pt_optim, perm_params
            )
            new_perm_params = optax.apply_updates(perm_params, perm_updates)
            return (new_perm_params, new_pt_optim)

        init_carry = (state.perm_params, state.pt_optim)
        new_perm_params, new_pt_optim = jax.lax.fori_loop(
            0, pt_update_steps, body_fn, init_carry
        )

        return replace(
            state,
            perm_params=new_perm_params,
            pt_optim=new_pt_optim,
        )

    # -------------------------------------------------------
    # -- Permanent loss: MSE regression towards Q_T + old_Q_P
    # -------------------------------------------------------
    def _permanent_loss(self, perm_params: Any, transient_params: Any, batch: Dict):
        """
        Permanent network loss using PM buffer.

        Reference train_P_Net():
          T_pred = T_Net(states)[actions]          # no_grad
          P_pred = P_Net(states)[actions]          # with grad
          loss = MSE(P_pred, T_pred + old_p_vals)  # old_p_vals from PM buffer

        The gradient flows through the entire permanent network (backbone + head).
        transient_params is the current T_Net weights (frozen during U steps).
        """
        x = batch["x"]
        a = batch["a"]
        scalars = batch["scalars"]
        old_q_p_a = batch["q_p_a"]

        # Q_T(s, a) from current transient network (no gradient)
        phi_t = self.phi(transient_params, x, scalars=scalars).out
        q_t = self.q(transient_params, phi_t)
        q_t_a = q_t[jnp.arange(a.shape[0]), a]
        q_t_a = jax.lax.stop_gradient(q_t_a)

        # Target = Q_T(s, a) + old_Q_P(s, a)   [both no gradient]
        target = jax.lax.stop_gradient(q_t_a + old_q_p_a)

        # Q_P(s, a) from permanent network (with gradient through all of P_Net)
        phi_p = self.perm_phi(perm_params, x, scalars=scalars).out
        q_p = self.perm_q(perm_params, phi_p)
        q_p_a = q_p[jnp.arange(a.shape[0]), a]

        # MSE loss (matching reference P_criterion = nn.MSELoss())
        loss = jnp.mean(optax.squared_error(q_p_a, target))

        return loss, {"pt_loss": loss}

    # -----------------------
    # -- Transient loss    --
    # -----------------------
    def _loss(  # type: ignore[override]
        self,
        params: hk.Params,
        target: hk.Params,
        perm_params: hk.Params,
        batch: Dict,
    ):
        """
        Transient TD loss using combined Q^(PT) from separate networks.

        Reference train_T_Net():
          T_pred = T_Net(s)[a]                       # with grad
          P_pred = P_Net(s)[a]                        # no_grad
          P_next = P_Net(s').max()                    # no_grad
          T_next = Target_net(s').max()               # no_grad
          targets = r + γ * (P_next + T_next)
          loss = MSE(T_pred + P_pred, targets)

        Online: Q_T(params) + sg(Q_P(perm_params))
        Target: Q_T(target) + sg(Q_P(perm_params))
        """
        x = batch["x"][:, 0]
        xp = batch["x"][:, -1]
        a = batch["a"][:, 0]
        scalars = batch["scalars"][:, 0]
        scalars_p = batch["scalars"][:, -1]

        rs = batch["r"]
        gs = batch["gamma"]
        gs = jnp.concatenate([jnp.ones((gs.shape[0], 1)), gs[:, :-1]], axis=1)
        gs = jnp.cumprod(gs, axis=1)

        r = jnp.sum(rs[:, :-1] * gs[:, :-1], axis=1)
        g = gs[:, -1]

        # Online: Q_T(s; params) + sg(Q_P(s; perm_params))
        phi_t = self.phi(params, x, scalars=scalars).out
        q_t = self.q(params, phi_t)
        phi_p = self.perm_phi(perm_params, x, scalars=scalars).out
        q_p = jax.lax.stop_gradient(self.perm_q(perm_params, phi_p))
        qs = q_t + q_p

        # Target: sg(Q_T(s'; target)) + sg(Q_P(s'; perm_params))
        phi_t_p = self.phi(target, xp, scalars=scalars_p).out
        q_t_p = jax.lax.stop_gradient(self.q(target, phi_t_p))
        phi_p_p = self.perm_phi(perm_params, xp, scalars=scalars_p).out
        q_p_p = jax.lax.stop_gradient(self.perm_q(perm_params, phi_p_p))
        qsp = q_t_p + q_p_p

        # Huber loss (consistent with DQN base class)
        batch_loss = jax.vmap(q_loss, in_axes=0)
        losses, batch_metrics = batch_loss(qs, a, r, g, qsp)

        loss = jnp.mean(losses)

        metrics = {
            "loss": loss,
            "abs_td_error": jnp.mean(jnp.abs(batch_metrics["delta"])),
            "squared_td_error": jnp.mean(jnp.square(batch_metrics["delta"])),
        }

        return loss, metrics
