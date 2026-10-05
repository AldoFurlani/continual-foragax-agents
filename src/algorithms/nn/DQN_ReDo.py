from dataclasses import replace
from functools import partial
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from ml_instrumentation.Collector import Collector
import optax

import utils.chex as cxu
from algorithms.nn.DQN import DQN
from algorithms.nn.DQN import AgentState as BaseAgentState
from algorithms.nn.DQN import Hypers as BaseHypers
from algorithms.nn.NNAgent import Metrics as BaseMetrics


@cxu.dataclass
class Hypers(BaseHypers):
    redo_freq: int
    redo_threshold: float


@cxu.dataclass
class Metrics(BaseMetrics):
    dormant_neurons: jax.Array


@cxu.dataclass
class AgentState(BaseAgentState):
    hypers: Hypers  # type: ignore
    metrics: Metrics  # type: ignore


def reset_momentum(momentum: jax.Array, mask: jax.Array) -> jax.Array:
    return jnp.where(mask, jnp.zeros_like(momentum), momentum)


class DQN_ReDo(DQN):
    def __init__(
        self,
        observations: Tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)

        rep_type = self.rep_params["type"]
        if rep_type != "ForagerNet":
            raise NotImplementedError(
                f"DQN_ReDo only supports ForagerNet. Got {rep_type!r}."
            )
        if self.rep_params.get("conv", "Conv2D") != "None":
            raise NotImplementedError(
                f"DQN_ReDo on ForagerNet only supports conv='None'; "
                f"got conv={self.rep_params.get('conv')!r}."
            )

        self._pre_core_layers = int(self.rep_params.get("pre_core_layers", 0))
        self._core_layers = int(self.rep_params.get("core_layers", 0))
        self._post_core_layers = int(self.rep_params.get("post_core_layers", 0))
        if self.rep_params.get("layers") is not None:
            raise NotImplementedError(
                "DQN_ReDo no longer supports the legacy `layers` knob; "
                "use pre_core_layers/core_layers/post_core_layers instead."
            )
        if (self._pre_core_layers + self._core_layers + self._post_core_layers) < 1:
            raise NotImplementedError(
                "DQN_ReDo on ForagerNet expects at least one of "
                "pre_core_layers, core_layers, post_core_layers >= 1; "
                f"got pre={self._pre_core_layers}, core={self._core_layers}, "
                f"post={self._post_core_layers}."
            )
        if params.get("swr") is not None:
            raise NotImplementedError("DQN_ReDo does not support combining with SWR.")

        self._reset_ln = bool(params.get("redo_reset_layernorm", True))
        self._use_ln = bool(self.rep_params.get("use_layernorm", False))
        self._preact_ln = bool(self.rep_params.get("preactivation_layer_norm", True))
        # By default we probe dormancy at the ReLU output. In post-act LN the
        # actual signal that reaches the next layer is post-LN, so the user can
        # opt to score there instead via redo_score_after_ln.
        self._score_after_ln = bool(params.get("redo_score_after_ln", False))
        if self._score_after_ln and not (self._use_ln and not self._preact_ln):
            raise NotImplementedError(
                "redo_score_after_ln=true is only meaningful for post-activation "
                "LayerNorm. Got use_layernorm="
                f"{self._use_ln}, preactivation_layer_norm={self._preact_ln}."
            )
        # Observe-only mode (redo_apply=false): compute dormant_neurons but skip
        # the param/optim recycle. Lets us instrument vanilla DQN with the same
        # logging path so the comparison baseline shares this agent's encoder.
        self._redo_apply = bool(params.get("redo_apply", True))

        self._stage_plan: List[Dict[str, object]] = []
        linear_idx = 0
        ln_idx = 0

        def linear_name(i: int) -> str:
            return "linear" if i == 0 else f"linear_{i}"

        def ln_name(i: int) -> str:
            return "layer_norm" if i == 0 else f"layer_norm_{i}"

        stages = [
            ("pre_core", self._pre_core_layers),
            ("core", self._core_layers),
            ("post_core", self._post_core_layers),
        ]

        for stage_name, n_layers in stages:
            if n_layers == 0:
                continue
            if n_layers != 1:
                raise NotImplementedError(
                    "DQN_ReDo currently supports exactly 1 layer per stage "
                    f"(got {stage_name}_layers={n_layers}). "
                    "Multi-layer per stage walks are not implemented."
                )
            # ReLU is a free function, so accumulatingSequence uses its bare
            # __name__ for every stage. LayerNorm is a Haiku module and gets
            # globally indexed by Haiku (layer_norm, layer_norm_1, ...), so its
            # probe key must use the same ln_idx that names the params.
            probe = ln_name(ln_idx) if self._score_after_ln else "relu"
            self._stage_plan.append(
                {
                    "stage": stage_name,
                    "act_key": f"{stage_name}/{probe}",
                    "linear_name": linear_name(linear_idx),
                    "ln_name": ln_name(ln_idx) if self._use_ln else None,
                }
            )
            linear_idx += 1
            if self._use_ln:
                ln_idx += 1

        for i, plan in enumerate(self._stage_plan):
            if i + 1 < len(self._stage_plan):
                plan["next_linear_name"] = self._stage_plan[i + 1]["linear_name"]
                plan["next_is_q"] = False
            else:
                plan["next_linear_name"] = None
                plan["next_is_q"] = True

        hypers = Hypers(
            **self.state.hypers.__dict__,
            redo_freq=int(params["redo_freq"]),
            redo_threshold=float(params["redo_threshold"]),
        )

        metrics = Metrics(
            **self.state.metrics.__dict__,
            dormant_neurons=jnp.float32(0.0),
        )

        self.state = AgentState(
            **{
                k: v
                for k, v in self.state.__dict__.items()
                if k not in ("hypers", "metrics")
            },
            hypers=hypers,
            metrics=metrics,
        )

        adam_state = self.state.optim[0]  # type: ignore
        assert isinstance(adam_state, optax.ScaleByAdamState), (
            "DQN_ReDo expects an Adam-based optimizer chain whose first element "
            f"is optax.ScaleByAdamState; got {type(adam_state).__name__}."
        )

        # redo_freq is in agent updates; periodic_freq is in env steps.
        self.periodic_freq = int(params["redo_freq"]) * int(
            self.state.hypers.update_freq
        )

    def _score(self, activation: jax.Array) -> jax.Array:
        reduce_axes = tuple(range(activation.ndim - 1))
        mean_activation = jnp.mean(jnp.abs(activation), axis=reduce_axes)
        score = mean_activation / (jnp.mean(mean_activation) + 1e-9)
        return score

    def _dormant(self, activation: jax.Array, threshold: float) -> jax.Array:
        return self._score(activation) <= threshold

    @partial(jax.jit, static_argnums=0)
    def _redo_step(self, state: AgentState) -> AgentState:
        key, sample_key, init_key = jax.random.split(state.key, 3)

        batch = self.buffer.sample(state.buffer_state, sample_key)
        x = batch.experience["x"][:, 0]
        scalars = batch.experience["scalars"][:, 0]

        feat = self.phi(state.params, x, scalars=scalars)
        threshold = state.hypers.redo_threshold

        # Mask trees mirror state.params; init to all-False.
        incoming_mask = jax.tree.map(
            lambda p: jnp.zeros(p.shape, dtype=bool), state.params
        )
        outgoing_mask = jax.tree.map(
            lambda p: jnp.zeros(p.shape, dtype=bool), state.params
        )

        phi = state.params["phi"]

        total_dormant = jnp.float32(0.0)
        total_neurons = jnp.float32(0.0)

        for plan in self._stage_plan:
            act_key = plan["act_key"]  # type: ignore[index]
            linear_name = plan["linear_name"]  # type: ignore[index]
            ln_name = plan["ln_name"]  # type: ignore[index]
            linear_path = f"phi/~/{linear_name}"

            dormant = self._dormant(feat.activations[act_key], threshold)
            total_dormant = total_dormant + dormant.sum().astype(jnp.float32)
            total_neurons = total_neurons + jnp.float32(dormant.shape[0])

            # Incoming weights / bias of this stage's Linear.
            incoming_mask["phi"][linear_path]["w"] = jnp.broadcast_to(
                dormant[None, :], phi[linear_path]["w"].shape
            )
            incoming_mask["phi"][linear_path]["b"] = dormant

            if self._use_ln and self._reset_ln and ln_name is not None:
                ln_path = f"phi/~/{ln_name}"
                incoming_mask["phi"][ln_path]["scale"] = dormant
                incoming_mask["phi"][ln_path]["offset"] = dormant

            # Outgoing weights of the next module (next Linear, or Q head).

            # Note: between the pre_core and core stages the ForagerNet
            # concatenates ``scalars`` onto the pre_core output, so the next
            # Linear's input dimension may be larger than this stage's output.
            # The dormant mask only applies to the first ``len(dormant)`` rows
            # (the pre_core contribution); rows from the scalars contribution
            # must remain untouched. We pad the mask with False rows.
            # The Q-head is treated as a sink: zero only its incoming weights
            # for dormant upstream neurons. Per Sokar et al., we don't probe
            # Q-output dormancy (no ReLU there) or re-init the head's own
            # weights — only sever the dead inputs.
            if plan["next_is_q"]:  # type: ignore[index]
                target_w = state.params["q"]["q"]["w"]
                col_mask = jnp.broadcast_to(
                    dormant[:, None], (dormant.shape[0], target_w.shape[1])
                )
                if col_mask.shape[0] != target_w.shape[0]:
                    pad = jnp.zeros(
                        (target_w.shape[0] - col_mask.shape[0], target_w.shape[1]),
                        dtype=bool,
                    )
                    col_mask = jnp.concatenate([col_mask, pad], axis=0)
                outgoing_mask["q"]["q"]["w"] = col_mask
            else:
                next_linear_path = f"phi/~/{plan['next_linear_name']}"  # type: ignore[index]
                target_w = phi[next_linear_path]["w"]
                col_mask = jnp.broadcast_to(
                    dormant[:, None], (dormant.shape[0], target_w.shape[1])
                )
                if col_mask.shape[0] != target_w.shape[0]:
                    pad = jnp.zeros(
                        (target_w.shape[0] - col_mask.shape[0], target_w.shape[1]),
                        dtype=bool,
                    )
                    col_mask = jnp.concatenate([col_mask, pad], axis=0)
                outgoing_mask["phi"][next_linear_path]["w"] = col_mask

        leaves, treedef = jax.tree.flatten(state.params)
        keys = jax.random.split(init_key, len(leaves))
        keys_tree = jax.tree.unflatten(treedef, list(keys))

        def _sample_new_param(path, init_fn, param, key):
            leaf_name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
            if leaf_name == "w":
                return init_fn(key, param.shape, param.dtype)
            return jnp.zeros_like(param)

        new_params = jax.tree_util.tree_map_with_path(
            _sample_new_param, self.initializers, state.params, keys_tree
        )

        def apply_redo(path, param, new_param, in_mask, out_mask):
            leaf_name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
            if leaf_name == "w":
                param = jnp.where(in_mask, new_param, param)
            elif leaf_name == "b":
                param = jnp.where(in_mask, jnp.zeros_like(param), param)
            elif leaf_name == "scale":
                param = jnp.where(in_mask, jnp.ones_like(param), param)
            elif leaf_name == "offset":
                param = jnp.where(in_mask, jnp.zeros_like(param), param)
            param = jnp.where(out_mask, jnp.zeros_like(param), param)
            return param

        new_params = jax.tree_util.tree_map_with_path(
            apply_redo, state.params, new_params, incoming_mask, outgoing_mask
        )

        # reset mu, nu of adam optimizer
        mask = jax.tree.map(lambda i, o: i | o, incoming_mask, outgoing_mask)
        adam_state: optax.ScaleByAdamState = state.optim[0]  # type: ignore
        new_mu = jax.tree.map(reset_momentum, adam_state.mu, mask)
        new_nu = jax.tree.map(reset_momentum, adam_state.nu, mask)
        new_adam_state = optax.ScaleByAdamState(adam_state.count, mu=new_mu, nu=new_nu)
        new_optim = (new_adam_state, *state.optim[1:])  # type: ignore

        dormant_neurons = total_dormant / total_neurons
        new_metrics = replace(state.metrics, dormant_neurons=dormant_neurons)

        if not self._redo_apply:
            return replace(state, key=key, metrics=new_metrics)

        return replace(
            state, key=key, params=new_params, optim=new_optim, metrics=new_metrics
        )

    def _periodic_step(self, state: AgentState) -> AgentState:
        return self._redo_step(state)
