# Modified from esraaelelimy/continuing_ppo

# Ensure third-party libraries that expect older JAX internals can import.
# This sets a small compatibility alias if needed before importing distrax/tfp.
import argparse
import logging
import os
import socket
import sys
import time
from collections.abc import Mapping
from functools import partial
from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct, traverse_util
from flax.training.train_state import TrainState
from gymnasium.utils.save_video import save_video
from jax.tree_util import tree_map
from jax_tqdm.base import PBar
from jax_tqdm.scan_pbar import scan_tqdm
from ml_instrumentation.Collector import Collector
from ml_instrumentation.metadata import attach_metadata
from ml_instrumentation.Sampler import Ignore, MovingAverage, Subsample
from ml_instrumentation.utils import Pipe
from PyExpUtils.results.tools import getParamsAsDict

import utils.jax_compat  # noqa: F401
from algorithms.nn.ACConv import ActorCriticConv
from algorithms.nn.ACMLP import ActorCriticMLP
from algorithms.nn.RealTimeACConv import RealTimeActorCriticConv
from algorithms.nn.RealTimeACConvMulti import RealTimeActorCriticConvMulti
from algorithms.nn.RealTimeACConvHint import RealTimeActorCriticConvHint
from algorithms.nn.RealTimeACConvHintRTU import RealTimeActorCriticConvHintRTU
from algorithms.nn.RealTimeACConvPooling import RealTimeActorCriticConvPooling
from algorithms.nn.RealTimeACMLP import RealTimeActorCriticMLP
from algorithms.nn.RealTimeACMLPMulti import RealTimeActorCriticMLPMulti
from algorithms.nn.RealTimeACMLPStacked import RealTimeActorCriticMLPStacked
from algorithms.PPORegistry import getAgent
from experiment import ExperimentModel
from utils.checkpoint import Checkpoint
from utils.ml_instrumentation.Sampler import Mean
from utils.ml_instrumentation.utils import Last
from utils.ppo_metrics import (
    compute_ppo_metrics,
    nan_ppo_metrics,
    nan_rtu_r_stats,
    nan_weight_drift,
    nan_weight_norm,
    rtu_r_stats,
    weight_drift,
    weight_norm,
)
from utils.preempt import TimeoutHandler

sys.path.insert(0, os.path.abspath("/tmp/src"))
from foragax.registry import make

PERIOD = 182500


def parse_indices(index_specs: list[str], total: int | None = None) -> list[int]:
    indices = []
    for spec in index_specs:
        if ":" not in spec:
            indices.append(int(spec))
            continue

        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid index slice '{spec}', expected START:STOP")

        start_s, stop_s = parts
        start = int(start_s) if start_s else 0
        if stop_s:
            stop = int(stop_s)
        elif total is not None:
            stop = total
        else:
            raise ValueError(f"Open-ended index slice '{spec}' requires total runs")

        indices.extend(range(start, stop))

    return indices


def _crossed_interval(start_step, end_step, interval):
    return (end_step > 0) & ((end_step // interval) > (start_step // interval))


@struct.dataclass
class LogEnvState:
    returned_returns: float
    timestep: int
    frames: Any


@struct.dataclass
class TrainConfig:
    # ---- STATIC (uniform across vmapped runs) ----
    d_hidden: int = struct.field(pytree_node=False)
    hidden_size: int = struct.field(pytree_node=False)
    activation: str = struct.field(pytree_node=False)
    agent_type: str = struct.field(pytree_node=False)
    rollout_steps: int = struct.field(pytree_node=False)
    epochs: int = struct.field(pytree_node=False)
    num_mini_batch: int = struct.field(pytree_node=False)
    gradient_clipping: bool = struct.field(pytree_node=False)
    num_updates: int = struct.field(pytree_node=False)
    env_id: str = struct.field(pytree_node=False)
    aperture_size: int = struct.field(pytree_node=False)
    render_mode: str = struct.field(pytree_node=False)
    env_kwargs: Any = struct.field(pytree_node=False)
    use_sinusoidal_encoding: bool = struct.field(pytree_node=False)
    use_reward_trace: bool = struct.field(pytree_node=False)
    use_hint_trace: bool = struct.field(pytree_node=False)
    use_layernorm: bool = struct.field(pytree_node=False)
    use_middle_layer: bool = struct.field(pytree_node=False)
    # Stacked-RTU (RealTimeActorCriticMLPStacked) only: number of [RTU + MLP]
    # residual blocks and whether residual adds are replaced by GTrXL GRU gates.
    n_blocks: int = struct.field(pytree_node=False)
    use_gating: bool = struct.field(pytree_node=False)
    conv: str = struct.field(pytree_node=False)
    allocate_frames: bool = struct.field(pytree_node=False)
    video_length: int = struct.field(pytree_node=False)
    use_l2_init: bool = struct.field(pytree_node=False)
    use_spectral_reg: bool = struct.field(pytree_node=False)
    use_reset: bool = struct.field(pytree_node=False)
    use_shrink_and_perturb: bool = struct.field(pytree_node=False)
    # NTK / churn plasticity metrics (computed inside the jitted scan)
    compute_ntk: bool = struct.field(pytree_node=False)
    ntk_freq: int = struct.field(pytree_node=False)
    n_ref: int = struct.field(pytree_node=False)
    # Row-chunk size for the memory-bounded NTK Gram build; result-invariant,
    # trades peak memory against recompute.  Defaults to n_ref // 4.
    chunked_ref: int = struct.field(pytree_node=False)
    # Weight-norm metric -- independent cadence / flag from the NTK metrics above
    compute_weight_norm: bool = struct.field(pytree_node=False)
    weight_norm_freq: int = struct.field(pytree_node=False)
    # RTU spectral-radius (r) stats -- mean/max pole magnitude over every RTU
    # diagonal, i.e. how long a memory the recurrence has actually learned.
    # Independent cadence / flag; works at any stack depth.
    compute_r_stats: bool = struct.field(pytree_node=False)
    r_stats_freq: int = struct.field(pytree_node=False)
    # Weight-drift metric (||theta - theta_0||, split pi/vf/total) -- independent
    # cadence / flag; the diagnostic counterpart to L2-to-init ("w0") reg.
    compute_weight_drift: bool = struct.field(pytree_node=False)
    weight_drift_freq: int = struct.field(pytree_node=False)
    # Plasticity probes (effective rank / dormancy / saturation). Explicit on/off
    # flag; probing additionally requires a probe-instrumented class (_is_probed).
    compute_plasticity: bool = struct.field(pytree_node=False)
    # ---- DYNAMIC (may vary per idx; arithmetic only) ----
    max_grad_norm: float
    l2_reg_pi: float
    l2_reg_vf: float
    alpha_pi: float
    alpha_vf: float
    adam_eps_pi: float
    adam_eps_vf: float
    adam_b1_pi: float
    adam_b2_pi: float
    adam_b1_vf: float
    adam_b2_vf: float

    sparsity: float
    spectral_radius: float

    id: int
    reward_trace_decay: float
    persist_decay: float
    sat_persist_threshold: float
    dormant_threshold: float
    gamma: float
    gae_lambda: float
    clip_eps: float
    vf_coef: float
    entropy_coef: float
    lambda_l2_init_pi: float = 0.0
    lambda_l2_init_vf: float = 0.0
    lambda_spectral_pi: float = 0.0
    lambda_spectral_vf: float = 0.0
    freeze_steps: int = -1
    reset_interval: int = -1
    sp_interval: int = -1
    shrink_factor: float = 1.0
    noise_scale: float = 0.0


class GymnaxEnvState(struct.PyTreeNode):
    to_render: bool = struct.field(pytree_node=True)
    cond_render: Callable = struct.field(pytree_node=False)
    env_step: Callable = struct.field(pytree_node=False)
    env_params: Any = struct.field(pytree_node=True)
    env_state: Any = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, env_step, env_params, env_state, **kwargs):
        """Creates a new instance"""
        return cls(
            env_step=env_step,
            env_params=env_params,
            env_state=env_state,
            **kwargs,
        )


class Transition(NamedTuple):
    action: jnp.ndarray  # a_t
    value: jnp.ndarray  # v(o_t)
    reward: jnp.ndarray  # r[t+1]
    log_prob: jnp.ndarray
    obs: Tuple[jnp.ndarray, ...]  # variable-length: (o_t, a_{t-1}, r_{t-1}, ...)
    info: jnp.ndarray


class Interaction(NamedTuple):
    a: int
    r: bool


# ----------------------------------------------------------------------------
# -- Plasticity metric helpers --
#
# Activation-aware probes captured during a rollout via Flax `sow` calls in
# RealTimeACMLP. Probe sites and metrics:
#   actor/critic_pre1, actor/critic_pre2  (pre-tanh)  → effective rank +
#       saturation rate  E_units[ 1{ |E_states[tanh(z)]| > 0.95 } ]: fraction of
#       units pinned to one rail across the probe set (constant / non-
#       differentiating), the tanh analogue of Sokar dormancy. sat_persist is
#       the same quantity with the per-unit mean EMA'd across rollouts.
#   actor/critic_rtu_out  (post-ReLU)  → effective rank + Sokar τ-dormant
#       fraction with τ = 0.025 (the regime Sokar 2023 was designed for).
# Expectations are estimated as sample means over the rollout's collapsed
# (T·B, H) activation matrix — Monte Carlo for E_{x∼D_rollout}.
# ----------------------------------------------------------------------------
# Network classes that carry the plasticity sow() probes. Probe-eligibility is
# decided by the resolved class (via the registry), NOT by exact agent name, so
# every suffixed variant (ActorCriticMLP-l2, ActorCriticMLP_relu_2, ...) and any
# activation (tanh/relu/crelu) is covered uniformly.
_PROBED_CLASSES = (ActorCriticMLP, RealTimeActorCriticMLP)


def _is_probed(agent_type):
    """True iff agent_type resolves to a probe-instrumented MLP/RTU-MLP class."""
    try:
        return getAgent(agent_type) in _PROBED_CLASSES
    except Exception:
        return False


def _should_probe(config):
    """Probing runs iff explicitly enabled AND the class carries probes."""
    return config.compute_plasticity and _is_probed(config.agent_type)


def _agent_is_rtu(agent_type):
    """Recurrent (RTU) MLP agent vs the feedforward vanilla MLP."""
    return agent_type.startswith("RealTime")


def _wide_site_name(config):
    """Sown intermediate for the wide middle layer, or None when it is absent.
    RTU nets always have it (rtu_out); a vanilla MLP has the wide Dense ('mid')
    only when use_middle_layer is set. Emitted into the unified '*_rtu' column."""
    if _agent_is_rtu(config.agent_type):
        return "rtu_out"
    return "mid" if config.use_middle_layer else None


def _effective_rank(activation):
    h = activation.reshape(-1, activation.shape[-1]).astype(jnp.float32)
    h = h - jnp.mean(h, axis=0, keepdims=True)
    s = jnp.linalg.svd(h, compute_uv=False)
    p = s / (jnp.sum(s) + 1e-9)
    entropy = -jnp.sum(p * jnp.log(p + 1e-9))
    return jnp.exp(entropy)


def _dormant_fraction(activation, threshold=0.025):
    h = activation.reshape(-1, activation.shape[-1])
    mean_act = jnp.mean(jnp.abs(h), axis=0)
    score = mean_act / (jnp.mean(mean_act) + 1e-9)
    return jnp.mean(score <= threshold).astype(jnp.float32)


def _saturation_rate(pre_activation, threshold=0.95):
    # Assumes a tanh nonlinearity follows this probe site. Signed
    # average-then-threshold: per unit take the signed mean of tanh over the
    # probe set (the rollout's states), then report the fraction of units whose
    # |mean| exceeds the threshold -- units pinned to one rail across these
    # states (constant / non-differentiating). This is the no-EMA, within-rollout
    # counterpart of sat_persist, which applies the *same* per-unit signed mean
    # (_mean_tanh_sites) but EMAs it across rollouts before thresholding.
    h = pre_activation.reshape(-1, pre_activation.shape[-1])
    mean_tanh = jnp.mean(jnp.tanh(h), axis=0)
    return jnp.mean(jnp.abs(mean_tanh) > threshold).astype(jnp.float32)


def _compute_plasticity_metrics(params, apply_fn, init_hstate, traj_obs, dormant_threshold=0.025):
    """RTU (tanh) plasticity metrics as a {column_name: scalar} dict from one
    extra forward pass: effective rank at pre1/rtu/pre2, signed saturation rate
    at the tanh sites (pre1/pre2), Sokar dormant fraction at the post-ReLU RTU
    outputs.
    """
    _, state = apply_fn(params, init_hstate, traj_obs, mutable=["intermediates"])
    inter = state["intermediates"]
    a_pre1, c_pre1 = inter["actor_pre1"][0], inter["critic_pre1"][0]
    a_rtu, c_rtu = inter["actor_rtu_out"][0], inter["critic_rtu_out"][0]
    a_pre2, c_pre2 = inter["actor_pre2"][0], inter["critic_pre2"][0]
    return {
        "eff_rank_actor_pre1": _effective_rank(a_pre1),
        "eff_rank_critic_pre1": _effective_rank(c_pre1),
        "eff_rank_actor_rtu": _effective_rank(a_rtu),
        "eff_rank_critic_rtu": _effective_rank(c_rtu),
        "eff_rank_actor_pre2": _effective_rank(a_pre2),
        "eff_rank_critic_pre2": _effective_rank(c_pre2),
        "sat_rate_actor_pre1": _saturation_rate(a_pre1),
        "sat_rate_critic_pre1": _saturation_rate(c_pre1),
        "sat_rate_actor_pre2": _saturation_rate(a_pre2),
        "sat_rate_critic_pre2": _saturation_rate(c_pre2),
        "dormant_actor_rtu": _dormant_fraction(a_rtu, dormant_threshold),
        "dormant_critic_rtu": _dormant_fraction(c_rtu, dormant_threshold),
    }


def _compute_plasticity_metrics_mlp(params, apply_fn, init_hstate, traj_obs, has_mid=True):
    """Vanilla ActorCriticMLP analogue of _compute_plasticity_metrics.

    ActorCriticMLP is feedforward and all-tanh, with a linear wide layer
    (dense2, width d_hidden) in the position the RTU occupies. Probe sites:
      actor/critic_pre1, actor/critic_pre2 (pre-tanh) -> effective rank +
          saturation rate, exactly as in the RTU net.
      actor/critic_mid (the linear wide layer)        -> effective rank only.

    There is no dormancy metric: that layer is linear (no ReLU) and has a
    rescaling symmetry (scale dense2 up, dense3 down -> identical function), so
    any per-unit magnitude/variance "dormant" measure is ill-defined. The linear
    analogue of dead capacity is rank deficiency, already captured by the mid
    effective rank. The dormant_*_rtu slots are therefore NaN (plots omit them).

    Returns the same dict keys as _compute_plasticity_metrics (eff_rank at the
    wide layer goes in the *_rtu slot); dormant_*_rtu are NaN.
    """
    _, state = apply_fn(params, init_hstate, traj_obs, mutable=["intermediates"])
    inter = state["intermediates"]
    a_pre1, c_pre1 = inter["actor_pre1"][0], inter["critic_pre1"][0]
    a_pre2, c_pre2 = inter["actor_pre2"][0], inter["critic_pre2"][0]
    nan = jnp.float32(jnp.nan)
    out = {
        "eff_rank_actor_pre1": _effective_rank(a_pre1),
        "eff_rank_critic_pre1": _effective_rank(c_pre1),
        "eff_rank_actor_pre2": _effective_rank(a_pre2),
        "eff_rank_critic_pre2": _effective_rank(c_pre2),
        "sat_rate_actor_pre1": _saturation_rate(a_pre1),
        "sat_rate_critic_pre1": _saturation_rate(c_pre1),
        "sat_rate_actor_pre2": _saturation_rate(a_pre2),
        "sat_rate_critic_pre2": _saturation_rate(c_pre2),
    }
    # Wide middle layer (the '*_rtu' slot): linear, so effective rank only and
    # NaN dormancy. Absent entirely when use_middle_layer is off.
    if has_mid:
        out["eff_rank_actor_rtu"] = _effective_rank(inter["actor_mid"][0])
        out["eff_rank_critic_rtu"] = _effective_rank(inter["critic_mid"][0])
        out["dormant_actor_rtu"] = nan
        out["dormant_critic_rtu"] = nan
    return out


def _compute_plasticity_metrics_relu(params, apply_fn, init_hstate, traj_obs,
                                     wide_inter: str | None = "rtu_out",
                                     dormant_threshold=0.025):
    """ReLU variant metrics dict: effective rank AND Sokar dormant fraction at
    every (post-ReLU) probe site -- pre1, the wide middle layer, pre2. No tanh
    saturation (this net has no tanh). The wide-layer activations come from
    `wide_inter` (the RTU output for the recurrent net, the wide Dense 'mid' for
    the feedforward MLP) but are emitted into the unified '*_rtu' column so the
    plotting layer set is architecture-independent."""
    _, state = apply_fn(params, init_hstate, traj_obs, mutable=["intermediates"])
    inter = state["intermediates"]
    sites = {
        "actor_pre1": inter["actor_pre1"][0], "critic_pre1": inter["critic_pre1"][0],
        "actor_pre2": inter["actor_pre2"][0], "critic_pre2": inter["critic_pre2"][0],
    }
    if wide_inter is not None:
        sites["actor_rtu"] = inter[f"actor_{wide_inter}"][0]
        sites["critic_rtu"] = inter[f"critic_{wide_inter}"][0]
    out = {}
    for k, h in sites.items():
        out[f"eff_rank_{k}"] = _effective_rank(h)
        out[f"dormant_{k}"] = _dormant_fraction(h, dormant_threshold)
    return out


def _zero_plasticity_metrics():
    return {}


def _mean_tanh_sites(params, apply_fn, init_hstate, traj_obs):
    """Per-unit signed mean tanh at the tanh sites (pre1/pre2), as a dict
    {site: (width,) vector}; feeds the persistent-saturation EMA. tanh is applied
    per state then averaged (signed) so a rail-flipping unit averages toward 0."""
    _, state = apply_fn(params, init_hstate, traj_obs, mutable=["intermediates"])
    inter = state["intermediates"]

    def _mt(name):
        h = inter[name][0]
        h = h.reshape(-1, h.shape[-1])
        return jnp.mean(jnp.tanh(h), axis=0)

    return {
        "actor_pre1": _mt("actor_pre1"), "critic_pre1": _mt("critic_pre1"),
        "actor_pre2": _mt("actor_pre2"), "critic_pre2": _mt("critic_pre2"),
    }


def _mean_abs_act_sites(params, apply_fn, init_hstate, traj_obs,
                        wide_inter: str | None = "rtu_out"):
    """Per-unit mean |activation| at the post-ReLU sites (pre1/wide/pre2), as a
    dict {site: (width,) vector}; feeds the persistent-dormancy EMA. The wide
    site reads `wide_inter` (RTU output or MLP 'mid') under the unified 'rtu'
    key."""
    _, state = apply_fn(params, init_hstate, traj_obs, mutable=["intermediates"])
    inter = state["intermediates"]
    names = {
        "actor_pre1": "actor_pre1", "critic_pre1": "critic_pre1",
        "actor_pre2": "actor_pre2", "critic_pre2": "critic_pre2",
    }
    if wide_inter is not None:
        names["actor_rtu"] = f"actor_{wide_inter}"
        names["critic_rtu"] = f"critic_{wide_inter}"

    def _ma(name):
        h = inter[name][0]
        h = h.reshape(-1, h.shape[-1])
        return jnp.mean(jnp.abs(h), axis=0)

    return {k: _ma(v) for k, v in names.items()}


def _sat_persist_from_ema(ema, threshold):
    """Persistent-saturation metrics from the EMA dict: fraction of units pinned
    to one rail (|EMA signed-mean-tanh| > threshold), per site."""
    return {
        f"sat_persist_{k}": jnp.mean((jnp.abs(v) > threshold).astype(jnp.float32))
        for k, v in ema.items()
    }


def _dormant_persist_from_ema(ema, threshold):
    """Persistent-dormancy metrics from the EMA dict: Sokar score on the EMA'd
    per-unit mean |activation|, fraction with score <= threshold, per site."""
    out = {}
    for k, v in ema.items():
        score = v / (jnp.mean(v) + 1e-9)
        out[f"dormant_persist_{k}"] = jnp.mean((score <= threshold).astype(jnp.float32))
    return out


def _pers_ema_init(config):
    """Initial per-unit EMA carry (dict {site: zeros(width)}), agent-specific.
    tanh nets track pre1/pre2 (hidden_size); the ReLU net also tracks the RTU
    output (2*d_hidden). Empty for agents without probes."""
    H = config.hidden_size
    if not _should_probe(config):
        return {}
    if config.activation in ("relu", "crelu"):
        ema = {
            "actor_pre1": jnp.zeros(H), "critic_pre1": jnp.zeros(H),
            "actor_pre2": jnp.zeros(H), "critic_pre2": jnp.zeros(H),
        }
        wide_inter = _wide_site_name(config)
        if wide_inter is not None:
            # wide middle layer: RTU output (2*d_hidden) or MLP wide Dense (d_hidden)
            wide = 2 * config.d_hidden if _agent_is_rtu(config.agent_type) else config.d_hidden
            ema["actor_rtu"] = jnp.zeros(wide)
            ema["critic_rtu"] = jnp.zeros(wide)
        return ema
    return {
        "actor_pre1": jnp.zeros(H), "critic_pre1": jnp.zeros(H),
        "actor_pre2": jnp.zeros(H), "critic_pre2": jnp.zeros(H),
    }


# ----------------------------------------------------------------------------
# -- Gradient-norm helpers --
#
# Per-layer l0 / l1 / l2 norms of the raw (pre-clip) gradient, computed each
# minibatch update and averaged over a rollout. Each top-level Flax module is
# one "layer" (the whole RTU is lumped); LayerNorm modules are excluded. l0 is
# the count of entries with |g| > eps (1e-8) — meaningful at the post-ReLU RTU
# site, ~flat for tanh layers (tanh' is never exactly 0), so l1/l2 carry the
# plasticity signal there. Param-count weighting across layers is done offline.
# ----------------------------------------------------------------------------
_GRAD_NORM_EPS = 1e-8


def _grad_layer_name(path):
    parts = list(path)
    if parts and parts[0] == "params":
        parts = parts[1:]
    return parts[0] if parts else ""


def _grad_layer_buckets(tree):
    """Map each non-LayerNorm leaf to its top-level module ('actor_dense1',
    'actor_rtu', ...), lumping the whole RTU together. Returns {layer: [leaves]}."""
    flat = traverse_util.flatten_dict(tree)
    buckets = {}
    for path, arr in flat.items():
        layer = _grad_layer_name(path)
        if not layer or "layernorm" in layer.lower():
            continue
        buckets.setdefault(layer, []).append(arr)
    return buckets


def _per_layer_grad_norms(grads, eps=_GRAD_NORM_EPS):
    """Per-layer (l0, l1, l2) norms of the raw gradient. Returns {layer: (l0, l1, l2)}."""
    buckets = _grad_layer_buckets(grads)
    norms = {}
    for layer in sorted(buckets):
        v = jnp.concatenate([a.reshape(-1) for a in buckets[layer]])
        absv = jnp.abs(v)
        norms[layer] = (
            jnp.sum(absv > eps).astype(jnp.float32),  # l0
            jnp.sum(absv),                            # l1
            jnp.sqrt(jnp.sum(v * v)),                 # l2
        )
    return norms


def _zero_grad_norms(params, epochs, num_mini_batch):
    """Zero grad-norm tree matching the scanned (epochs, num_mini_batch) shape,
    for the frozen branch of the update cond."""
    z = jnp.zeros((epochs, num_mini_batch), dtype=jnp.float32)
    return {layer: (z, z, z) for layer in sorted(_grad_layer_buckets(params))}


def _grad_layer_param_counts(params):
    """Static per-layer parameter counts for offline param-count weighting."""
    buckets = _grad_layer_buckets(params)
    return {
        layer: jnp.float32(sum(int(a.size) for a in buckets[layer]))
        for layer in sorted(buckets)
    }


@jax.jit
def calculate_gae(traj_batch, last_val, gamma, gae_lambda):
    def _get_advantages(carry, transition):
        gae, next_value = carry
        value, reward = (
            transition.value,
            transition.reward,
        )
        delta = reward + gamma * next_value - value
        gae = delta + gamma * gae_lambda * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value


@jax.jit
def calculate_average_reward_gae(traj_batch, last_val, gamma, gae_lambda):
    sample_avg_reward = jnp.mean(traj_batch.reward)  # r_\pi

    def _get_advantages(next_value, transition):
        value, reward = (
            transition.value,
            transition.reward,
        )
        gae = reward - sample_avg_reward + next_value - value
        return value, gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        last_val,
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value


@partial(jax.jit, static_argnums=(1, 9, 12))
def loss_fn(
    params,
    agent_fn,
    traj_batch,
    gae,
    targets,
    init_hstate,
    clip_eps,
    vf_coef,
    ent_coef,
    use_l2_init=False,
    initial_params=None,
    l2_init_multipliers=None,
    use_spectral_reg=False,
    spectral_reg_multipliers=None,
):
    rnn_in = traj_batch.obs
    _, pi, value = agent_fn(params, init_hstate, rnn_in)
    log_prob = pi.log_prob(traj_batch.action)
    # CALCULATE VALUE LOSS
    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
        -clip_eps, clip_eps
    )
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    # CALCULATE ACTOR LOSS
    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - clip_eps,
            1.0 + clip_eps,
        )
        * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()
    total_loss = loss_actor + vf_coef * value_loss - ent_coef * entropy

    # L2-to-init regularisation (guard: compiled away when use_l2_init is False)
    if use_l2_init:
        # optax.l2_loss(a, b) = 0.5 * (a - b)^2 per element
        l2_init_loss = jax.tree_util.tree_map(
            lambda m, p, p0: m * optax.l2_loss(p, p0).sum(),
            l2_init_multipliers,
            params,
            initial_params,
        )
        total_loss = total_loss + jax.tree_util.tree_reduce(
            lambda a, b: a + b, l2_init_loss
        )

    # Spectral regularisation (Lyle, Rowland, Dabney & Gal, 2024; adapted
    # following Machado's group work on loss of plasticity).
    #
    # For each layer l with weight W_l and bias b_l the regulariser is:
    #   R(θ_l) = (σ₁(W_l)^k − 1)² + ‖b_l‖₂^(2k)
    # with k = 2 (paper default).
    #
    # Special cases handled at trace time via parameter name:
    #   • Dense/Conv kernel  → (σ₁^k − 1)²   (spectral norm towards 1)
    #   • Bias               → ‖b‖₂^(2k)      (towards 0)
    #   • LayerNorm scale    → Σ(γ_i − 1)²    (element-wise towards 1)
    #   • Conv 4-D tensors   → reshape to (d_out, k·k·d_in) then σ₁
    #
    # σ₁ is estimated with a single power-iteration step (Yoshida & Miyato,
    # 2017) which the paper finds sufficient.  The compile-time flag
    # `use_spectral_reg` causes this entire block to be compiled away when
    # spectral regularisation is not requested.
    if use_spectral_reg:
        _SR_K = 2  # exponent k from the paper

        def _power_iteration_sigma1(w_2d, num_iters=1):
            """Estimate σ₁(w_2d) via power iteration (1 step by default)."""
            u = jnp.ones((w_2d.shape[0],), dtype=w_2d.dtype)
            u = u / (jnp.linalg.norm(u) + 1e-12)

            def _step(u, _):
                v = w_2d.T @ u
                v = v / (jnp.linalg.norm(v) + 1e-12)
                u_new = w_2d @ v
                u_new = u_new / (jnp.linalg.norm(u_new) + 1e-12)
                return u_new, v

            u, vs = jax.lax.scan(_step, u, None, length=num_iters)
            v = vs[-1]
            sigma = u @ w_2d @ v
            return sigma

        def _spectral_leaf(path, multiplier, param):
            """Per-leaf spectral regularisation loss (dispatched at trace time)."""
            # Identify the leaf name from its path key
            leaf_name = (
                path[-1].key.lower()
                if hasattr(path[-1], "key")
                else str(path[-1]).lower()
            )

            # --- LayerNorm / normalisation scale parameters ---
            # Paper: "regularize each weight towards 1"
            if leaf_name in ("scale", "gamma"):
                return multiplier * jnp.sum(jnp.square(param - 1.0))

            # --- Bias / additive parameters ---
            # Paper: ‖b‖₂^(2k)
            if leaf_name in ("bias", "beta", "offset") or param.ndim == 1:
                return multiplier * jnp.linalg.norm(param) ** (2 * _SR_K)

            # --- Convolutional kernels (4-D) ---
            # Paper (Appendix A.7): reshape to (d_out, k·k·d_in)
            # Flax Conv default layout: (spatial..., d_in, d_out)
            if param.ndim == 4:
                # (h, w, d_in, d_out) → (d_out, h*w*d_in)
                d_out = param.shape[-1]
                w_2d = jnp.transpose(param, (3, 0, 1, 2)).reshape((d_out, -1))
                sigma = _power_iteration_sigma1(w_2d)
                return multiplier * jnp.square(sigma**_SR_K - 1.0)

            # --- Dense / multiplicative weight matrices (2-D) ---
            # Paper: (σ₁(W)^k − 1)²
            if param.ndim == 2:
                sigma = _power_iteration_sigma1(param)
                return multiplier * jnp.square(sigma**_SR_K - 1.0)

            # Anything else (scalars, etc.) – skip
            return jnp.zeros((), dtype=param.dtype)

        spectral_losses = jax.tree_util.tree_map_with_path(
            _spectral_leaf,
            spectral_reg_multipliers,
            params,
        )
        total_loss = total_loss + jax.tree_util.tree_reduce(
            lambda a, b: a + b, spectral_losses
        )

    return total_loss, (value_loss, loss_actor, entropy)


@jax.jit
def agent_step(last_obs, train_state, rng, hstate):
    last_obs, last_action_encoded, last_reward, sine, cosine, reward_trace = last_obs
    rnn_in = (
        jnp.expand_dims(last_obs, 0),
        jnp.expand_dims(last_action_encoded, 0),
        jnp.expand_dims(last_reward, 0),
        jnp.expand_dims(sine, 0),
        jnp.expand_dims(cosine, 0),
        jnp.expand_dims(reward_trace, 0),
    )
    last_hidden, pi, value = train_state.apply_fn(train_state.params, hstate, rnn_in)
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)
    return action, log_prob, value, last_hidden


def env_step(runner_state, _):
    (
        train_state,
        gymnax_state,
        log_env_state,
        config,
        last_obs,
        last_action,
        last_reward,
        reward_trace,
        hint_trace,
        rng,
        hstate,
    ) = runner_state

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    action_encoded = jnp.zeros((4,))
    action_encoded = action_encoded.at[last_action].set(1)
    last_reward_encoded = jnp.expand_dims(last_reward, 0)

    if isinstance(last_obs, Mapping):
        obs_img = last_obs["image"]
        hint = last_obs["hint"]
        if config.use_hint_trace:
            hint_trace = (
                config.reward_trace_decay * hint_trace
                + (1.0 - config.reward_trace_decay) * hint
            )
            hint = hint_trace
        last_reward_encoded = jnp.concatenate((last_reward_encoded, hint), axis=-1)
    else:
        obs_img = last_obs

    sine = jnp.expand_dims(jnp.sin(2 * jnp.pi * log_env_state.timestep / PERIOD), 0)
    cosine = jnp.expand_dims(jnp.cos(2 * jnp.pi * log_env_state.timestep / PERIOD), 0)
    reward_trace = (
        config.reward_trace_decay * reward_trace
        + (1.0 - config.reward_trace_decay) * last_reward
    )
    reward_trace_encoded = jnp.expand_dims(reward_trace, 0)
    last_obs_encoded = (
        obs_img,
        action_encoded,
        last_reward_encoded,
        sine,
        cosine,
        reward_trace_encoded,
    )

    action, log_prob, value, last_hidden = agent_step(
        last_obs_encoded, train_state, _rng, hstate
    )
    # STEP ENV
    obs, env_state, reward, done, info = gymnax_state.env_step(
        _rng, gymnax_state.env_state, action.squeeze(), gymnax_state.env_params
    )
    step = log_env_state.timestep + 1
    new_return = 0.999 * log_env_state.returned_returns + (1.0 - 0.999) * (reward)

    frame = gymnax_state.cond_render(gymnax_state.to_render, gymnax_state.env_state)

    log_env_state = LogEnvState(
        returned_returns=new_return, timestep=step, frames=log_env_state.frames
    )

    info["reward"] = reward
    info["moving_average"] = new_return
    info["timestep"] = log_env_state.timestep
    info["pos"] = env_state.pos
    info["frame"] = frame

    ### Create transition
    transition = Transition(
        action.squeeze(),
        value.squeeze(),
        reward,
        log_prob.squeeze(),
        last_obs_encoded,
        info,
    )
    ### Update runner state
    gymnax_state = GymnaxEnvState.create(
        to_render=gymnax_state.to_render,
        cond_render=gymnax_state.cond_render,
        env_step=gymnax_state.env_step,
        env_params=gymnax_state.env_params,
        env_state=env_state,
    )
    runner_state = (
        train_state,
        gymnax_state,
        log_env_state,
        config,
        obs,
        action.squeeze(),
        reward,
        reward_trace,
        hint_trace,
        rng,
        last_hidden,
    )
    return runner_state, (transition, hstate)


@jax.jit
def update_minbatch(carry_in, batch_info):
    (
        train_state,
        config,
        initial_params,
        l2_init_multipliers,
        spectral_reg_multipliers,
    ) = carry_in
    minibatch, init_hstate = batch_info
    # minibatch: (seq_len,minibatch_size, _)
    # init_hstate: (1, d_hidden)
    traj_batch, advantages, targets = minibatch
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    total_loss, grads = grad_fn(
        train_state.params,
        train_state.apply_fn,
        traj_batch,
        advantages,
        targets,
        init_hstate,
        config.clip_eps,
        config.vf_coef,
        config.entropy_coef,
        config.use_l2_init,
        initial_params,
        l2_init_multipliers,
        config.use_spectral_reg,
        spectral_reg_multipliers,
    )
    # Per-layer grad norms on the RAW (pre-clip) gradient, before the optimizer
    # chain's clip_by_global_norm would cap l2 at max_grad_norm.
    grad_norms = _per_layer_grad_norms(grads)
    train_state = train_state.apply_gradients(grads=grads)
    return (train_state, config, initial_params, l2_init_multipliers, spectral_reg_multipliers), (total_loss, grad_norms)


"""
Batch shape = (num_steps, _)
Divide the batch into n minibatches
each minibatch has the shape of (seq_len, minibatch_size, _)
minibatch_size = num_steps//n*seq_len

1. re-run the network through the batch and store hiddens states for positions (0,seq_len,2*seq_len,...)
2. Divide num_steps into sequences of length seq_len : number of sequences = num_steps//seq_len
3. Divide the sequences into n minibatches
4. shuffle the minibatches
output shape = (num_minibatches, seq_len, minibatch_size, _)
"""


@jax.jit
def create_minibaches(config: TrainConfig, hstate_batch, batch, rng, train_state):
    batch_hstate = jax.tree_util.tree_map(
        lambda y: jnp.squeeze(y, axis=1), hstate_batch
    )
    traj_batch, advantages, targets = batch
    batch = (batch_hstate, traj_batch, advantages, targets)

    rng, _rng = jax.random.split(rng)
    permutation = jax.random.permutation(_rng, config.rollout_steps)
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch
    )

    minibatch_size = config.rollout_steps // config.num_mini_batch
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: x.reshape(
            (
                config.num_mini_batch,
                minibatch_size,
            )
            + x.shape[1:]
        ),
        shuffled_batch,
    )

    batch_hstate, traj_batch, advantages, targets = shuffled_batch
    minibatches_info = ((traj_batch, advantages, targets), batch_hstate)
    return minibatches_info, rng


@jax.jit
def update_epoch(update_state, unused):
    (
        train_state,
        init_hstate,
        traj_batch,
        hstate_batch,
        advantages,
        targets,
        rng,
        config,
        initial_params,
        l2_init_multipliers,
        spectral_reg_multipliers,
    ) = update_state
    batch = (traj_batch, advantages, targets)
    minibatches_info, rng = create_minibaches(
        config, hstate_batch, batch, rng, train_state
    )
    carry_in = (train_state, config, initial_params, l2_init_multipliers, spectral_reg_multipliers)
    carry_out, minibatch_out = jax.lax.scan(update_minbatch, carry_in, minibatches_info)
    train_state = carry_out[0]
    update_state = (
        train_state,
        init_hstate,
        traj_batch,
        hstate_batch,
        advantages,
        targets,
        rng,
        config,
        initial_params,
        l2_init_multipliers,
        spectral_reg_multipliers,
    )
    return update_state, minibatch_out


@jax.jit
def update_step(update_state):
    (
        train_state,
        init_hstate,
        traj_batch,
        hstate_batch,
        advantages,
        targets,
        rng,
        config,
        initial_params,
        l2_init_multipliers,
        spectral_reg_multipliers,
    ) = update_state
    update_state, loss_info = jax.lax.scan(
        update_epoch, update_state, None, config.epochs
    )
    ## Update runner state
    train_state = update_state[0]
    rng = update_state[6]
    return (train_state, rng), loss_info


def experiment(rng, config: TrainConfig):
    kwargs = dict(config.env_kwargs)

    print(
        f"Creating env {config.env_id} with aperture size {config.aperture_size} and kwargs {kwargs}"
    )

    env = make(config.env_id, aperture_size=config.aperture_size, **kwargs)

    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env.default_params)

    def real_render(env_state):
        return env.render(env_state, None, render_mode=config.render_mode).astype(
            jnp.uint8
        )

    render_shape = real_render(env_state).shape

    ### Initialize the environment states
    if config.allocate_frames:
        updates_per_video = (
            config.video_length + config.rollout_steps - 1
        ) // config.rollout_steps
        frames = jnp.zeros(
            (
                updates_per_video * config.rollout_steps,
                *render_shape,
            ),
            dtype=jnp.uint8,
        )
    else:
        frames = jnp.zeros((0, *render_shape), dtype=jnp.uint8)
    log_env_state = LogEnvState(returned_returns=0, timestep=0, frames=frames)

    def void_render(env_state):
        return jnp.zeros(render_shape, dtype=jnp.uint8)

    def render(cond, env_state):
        return jax.lax.cond(cond, real_render, void_render, env_state)

    gymnax_state = GymnaxEnvState.create(
        to_render=False,
        cond_render=render,
        env_step=env.step,
        env_params=env.default_params,
        env_state=env_state,
    )
    action_dim = 4

    _agent_class = getAgent(config.agent_type)
    agent = _agent_class

    if config.activation == "crelu" and _agent_class not in (
        ActorCriticConv,
        ActorCriticMLP,
        RealTimeActorCriticConv,
        RealTimeActorCriticMLP,
    ):
        raise NotImplementedError(
            "CReLU activation is currently wired for ActorCriticConv, "
            "ActorCriticMLP, RealTimeActorCriticConv, and "
            "RealTimeActorCriticMLP."
        )

    kwargs = {}
    if config.sparsity is not None:
        kwargs["sparsity"] = config.sparsity
    if config.spectral_radius is not None:
        kwargs["spectral_radius"] = config.spectral_radius
    if _agent_class in (
        ActorCriticConv,
        RealTimeActorCriticConv,
        RealTimeActorCriticConvMulti,
        RealTimeActorCriticConvPooling,
        RealTimeActorCriticConvHint,
        RealTimeActorCriticConvHintRTU,
    ):
        kwargs["conv"] = config.conv
    if _agent_class is ActorCriticMLP:
        kwargs["use_middle_layer"] = config.use_middle_layer
    if _agent_class is RealTimeActorCriticMLPStacked:
        kwargs["n_blocks"] = config.n_blocks
        kwargs["use_gating"] = config.use_gating

    # Create and initialize the network. `agent` is dynamically dispatched via
    # getAgent(config.agent_type); pyright sees only the base type so it can't
    # verify variant-specific kwargs like use_layernorm. The activation (tanh or
    # relu) comes from the explicit `representation.activation` config field.
    network = agent(
        action_dim=action_dim,
        activation=config.activation,
        hidden_size=config.hidden_size,
        d_hidden=config.d_hidden,
        cont=False,
        use_sinusoidal_encoding=config.use_sinusoidal_encoding,  # pyright: ignore[reportCallIssue]
        use_reward_trace=config.use_reward_trace,  # pyright: ignore[reportCallIssue]
        use_layernorm=config.use_layernorm,  # pyright: ignore[reportCallIssue]
        **kwargs,
    )

    rng, _rng = jax.random.split(rng)

    if isinstance(obs, Mapping):
        obs_img_shape = obs["image"].shape
        hint_shape = (1 + obs["hint"].shape[-1],)
        hint_dim = obs["hint"].shape[-1]
    else:
        obs_img_shape = obs.shape
        hint_shape = (1,)
        hint_dim = 1  # placeholder; not used for non-hint envs

    init_x = (
        jnp.zeros((1, *obs_img_shape)),
        jnp.zeros((1, action_dim)),
        jnp.zeros((1, *hint_shape)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
    )

    _is_conv_rtu = _agent_class in (
        RealTimeActorCriticConv,
        RealTimeActorCriticConvPooling,
        RealTimeActorCriticConvHint,
    )
    _is_plain_conv_rtu = _agent_class is RealTimeActorCriticConv
    _is_conv_hint_rtu = _agent_class is RealTimeActorCriticConvHintRTU
    _is_stacked_rtu = _agent_class is RealTimeActorCriticMLPStacked
    # Conv-Multi shares MLPMulti's 4-RTU carry and width-hidden_size RTU input.
    _is_multi_rtu = _agent_class in (
        RealTimeActorCriticMLPMulti,
        RealTimeActorCriticConvMulti,
    )
    _is_mlp_rtu = _agent_class in (
        RealTimeActorCriticMLP,
        ActorCriticMLP,
    )
    activation_multiplier = 2 if config.activation == "crelu" else 1
    if _is_stacked_rtu:
        # Every block's RTU reads the LayerNorm'd width-W stream (W = hidden_size);
        # action/reward are folded into the stream by the input projection.
        d_input = config.hidden_size
    elif _is_multi_rtu:
        # Both RTUs read a width-hidden_size vector (obs embedding sized to
        # obs_hidden_size, then action/reward re-concatenated before each RTU).
        d_input = config.hidden_size
    elif _is_plain_conv_rtu:
        # RTU receives [conv Dense(hidden_size), action, last_reward+hint, ...]
        d_input = (
            config.hidden_size * activation_multiplier + action_dim + hint_shape[0]
        )
        if config.use_sinusoidal_encoding:
            d_input += 2
        if config.use_reward_trace:
            d_input += 1
    elif _is_conv_rtu:
        d_input = config.hidden_size * activation_multiplier
    elif _is_conv_hint_rtu:
        # No main RTU — d_input is ignored by initialize_memory, pass hint input size
        d_input = hint_dim
    elif _is_mlp_rtu:
        # RTU receives [Dense(hidden_size), action, last_reward+hint, ...]
        # hint_shape[0] = 1 + hint_dim (accounts for reward + hint)
        d_input = (
            config.hidden_size * activation_multiplier + action_dim + hint_shape[0]
        )
        if config.use_sinusoidal_encoding:
            d_input += 2
        if config.use_reward_trace:
            d_input += 1
    else:
        d_input = config.hidden_size
    if _is_conv_hint_rtu:
        init_hstate = agent.initialize_memory(1, config.d_hidden, hint_dim)
    elif _is_stacked_rtu:
        init_hstate = agent.initialize_memory(
            1, config.d_hidden, d_input, n_blocks=config.n_blocks  # pyright: ignore[reportCallIssue]
        )
    else:
        init_hstate = agent.initialize_memory(1, config.d_hidden, d_input)
    network_params = network.init(_rng, init_hstate, init_x)

    # Static per-layer parameter counts for offline param-count weighting of
    # the gradient norms.
    grad_nparams = _grad_layer_param_counts(network_params)

    def make_label_tree(params):
        flat = traverse_util.flatten_dict(params, sep="/")

        def label_for_path(path_str):
            if "critic" in path_str:
                return "vf"
            elif "actor" in path_str:
                return "pi"
            elif "frozen" in path_str:
                return "frozen"
            return ""

        labels_flat = {k: label_for_path(k) for k in flat.keys()}
        return traverse_util.unflatten_dict(
            {tuple(k.split("/")): v for k, v in labels_flat.items()}
        )

    labels = make_label_tree(network_params)

    if config.gradient_clipping:
        tx = optax.partition(
            {
                "pi": optax.chain(
                    optax.clip_by_global_norm(config.max_grad_norm),
                    optax.add_decayed_weights(config.l2_reg_pi),
                    optax.adam(
                        config.alpha_pi,
                        b1=config.adam_b1_pi,
                        b2=config.adam_b2_pi,
                        eps=config.adam_eps_pi,
                    ),
                ),
                "vf": optax.chain(
                    optax.clip_by_global_norm(config.max_grad_norm),
                    optax.add_decayed_weights(config.l2_reg_vf),
                    optax.adam(
                        config.alpha_vf,
                        b1=config.adam_b1_vf,
                        b2=config.adam_b2_vf,
                        eps=config.adam_eps_vf,
                    ),
                ),
                "frozen": optax.set_to_zero(),
            },
            labels,
        )
    else:
        tx = optax.partition(
            {
                "pi": optax.chain(
                    optax.add_decayed_weights(config.l2_reg_pi),
                    optax.adam(
                        config.alpha_pi,
                        b1=config.adam_b1_pi,
                        b2=config.adam_b2_pi,
                        eps=config.adam_eps_pi,
                    ),
                ),
                "vf": optax.chain(
                    optax.add_decayed_weights(config.l2_reg_vf),
                    optax.adam(
                        config.alpha_vf,
                        b1=config.adam_b1_vf,
                        b2=config.adam_b2_vf,
                        eps=config.adam_eps_vf,
                    ),
                ),
                "frozen": optax.set_to_zero(),
            },
            labels,
        )

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    # L2-to-init: only allocate the frozen copy and multiplier tree when enabled.
    # When disabled, these are None and loss_fn's guard skips the regularisation.
    if config.use_l2_init:
        initial_params = jax.tree_util.tree_map(lambda p: p.copy(), network_params)

        def _make_l2_init_multipliers(params, labels, lambda_pi, lambda_vf):
            flat_params = traverse_util.flatten_dict(params, sep="/")
            flat_labels = traverse_util.flatten_dict(labels, sep="/")
            lam_map = {"pi": lambda_pi, "vf": lambda_vf}
            flat_mult = {
                k: jnp.array(lam_map.get(str(flat_labels[k]), 0.0))
                for k in flat_params
            }
            return traverse_util.unflatten_dict(
                {tuple(k.split("/")): v for k, v in flat_mult.items()}
            )

        l2_init_multipliers = _make_l2_init_multipliers(
            network_params,
            labels,
            config.lambda_l2_init_pi,
            config.lambda_l2_init_vf,
        )
    else:
        initial_params = None
        l2_init_multipliers = None

    # Spectral regularisation: build per-leaf multiplier tree.
    # Every parameter is regularised (weights, biases, norm scales) — the
    # specific regularisation form for each leaf is decided at trace time in
    # loss_fn based on the parameter name / ndim.  When disabled, the
    # multiplier tree is None and the compile-time guard in loss_fn skips
    # the whole block.
    if config.use_spectral_reg:

        def _make_spectral_reg_multipliers(params, labels, lambda_pi, lambda_vf):
            flat_params = traverse_util.flatten_dict(params, sep="/")
            flat_labels = traverse_util.flatten_dict(labels, sep="/")
            lam_map = {"pi": lambda_pi, "vf": lambda_vf}
            flat_mult = {
                k: jnp.array(lam_map.get(str(flat_labels[k]), 0.0))
                for k in flat_params
            }
            return traverse_util.unflatten_dict(
                {tuple(k.split("/")): v for k, v in flat_mult.items()}
            )

        spectral_reg_multipliers = _make_spectral_reg_multipliers(
            network_params,
            labels,
            config.lambda_spectral_pi,
            config.lambda_spectral_vf,
        )
    else:
        spectral_reg_multipliers = None

    ### Experiment
    def _zero_loss_info(config: TrainConfig):
        zeros = jnp.zeros((config.epochs, config.num_mini_batch), dtype=jnp.float32)
        return (zeros, (zeros, zeros, zeros))

    # Last-layer reset: reinitialize actor_mean and critic_value params + optimizer states.
    # The network is re-initialized from scratch with a fresh key; only the last-layer
    # leaves are swapped in (params and optimizer state).  Guarded by `config.use_reset`.
    if config.use_reset:
        # Pre-compute which leaves belong to the last layer, keyed by flattened path.
        _flat_params = traverse_util.flatten_dict(network_params, sep="/")
        _last_layer_keys = frozenset(
            k for k in _flat_params if "actor_mean" in k or "critic_value" in k
        )

        def _is_last_layer(path_str):
            return path_str in _last_layer_keys

        _last_layer_mask = traverse_util.unflatten_dict(
            {tuple(k.split("/")): _is_last_layer(k) for k in _flat_params}
        )

        def _reset_last_layer(train_state, rng):
            """Reinitialize last-layer params and their optimizer states."""
            rng, init_rng = jax.random.split(rng)
            # Get fresh random parameters for the entire network
            fresh_params = network.init(init_rng, init_hstate, init_x)

            # Selectively replace only last-layer params
            new_params = jax.tree_util.tree_map(
                lambda mask, fresh, old: jnp.where(mask, fresh, old),
                _last_layer_mask,
                fresh_params,
                train_state.params,
            )

            # Reinitialize optimizer state: build fresh opt state from new params,
            # then selectively swap only last-layer entries.
            fresh_opt_state = tx.init(new_params)
            new_opt_state = jax.tree_util.tree_map_with_path(
                lambda path, fresh_val, old_val: (
                    fresh_val
                    if any(
                        (
                            hasattr(k, "key")
                            and ("actor_mean" in k.key or "critic_value" in k.key)
                        )
                        for k in path
                    )
                    else old_val
                ),
                fresh_opt_state,
                train_state.opt_state,
            )

            train_state = train_state.replace(
                params=new_params,
                opt_state=new_opt_state,
            )
            return train_state, rng

    # Shrink and Perturb: shrink all params towards zero and add Gaussian noise,
    # then reinitialize optimizer state.  Guarded by `config.use_shrink_and_perturb`.
    if config.use_shrink_and_perturb:

        def _shrink_and_perturb(train_state, rng):
            """Apply shrink-and-perturb to all network parameters."""
            rng, subkey = jax.random.split(rng)

            leaves, treedef = jax.tree_util.tree_flatten(train_state.params)
            leaf_keys = jax.random.split(subkey, len(leaves))
            keys_tree = jax.tree_util.tree_unflatten(treedef, leaf_keys)

            def sp(k, p):
                noise = jax.random.normal(k, shape=p.shape, dtype=p.dtype)
                return p * config.shrink_factor + noise * config.noise_scale

            new_params = jax.tree_util.tree_map(sp, keys_tree, train_state.params)

            # Reinitialize optimizer state for the perturbed parameters
            new_opt_state = tx.init(new_params)

            train_state = train_state.replace(
                params=new_params,
                opt_state=new_opt_state,
            )
            return train_state, rng

    # NTK / churn metrics: collect a fixed batch of reference observations from a
    # short random-policy rollout.  This branches off the env at its reset state
    # via a derived key (rng is not consumed, so the training stream is
    # unaffected) and the rollout's terminal state is discarded.  `reward_dim`
    # is the width of the `last_reward` feature (1, or 1 + hint_dim for hint
    # envs), needed to build zero-padded reference obs tuples.
    reward_dim = hint_shape[0]
    if config.compute_ntk:
        ref_rng = jax.random.fold_in(rng, 7919)

        def _ref_step(carry, _):
            r, st = carry
            r, a_rng, s_rng = jax.random.split(r, 3)
            ref_action = jax.random.randint(a_rng, (), 0, action_dim)
            ref_obs, st, _, _, _ = env.step(s_rng, st, ref_action, env.default_params)
            ref_img = ref_obs["image"] if isinstance(ref_obs, Mapping) else ref_obs
            return (r, st), ref_img

        (_, _), x_ref = jax.lax.scan(
            _ref_step, (ref_rng, env_state), None, length=config.n_ref
        )
    else:
        x_ref = None

    # Weight-drift reference: a frozen copy of theta_0 for the ||theta - theta_0||
    # metric.  Deliberately independent of `use_l2_init` (which only allocates
    # `initial_params` when the mitigation is on) so drift is measurable for the
    # vanilla agent too -- enabling an apples-to-apples vanilla-vs-w0-reg compare.
    # Closed over by experiment_step like `x_ref`; never threaded through carry.
    if config.compute_weight_drift:
        drift_w0 = jax.tree_util.tree_map(lambda p: p.copy(), network_params)
    else:
        drift_w0 = None

    env_step_state = (
        train_state,
        gymnax_state,
        log_env_state,
        config,
        obs,
        0,
        0,
        0,
        jnp.zeros((hint_dim,)),  # hint_trace
        rng,
        init_hstate,
    )

    @scan_tqdm(
        config.num_updates, print_rate=max(1, min(100, config.num_updates // 20))
    )
    def experiment_step(carry, iteration_idx):
        env_step_state, train_state, rng, initial_params, l2_init_multipliers, spectral_reg_multipliers, pers_ema = carry
        (
            train_state,
            gymnax_state,
            log_env_state,
            config,
            last_obs,
            last_action,
            last_reward,
            reward_trace,
            hint_trace,
            rng,
            hstate,
        ) = env_step_state
        start_timestep = log_env_state.timestep

        updates_per_video = (
            config.video_length + config.rollout_steps - 1
        ) // config.rollout_steps
        start_recording_update = config.num_updates - updates_per_video
        to_render = (iteration_idx >= start_recording_update) & config.allocate_frames

        gymnax_state = GymnaxEnvState.create(
            to_render=to_render,
            cond_render=gymnax_state.cond_render,
            env_step=gymnax_state.env_step,
            env_params=gymnax_state.env_params,
            env_state=gymnax_state.env_state,
        )

        env_step_state = (
            train_state,
            gymnax_state,
            log_env_state,
            config,
            last_obs,
            last_action,
            last_reward,
            reward_trace,
            hint_trace,
            rng,
            hstate,
        )

        # Roll out for config.rollout_steps
        env_step_state, traj_hstate_batch = jax.lax.scan(
            env_step, env_step_state, length=config.rollout_steps
        )
        traj_batch, hstate_batch = traj_hstate_batch
        (
            train_state,
            gymnax_state,
            log_env_state,
            config,
            last_obs,
            last_action,
            last_reward,
            reward_trace,
            hint_trace,
            rng,
            last_hstate,
        ) = env_step_state

        # Build last observation with previous action encoding
        action_encoded = jnp.zeros((4,))
        action_encoded = action_encoded.at[last_action].set(1)
        last_reward_encoded = jnp.expand_dims(last_reward, 0)

        if isinstance(last_obs, Mapping):
            obs_img = last_obs["image"]
            hint = last_obs["hint"]
            if config.use_hint_trace:
                hint_trace = (
                    config.reward_trace_decay * hint_trace
                    + (1.0 - config.reward_trace_decay) * hint
                )
                hint = hint_trace
            last_reward_encoded = jnp.concatenate((last_reward_encoded, hint), axis=-1)
        else:
            obs_img = last_obs

        sine = jnp.expand_dims(jnp.sin(2 * jnp.pi * log_env_state.timestep / PERIOD), 0)
        cosine = jnp.expand_dims(
            jnp.cos(2 * jnp.pi * log_env_state.timestep / PERIOD), 0
        )
        reward_trace = (
            config.reward_trace_decay * reward_trace
            + (1.0 - config.reward_trace_decay) * last_reward
        )
        reward_trace_encoded = jnp.expand_dims(reward_trace, 0)
        last_obs_encoded = (
            obs_img,
            action_encoded,
            last_reward_encoded,
            sine,
            cosine,
            reward_trace_encoded,
        )

        # Bootstrap value at last state
        _, _, last_value, _ = agent_step(
            last_obs_encoded, train_state, rng, last_hstate
        )
        last_val = last_value.squeeze()

        # Calculate GAE
        advantages, targets = calculate_gae(
            traj_batch, last_val, config.gamma, config.gae_lambda
        )

        # Plasticity-metric forward pass on the just-completed rollout.
        # Uses pre-update params and the rollout's actual init hidden state
        # (`hstate`, the carry from the previous iteration), so the captured
        # activations match what the rollout actually saw. Guarded statically on
        # agent_type; each agent returns a {column_name: scalar} plasticity dict.
        _pp = (train_state.params, train_state.apply_fn, hstate, traj_batch.obs)
        _relu = config.activation in ("relu", "crelu")
        _probe = _should_probe(config)
        _wide = _wide_site_name(config)
        if _probe:
            if _relu:
                # all-ReLU/crelu net (RTU or MLP): dormancy + effective rank at
                # every post-ReLU site. The wide middle layer (RTU output or the
                # MLP's wide Dense) goes into the unified '*_rtu' column; it is
                # omitted entirely for a vanilla MLP with use_middle_layer off.
                plasticity = _compute_plasticity_metrics_relu(
                    *_pp, wide_inter=_wide, dormant_threshold=config.dormant_threshold
                )
            elif _agent_is_rtu(config.agent_type):
                plasticity = _compute_plasticity_metrics(
                    *_pp, dormant_threshold=config.dormant_threshold
                )
            else:
                plasticity = _compute_plasticity_metrics_mlp(
                    *_pp, has_mid=_wide is not None
                )
        else:
            plasticity = _zero_plasticity_metrics()

        # Persistent metric. Per rollout take the per-unit mean (signed tanh for
        # the tanh nets, |activation| for the ReLU nets), EMA it across rollouts
        # (decay persist_decay) in the pers_ema dict carry, then threshold
        # AFTER the EMA: persistent saturation (|EMA| > 0.95) or persistent
        # dormancy (Sokar score <= 0.025). Empty for agents without probes.
        if _probe and _relu:
            means = _mean_abs_act_sites(*_pp, wide_inter=_wide)
            pers_ema = {
                k: config.persist_decay * pers_ema[k]
                + (1.0 - config.persist_decay) * means[k]
                for k in means
            }
            persist = _dormant_persist_from_ema(pers_ema, config.dormant_threshold)
        elif _probe:
            means = _mean_tanh_sites(*_pp)
            pers_ema = {
                k: config.persist_decay * pers_ema[k]
                + (1.0 - config.persist_decay) * means[k]
                for k in means
            }
            persist = _sat_persist_from_ema(pers_ema, config.sat_persist_threshold)
        else:
            persist = {}
        metrics = {**plasticity, **persist}

        # Conditionally perform the update based on how many env steps have elapsed.
        # If freeze_steps <= 0, updates are always performed.
        # Otherwise, once log_env_state.timestep exceeds freeze_steps, we stop updating.
        rng, update_rng = jax.random.split(rng)

        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            hstate_batch,
            advantages,
            targets,
            update_rng,
            config,
            initial_params,
            l2_init_multipliers,
            spectral_reg_multipliers,
        )

        def skip_update(update_state):
            (
                train_state,
                init_hstate,
                traj_batch,
                hstate_batch,
                advantages,
                targets,
                rng,
                config,
                _initial_params,
                _l2_init_multipliers,
                _spectral_reg_multipliers,
            ) = update_state
            return (train_state, rng), (
                _zero_loss_info(config),
                _zero_grad_norms(network_params, config.epochs, config.num_mini_batch),
            )

        should_update = jnp.logical_or(
            config.freeze_steps <= 0,
            log_env_state.timestep <= config.freeze_steps,
        )
        (train_state, rng), (loss_info, grad_norms) = jax.lax.cond(
            should_update, update_step, skip_update, update_state
        )

        # Capture params around the PPO update for per-update churn.  Taken
        # before the (rare) reset / shrink-and-perturb interventions below so
        # churn reflects the gradient update itself, not those resets.
        ntk_params_before = update_state[0].params
        ntk_params_after = train_state.params

        # Periodically reset last layer (actor_mean + critic_value) params and optimizer
        # states.  Guarded at trace time by config.use_reset so no overhead when disabled.
        if config.use_reset:
            should_reset = should_update & _crossed_interval(
                start_timestep, log_env_state.timestep, config.reset_interval
            )
            train_state, rng = jax.lax.cond(
                should_reset,
                _reset_last_layer,
                lambda ts, r: (ts, r),
                train_state,
                rng,
            )

        # Periodically shrink all params and add noise, then reinit optimizer.
        # Guarded at trace time by config.use_shrink_and_perturb.
        if config.use_shrink_and_perturb:
            should_sp = should_update & _crossed_interval(
                start_timestep, log_env_state.timestep, config.sp_interval
            )
            train_state, rng = jax.lax.cond(
                should_sp,
                _shrink_and_perturb,
                lambda ts, r: (ts, r),
                train_state,
                rng,
            )

        # NTK rank / condition number and per-update churn for the value and
        # policy heads, evaluated on the fixed reference batch whenever this
        # update's env-step window crosses a multiple of ntk_freq.  ntk_freq is
        # in *env steps* (matching the DQN ntk_freq and this file's
        # reset_interval / sp_interval), but metrics can only be produced at
        # update boundaries, so the finest achievable spacing is rollout_steps.
        # lax.cond skips the (expensive) Jacobian work on non-metric updates;
        # disabled updates / heads report NaN.  Emitted as per-update scalars.
        if config.compute_ntk:

            def _do_ntk(_):
                assert x_ref is not None  # always an array when compute_ntk is True
                return compute_ppo_metrics(
                    train_state.apply_fn,
                    ntk_params_before,
                    ntk_params_after,
                    init_hstate,
                    x_ref,
                    action_dim,
                    reward_dim,
                    config.chunked_ref,
                    compute_value=True,
                    compute_policy=True,
                )

            is_metric_step = _crossed_interval(
                start_timestep, log_env_state.timestep, config.ntk_freq
            )
            ntk_metrics = jax.lax.cond(
                is_metric_step, _do_ntk, lambda _: nan_ppo_metrics(), operand=None
            )
        else:
            ntk_metrics = nan_ppo_metrics()

        # Weight norm: global L2 norm of the post-update params, gated on its own
        # weight_norm_freq (independent of the NTK cadence above).  Cheap -- no
        # reference batch or Jacobian -- so this is just an L2 reduction over the
        # param tree.  NaN on non-metric updates / when disabled.  Per-update scalar.
        if config.compute_weight_norm:
            is_wn_step = _crossed_interval(
                start_timestep, log_env_state.timestep, config.weight_norm_freq
            )
            weight_norm_metric = jax.lax.cond(
                is_wn_step,
                lambda _: weight_norm(ntk_params_after),
                lambda _: nan_weight_norm(),
                operand=None,
            )
        else:
            weight_norm_metric = nan_weight_norm()

        # RTU spectral radius: mean / max of r = hypot(g, phi) over every RTU
        # diagonal in the post-update params (all blocks, actor + critic).  r is
        # the per-unit pole magnitude, so tau = -1/ln(r) is the memory horizon in
        # steps -- this tracks how long a dependency the recurrence has learned to
        # hold.  Pure elementwise reduction over two small leaves per RTU, so it
        # is cheaper than the weight norm.  NaN pair when disabled / off-cadence.
        if config.compute_r_stats:
            is_r_step = _crossed_interval(
                start_timestep, log_env_state.timestep, config.r_stats_freq
            )
            r_mean_metric, r_max_metric = jax.lax.cond(
                is_r_step,
                lambda _: rtu_r_stats(ntk_params_after),
                lambda _: nan_rtu_r_stats(),
                operand=None,
            )
        else:
            r_mean_metric, r_max_metric = nan_rtu_r_stats()

        # Weight drift: ||theta - theta_0|| (split pi / vf / total), gated on its
        # own weight_drift_freq.  Like the weight norm it is a pure L2 reduction
        # over the post-update params -- here against the frozen theta_0 closure
        # `drift_w0` -- so no reference batch / Jacobian.  NaN triple on
        # non-metric updates / when disabled.  Per-update scalars.
        if config.compute_weight_drift:
            is_wd_step = _crossed_interval(
                start_timestep, log_env_state.timestep, config.weight_drift_freq
            )
            weight_drift_metric = jax.lax.cond(
                is_wd_step,
                lambda _: weight_drift(ntk_params_after, drift_w0, labels),
                lambda _: nan_weight_drift(),
                operand=None,
            )
        else:
            weight_drift_metric = nan_weight_drift()

        # Collect a scalar reward summary for this iteration (mean reward over rollout)
        rewards = traj_batch.reward
        pos = traj_batch.info["pos"]
        biome_id = traj_batch.info["biome_id"]
        object_collected_id = traj_batch.info["object_collected_id"]
        biome_regret = traj_batch.info["biome_regret"]
        biome_rank = traj_batch.info["biome_rank"]
        if config.allocate_frames:
            # Update the frames buffer
            def update_frames(frames):
                idx_in_video = iteration_idx - start_recording_update
                start_idx = idx_in_video * config.rollout_steps
                return jax.lax.dynamic_update_slice(
                    frames, traj_batch.info["frame"], (start_idx, 0, 0, 0)
                )

            frames = jax.lax.cond(
                to_render, update_frames, lambda x: x, log_env_state.frames
            )
        else:
            frames = log_env_state.frames
        log_env_state = LogEnvState(
            returned_returns=log_env_state.returned_returns,
            timestep=log_env_state.timestep,
            frames=frames,
        )

        # Rebuild env_step_state for next iteration
        env_step_state = (
            train_state,
            gymnax_state,
            log_env_state,
            config,
            last_obs,
            last_action,
            last_reward,
            reward_trace,
            hint_trace,
            rng,
            last_hstate,
        )

        # Optional lightweight debug
        carry_out = (env_step_state, train_state, rng, initial_params, l2_init_multipliers, spectral_reg_multipliers, pers_ema)
        return carry_out, (
            rewards,
            pos,
            loss_info,
            biome_id,
            object_collected_id,
            biome_regret,
            biome_rank,
            metrics,
            grad_norms,
            ntk_metrics,
            weight_norm_metric,
            (r_mean_metric, r_max_metric),
            weight_drift_metric,
        )

    # Run training loop with lax.scan (collect per-iteration rewards)
    pers_ema_init = _pers_ema_init(config)
    init_carry = (env_step_state, train_state, rng, initial_params, l2_init_multipliers, spectral_reg_multipliers, pers_ema_init)
    last_carry, info = jax.lax.scan(
        experiment_step,
        PBar(id=config.id, carry=init_carry),
        xs=jnp.arange(int(config.num_updates)),
    )
    (
        rewards,
        pos,
        loss_info,
        biome_id,
        object_collected_id,
        biome_regret,
        biome_rank,
        metrics,
        grad_norms,
        ntk_metrics,
        weight_norm_metric,
        (r_mean_metric, r_max_metric),
        weight_drift_metric,
    ) = info
    rewards = rewards.reshape((-1))
    pos = pos.reshape((-1, pos.shape[-1]))
    total_loss = jnp.mean(loss_info[0], axis=(-1, -2))
    value_loss = jnp.mean(loss_info[1][0], axis=(-1, -2))
    policy_loss = jnp.mean(loss_info[1][1], axis=(-1, -2))
    entropy = jnp.mean(loss_info[1][2], axis=(-1, -2))
    # Average each per-layer norm over the rollout's (epochs, num_mini_batch)
    # updates -> one value per layer per norm per rollout.
    grad_norms = jax.tree_util.tree_map(
        lambda x: jnp.mean(x, axis=(-1, -2)), grad_norms
    )
    biome_id = biome_id.reshape((-1))
    object_collected_id = object_collected_id.reshape((-1))
    biome_regret = biome_regret.reshape((-1))
    biome_rank = biome_rank.reshape((-1))
    # metrics is a dict {column_name: (num_updates,) array} of plasticity
    # metrics (eff_rank / saturation / dormancy / persistent variants), the set
    # of keys depending on the agent.
    # Per-update NTK / churn metrics, one scalar per update (NaN on non-metric
    # updates).  Kept at per-update resolution; consumers subsample as needed.
    (
        value_ntk_rank,
        value_ntk_eff_rank,
        value_ntk_cond,
        value_churn,
        policy_ntk_rank,
        policy_ntk_eff_rank,
        policy_ntk_cond,
        policy_churn,
    ) = ntk_metrics
    weight_drift_pi, weight_drift_vf, weight_drift_total = weight_drift_metric
    env_step_state = last_carry.carry[0]
    frames = env_step_state[2].frames
    return (
        rewards,
        pos,
        (total_loss, (value_loss, policy_loss, entropy)),
        biome_id,
        object_collected_id,
        biome_regret,
        biome_rank,
        metrics,
        (
            value_ntk_rank,
            value_ntk_eff_rank,
            value_ntk_cond,
            value_churn,
            policy_ntk_rank,
            policy_ntk_eff_rank,
            policy_ntk_cond,
            policy_churn,
        ),
        weight_norm_metric,
        (r_mean_metric, r_max_metric),
        (weight_drift_pi, weight_drift_vf, weight_drift_total),
        frames,
        grad_norms,
        grad_nparams,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", type=str, required=True)
    parser.add_argument("-i", "--idxs", nargs="+", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/")
    parser.add_argument("--silent", action="store_true", default=False)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--max_steps", type=int, default=None)

    args = parser.parse_args()

    if not args.gpu:
        jax.config.update("jax_platform_name", "cpu")

    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("filelock").setLevel(logging.ERROR)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("jax").setLevel(logging.WARNING)
    logger = logging.getLogger("exp")
    prod = "cdr" in socket.gethostname() or args.silent
    if not prod:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # ----------------------
    # -- Experiment Def'n --
    # ----------------------
    timeout_handler = TimeoutHandler()

    exp = ExperimentModel.load(args.exp)

    try:
        indices = parse_indices(args.idxs, exp.numPermutations())
    except ValueError as e:
        parser.error(str(e))
    allocate_frames = len(indices) == 1

    # --------------------
    # -- Batch Set-up --
    # --------------------
    start_time = time.time()

    collectors = []
    rngs = []
    chks = []
    configs = []
    for idx in indices:
        chk = Checkpoint(exp, idx, base_path=args.checkpoint_path)
        chk.load_if_exists()
        timeout_handler.before_cancel(chk.save)
        chks.append(chk)

        collector = chk.build(
            "collector",
            lambda: Collector(
                # specify which keys to actually store and ultimately save
                # Options are:
                #  - Identity() (save everything)
                #  - Window(n)  take a window average of size n
                #  - Subsample(n) save one of every n elements
                config={
                    "ewm_reward": Pipe(
                        MovingAverage(0.999),
                        Subsample(max(exp.total_steps // 1000, 1)),
                    ),
                    "mean_ewm_reward": Last(
                        MovingAverage(0.999),
                        Mean(),
                    ),
                },
                # by default, ignore keys that are not explicitly listed above
                default=Ignore(),
            ),
        )
        collector.set_experiment_id(idx)
        collectors.append(collector)

        hypers = exp.get_hypers(idx)

        seed = exp.getRun(idx) + hypers.get("seed_offset", 0)
        rng = jax.random.PRNGKey(seed)

        freeze_steps = hypers.get("freeze_after_steps", hypers.get("freeze_steps", -1))
        rngs.append(rng)

        # derive num_updates if not explicitly present
        num_updates = (
            int(hypers["num_updates"])
            if "num_updates" in hypers
            else (exp.total_steps // int(hypers["rollout_steps"]) + 1)
        )
        if args.max_steps is not None:
            num_updates = args.max_steps
        reset_interval = hypers.get("reset_interval", hypers.get("reset_steps", -1))
        sp_interval = hypers.get("sp_interval", hypers.get("sp_steps", -1))
        # NTK / churn metrics: only computed when experiment.ntk_freq is set.
        # ntk_freq is in env steps (like the DQN ntk_freq), but metrics can only
        # be emitted at update boundaries, so the effective spacing is rounded
        # up to a multiple of rollout_steps.
        ntk_freq = int(hypers.get("experiment", {}).get("ntk_freq", 0))
        compute_ntk = ntk_freq > 0
        n_ref = int(hypers.get("experiment", {}).get("x_ref_steps", 128))
        # Row-chunk size for the memory-bounded NTK Gram build; result-invariant,
        # only trades peak memory against recompute.  Defaults to n_ref // 4.
        chunked_ref = int(
            hypers.get("experiment", {}).get("chunked_ref", max(n_ref // 4, 1))
        )
        chunked_ref = max(min(chunked_ref, n_ref), 1)
        # Weight norm: independent of the NTK metrics, controlled by its own
        # experiment.weight_norm_freq (env steps, rounded up to rollout_steps).
        weight_norm_freq = int(hypers.get("experiment", {}).get("weight_norm_freq", 0))
        compute_weight_norm = weight_norm_freq > 0
        # experiment.r_stats_freq (env steps, rounded up to rollout_steps).
        r_stats_freq = int(hypers.get("experiment", {}).get("r_stats_freq", 0))
        compute_r_stats = r_stats_freq > 0
        # Weight drift (||theta - theta_0||): independent of the NTK / weight-norm
        # metrics, controlled by its own experiment.weight_drift_freq (env steps,
        # rounded up to rollout_steps).
        weight_drift_freq = int(
            hypers.get("experiment", {}).get("weight_drift_freq", 0)
        )
        compute_weight_drift = weight_drift_freq > 0
        activation = str(
            hypers.get("representation", {}).get("activation", "tanh")
        ).lower()
        if "crelu" in exp.agent.lower():
            activation = "crelu"
        config = TrainConfig(
            d_hidden=int(hypers["representation"]["d_hidden"]),
            agent_type=exp.agent,
            hidden_size=int(hypers["representation"]["hidden"]),
            activation=activation,
            rollout_steps=int(hypers["rollout_steps"]),
            epochs=int(hypers["epochs"]),
            num_mini_batch=int(hypers["num_mini_batch"]),
            gradient_clipping=bool(hypers["gradient_clipping"]),
            max_grad_norm=float(hypers["max_grad_norm"]),
            alpha_pi=float(hypers["optimizer_actor"]["alpha"]),
            alpha_vf=float(
                hypers["optimizer_critic"].get(
                    "alpha",
                    hypers["optimizer_critic"].get("lr_scale", jnp.nan)
                    * hypers["optimizer_actor"]["alpha"],
                )
            ),
            adam_eps_pi=float(hypers["optimizer_actor"]["eps"]),
            adam_eps_vf=float(hypers["optimizer_critic"]["eps"]),
            adam_b1_pi=float(hypers["optimizer_actor"].get("beta1", 0.9)),
            adam_b2_pi=float(hypers["optimizer_actor"].get("beta2", 0.999)),
            adam_b1_vf=float(hypers["optimizer_critic"].get("beta1", 0.9)),
            adam_b2_vf=float(hypers["optimizer_critic"].get("beta2", 0.999)),
            l2_reg_pi=float(hypers.get("l2_reg_pi", hypers.get("l2_reg", 0.0))),
            l2_reg_vf=float(hypers.get("l2_reg_vf", hypers.get("l2_reg", 0.0))),
            lambda_l2_init_pi=float(
                hypers.get("lambda_l2_init_pi", hypers.get("lambda_l2_init", 0.0))
            ),
            lambda_l2_init_vf=float(
                hypers.get("lambda_l2_init_vf", hypers.get("lambda_l2_init", 0.0))
            ),
            use_l2_init=bool(
                hypers.get("lambda_l2_init_pi", hypers.get("lambda_l2_init", 0.0))
                != 0.0
                or hypers.get("lambda_l2_init_vf", hypers.get("lambda_l2_init", 0.0))
                != 0.0
            ),
            lambda_spectral_pi=float(
                hypers.get("lambda_spectral_pi", hypers.get("lambda_spectral", 0.0))
            ),
            lambda_spectral_vf=float(
                hypers.get("lambda_spectral_vf", hypers.get("lambda_spectral", 0.0))
            ),
            use_spectral_reg=bool(
                hypers.get("lambda_spectral_pi", hypers.get("lambda_spectral", 0.0))
                != 0.0
                or hypers.get("lambda_spectral_vf", hypers.get("lambda_spectral", 0.0))
                != 0.0
            ),
            sparsity=hypers["representation"].get("sparsity", None),
            spectral_radius=hypers["representation"].get("spectral_radius", None),
            use_sinusoidal_encoding=bool(hypers.get("use_sinusoidal_encoding", False)),
            use_reward_trace=bool(
                hypers.get(
                    "use_reward_trace",
                    hypers.get("representation", {}).get("use_reward_trace", False),
                )
            ),
            use_hint_trace=bool("_HT" in exp.agent),
            use_layernorm=bool(
                hypers.get(
                    "use_layernorm",
                    hypers.get("representation", {}).get("use_layernorm", False),
                )
            ),
            use_middle_layer=bool(
                hypers.get("representation", {}).get("use_middle_layer", True)
            ),
            n_blocks=int(hypers.get("representation", {}).get("n_blocks", 2)),
            use_gating=bool(
                hypers.get("representation", {}).get("use_gating", False)
            ),
            conv=str(hypers.get("representation", {}).get("conv", "Conv2D")),
            reward_trace_decay=float(
                hypers.get(
                    "reward_trace_decay",
                    hypers.get("representation", {}).get("reward_trace_decay", 1.0),
                )
            ),
            persist_decay=float(
                hypers.get("experiment", {}).get("persist_decay", 0.99)
            ),
            sat_persist_threshold=float(
                hypers.get("experiment", {}).get("sat_persist_threshold", 0.95)
            ),
            dormant_threshold=float(
                hypers.get("experiment", {}).get("dormant_threshold", 0.025)
            ),
            num_updates=num_updates,
            aperture_size=int(hypers["environment"]["aperture_size"]),
            render_mode=hypers["environment"].get("render_mode", "world_reward"),
            env_kwargs=tuple(
                sorted(
                    (k, v)
                    for k, v in hypers["environment"].items()
                    if k not in ["aperture_size", "env_id", "render_mode"]
                    and v is not None
                )
            ),
            env_id=hypers["environment"]["env_id"],
            gamma=float(hypers["gamma"]),
            gae_lambda=float(hypers["gae_lambda"]),
            clip_eps=float(hypers["clip_eps"]),
            vf_coef=float(hypers["vf_coef"]),
            entropy_coef=float(hypers["entropy_coef"]),
            id=idx,
            freeze_steps=int(freeze_steps),
            allocate_frames=allocate_frames,
            video_length=int(hypers.get("experiment", {}).get("video_length", 1000)),
            use_reset=bool(reset_interval > 0),
            reset_interval=int(reset_interval),
            use_shrink_and_perturb=bool(sp_interval > 0),
            sp_interval=int(sp_interval),
            shrink_factor=float(hypers.get("shrink_factor", 1.0)),
            noise_scale=float(hypers.get("noise_scale", 0.0)),
            compute_ntk=compute_ntk,
            ntk_freq=max(ntk_freq, 1),
            n_ref=n_ref,
            chunked_ref=chunked_ref,
            compute_weight_norm=compute_weight_norm,
            weight_norm_freq=max(weight_norm_freq, 1),
            compute_r_stats=compute_r_stats,
            r_stats_freq=max(r_stats_freq, 1),
            compute_weight_drift=compute_weight_drift,
            weight_drift_freq=max(weight_drift_freq, 1),
            compute_plasticity=bool(
                hypers.get("experiment", {}).get("compute_plasticity", False)
            ),
        )
        configs.append(config)

    batch_experiment = jax.vmap(experiment, in_axes=(0, 0))
    rngs = jnp.stack(rngs)
    configs_stacked = tree_map(lambda *xs: jnp.stack(xs), *configs)
    results = batch_experiment(rngs, configs_stacked)
    (
        rewards,
        pos,
        (total_loss, (value_loss, policy_loss, entropy)),
        biome_id,
        object_collected_id,
        biome_regret,
        biome_rank,
        metrics,
        (
            value_ntk_rank,
            value_ntk_eff_rank,
            value_ntk_cond,
            value_churn,
            policy_ntk_rank,
            policy_ntk_eff_rank,
            policy_ntk_cond,
            policy_churn,
        ),
        weight_norm_metric,
        (r_mean_metric, r_max_metric),
        (weight_drift_pi, weight_drift_vf, weight_drift_total),
        frames,
        grad_norms,
        grad_nparams,
    ) = results
    # metrics: dict {column_name: (num_runs, num_updates) array}. Keys depend on
    # the agent (eff_rank / sat_rate / dormant / *_persist); saved per run below.

    # --------------------
    # -- Saving --
    # --------------------
    total_collect_time = 0
    total_numpy_time = 0
    total_db_time = 0
    num_indices = len(indices)
    for i, idx in enumerate(indices):
        collector = collectors[i]
        chk = chks[i]
        config = configs[i]
        # process rewards for this run
        run_rewards = rewards[i]
        run_pos = pos[i]
        run_total_loss = total_loss[i]
        run_value_loss = value_loss[i]
        run_policy_loss = policy_loss[i]
        run_entropy = entropy[i]
        run_biome_id = biome_id[i]
        run_object_collected_id = object_collected_id[i]
        run_biome_regret = biome_regret[i]
        run_biome_rank = biome_rank[i]
        # All plasticity columns for this run, keyed by name (agent-dependent).
        run_metrics = {k: v[i] for k, v in metrics.items()}
        run_value_ntk_rank = value_ntk_rank[i]
        run_value_ntk_eff_rank = value_ntk_eff_rank[i]
        run_value_ntk_cond = value_ntk_cond[i]
        run_value_churn = value_churn[i]
        run_policy_ntk_rank = policy_ntk_rank[i]
        run_policy_ntk_eff_rank = policy_ntk_eff_rank[i]
        run_policy_ntk_cond = policy_ntk_cond[i]
        run_policy_churn = policy_churn[i]
        run_weight_norm = weight_norm_metric[i]
        run_r_mean = r_mean_metric[i]
        run_r_max = r_max_metric[i]
        run_weight_drift_pi = weight_drift_pi[i]
        run_weight_drift_vf = weight_drift_vf[i]
        run_weight_drift_total = weight_drift_total[i]
        run_frames = frames[i]
        start_time = time.time()
        # for reward in run_rewards:
        #     collector.next_frame()
        #     collector.collect("ewm_reward", reward.item())
        #     collector.collect("mean_ewm_reward", reward.item())
        logger.debug(f"Mean rewards {run_rewards.mean()}")
        collector.reset()
        total_collect_time += time.time() - start_time

        # ------------
        # -- Saving --
        # ------------
        context = exp.buildSaveContext(idx, base=args.save_path)
        save_path = context.resolve("results.db")
        data_path = context.resolve(f"data/{idx}.npz")
        video_path = context.resolve(f"videos/{idx}")
        context.ensureExists(data_path, is_file=True)
        context.ensureExists(video_path, is_file=True)

        # Per-layer gradient norms for this run: one time series per
        # (norm, layer), plus the constant per-layer parameter count for
        # offline param-count weighting.
        run_grad_norm_kwargs = {}
        for layer, (l0, l1, l2) in grad_norms.items():
            run_grad_norm_kwargs[f"grad_l0_{layer}"] = l0[i]
            run_grad_norm_kwargs[f"grad_l1_{layer}"] = l1[i]
            run_grad_norm_kwargs[f"grad_l2_{layer}"] = l2[i]
            # 1-d (not 0-d scalar) so the generic npz->DataFrame loader, which
            # reads v.shape[0] on every key, doesn't choke.
            run_grad_norm_kwargs[f"grad_nparams_{layer}"] = np.atleast_1d(
                np.asarray(grad_nparams[layer][i])
            )

        start_time = time.time()
        if config.allocate_frames:
            start_frame = (
                config.num_updates * config.rollout_steps - run_frames.shape[0]
            )
            end_frame = config.num_updates * config.rollout_steps
            save_video(
                list(run_frames),
                video_path,
                name_prefix=f"{start_frame}_{end_frame}",
                fps=8,
            )
        np.savez_compressed(
            data_path,
            rewards=run_rewards,
            pos=run_pos,
            total_loss=run_total_loss,
            value_loss=run_value_loss,
            policy_loss=run_policy_loss,
            entropy=run_entropy,
            biome_id=run_biome_id,
            object_collected_id=run_object_collected_id,
            biome_regret=run_biome_regret,
            biome_rank=run_biome_rank,
            **run_metrics,
            value_ntk_rank=run_value_ntk_rank,
            value_ntk_eff_rank=run_value_ntk_eff_rank,
            value_ntk_cond=run_value_ntk_cond,
            value_churn=run_value_churn,
            policy_ntk_rank=run_policy_ntk_rank,
            policy_ntk_eff_rank=run_policy_ntk_eff_rank,
            policy_ntk_cond=run_policy_ntk_cond,
            policy_churn=run_policy_churn,
            weight_norm=run_weight_norm,
            rtu_r_mean=run_r_mean,
            rtu_r_max=run_r_max,
            weight_drift_pi=run_weight_drift_pi,
            weight_drift_vf=run_weight_drift_vf,
            weight_drift_total=run_weight_drift_total,
            **run_grad_norm_kwargs,
        )
        total_numpy_time += time.time() - start_time

        meta = getParamsAsDict(exp, idx)
        meta |= {"seed": exp.getRun(idx)}
        attach_metadata(save_path, idx, meta)

        start_time = time.time()
        collector.merge(context.resolve("results.db"))
        total_db_time += time.time() - start_time

        collector.close()
        chk.delete()
    logger.debug("--- Saving Timings ---")
    logger.debug(
        f"Total collect time: {total_collect_time:.4f}s | Average: {total_collect_time / num_indices:.4f}s"
    )
    logger.debug(
        f"Total numpy save time: {total_numpy_time:.4f}s | Average: {total_numpy_time / num_indices:.4f}s"
    )
    logger.debug(
        f"Total db save time: {total_db_time:.4f}s | Average: {total_db_time / num_indices:.4f}s"
    )
    total_save_time = total_collect_time + total_numpy_time + total_db_time
    logger.debug(
        f"Total save time: {total_save_time:.4f}s | Average: {total_save_time / num_indices:.4f}s"
    )


if __name__ == "__main__":
    main()
