"""Tests for the T-BPTT actor-critic agents (BPTTActorCriticConv / MLP).

The properties that matter are:
  1. the module accepts the (seq_len, seq_batch, ...) shape contract;
  2. unrolling a sequence in one call agrees with stepping it one timestep at a
     time from the same carry -- this is what lets the rollout (which acts
     step-by-step) and the update (which unrolls chunks) see the same network;
  3. gradients actually cross timesteps inside a chunk, and stop at its start.
"""

import jax
import jax.numpy as jnp
import pytest

from algorithms.nn.BPTTACConv import BPTTActorCriticConv
from algorithms.nn.BPTTACMLP import BPTTActorCriticMLP

ACTION_DIM = 4
D_HIDDEN = 8
HIDDEN_SIZE = 6
OBS_SHAPE = (5, 5, 3)


def _obs(T, B, key):
    """An obs tuple with the (seq_len, seq_batch, ...) leading axes."""
    keys = jax.random.split(key, 6)
    return (
        jax.random.normal(keys[0], (T, B, *OBS_SHAPE)),
        jax.random.normal(keys[1], (T, B, ACTION_DIM)),
        jax.random.normal(keys[2], (T, B, 1)),
        jax.random.normal(keys[3], (T, B, 1)),
        jax.random.normal(keys[4], (T, B, 1)),
        jax.random.normal(keys[5], (T, B, 1)),
    )


def _build(cls, rtu_type, **kwargs):
    return cls(
        action_dim=ACTION_DIM,
        d_hidden=D_HIDDEN,
        hidden_size=HIDDEN_SIZE,
        activation="tanh",
        rtu_type=rtu_type,
        **kwargs,
    )


AGENTS = [
    ("conv-linear", BPTTActorCriticConv, "linear_rtu"),
    ("conv-nonlinear", BPTTActorCriticConv, "non_linear_rtu"),
    ("mlp-linear", BPTTActorCriticMLP, "linear_rtu"),
    ("mlp-nonlinear", BPTTActorCriticMLP, "non_linear_rtu"),
]


class TestShapes:
    @pytest.mark.parametrize("name,cls,rtu_type", AGENTS)
    def test_forward_shapes(self, name, cls, rtu_type):
        T, B = 7, 3
        net = _build(cls, rtu_type)
        carry = cls.initialize_memory(B, D_HIDDEN)
        obs = _obs(T, B, jax.random.PRNGKey(0))
        params = net.init(jax.random.PRNGKey(1), carry, obs)

        new_carry, pi, value = net.apply(params, carry, obs)

        assert value.shape == (T, B)
        assert pi.logits.shape == (T, B, ACTION_DIM)
        # The returned carry is the state at the sequence's LAST step: one per
        # sequence, with no time axis.
        for branch in new_carry:
            for leaf in branch:
                assert leaf.shape == (B, D_HIDDEN)

    @pytest.mark.parametrize("name,cls,rtu_type", AGENTS)
    def test_initialize_memory_ignores_d_input(self, name, cls, rtu_type):
        """No RTRL sensitivity carry, so d_input is accepted and ignored -- this
        is what keeps the rtu_ppo call site shared with the RealTime* classes."""
        a = cls.initialize_memory(2, D_HIDDEN)
        b = cls.initialize_memory(2, D_HIDDEN, 37)
        assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
        for x, y in zip(
            jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True
        ):
            assert x.shape == y.shape


class TestUnrollMatchesStepping:
    """A chunk unrolled in one call must equal the same steps taken one at a
    time, since acting steps (T=1) and updating unrolls (T=seq_len) over the
    identical parameters. Any mismatch means the fold/unfold or the carry
    threading is wrong."""

    @pytest.mark.parametrize("name,cls,rtu_type", AGENTS)
    def test_sequential_matches_unrolled(self, name, cls, rtu_type):
        T, B = 6, 1
        net = _build(cls, rtu_type)
        carry0 = cls.initialize_memory(B, D_HIDDEN)
        obs = _obs(T, B, jax.random.PRNGKey(2))
        params = net.init(jax.random.PRNGKey(3), carry0, obs)

        _, pi_full, v_full = net.apply(params, carry0, obs)

        carry = carry0
        stepped_v, stepped_logits = [], []
        for t in range(T):
            step_obs = tuple(x[t : t + 1] for x in obs)  # (1, B, ...)
            carry, pi_t, v_t = net.apply(params, carry, step_obs)
            stepped_v.append(v_t)
            stepped_logits.append(pi_t.logits)

        assert jnp.allclose(
            jnp.concatenate(stepped_v, axis=0), v_full, atol=1e-5
        )
        assert jnp.allclose(
            jnp.concatenate(stepped_logits, axis=0), pi_full.logits, atol=1e-5
        )


class TestGradientFlow:
    @pytest.mark.parametrize("name,cls,rtu_type", AGENTS)
    def test_gradient_crosses_timesteps(self, name, cls, rtu_type):
        """The value at the last step must depend on the observation at the
        first -- that dependence is the whole point of T-BPTT, and it is exactly
        what the real-time cell's stop_gradient removes."""
        T, B = 5, 1
        net = _build(cls, rtu_type)
        carry = cls.initialize_memory(B, D_HIDDEN)
        obs = _obs(T, B, jax.random.PRNGKey(4))
        params = net.init(jax.random.PRNGKey(5), carry, obs)

        def last_value(obs_img):
            full = (obs_img,) + obs[1:]
            _, _, value = net.apply(params, carry, full)
            return value[-1].sum()

        g = jax.grad(last_value)(obs[0])
        assert jnp.abs(g[0]).sum() > 0.0, "no gradient reached the first timestep"

    @pytest.mark.parametrize("name,cls,rtu_type", AGENTS)
    def test_gradient_stops_at_chunk_start(self, name, cls, rtu_type):
        """Splitting a chunk in two must change the parameter gradient: the
        second half no longer receives credit through the first. If these agreed,
        the truncation boundary would not exist."""
        T, B = 6, 1
        net = _build(cls, rtu_type)
        carry0 = cls.initialize_memory(B, D_HIDDEN)
        obs = _obs(T, B, jax.random.PRNGKey(6))
        params = net.init(jax.random.PRNGKey(7), carry0, obs)

        # Touch both branches: `value` only flows through the critic RTU, so a
        # value-only loss would leave the actor RTU at zero gradient in both
        # arms and the comparison would be vacuous.
        def _objective(pi, value):
            return (value**2).sum() + (pi.logits**2).sum()

        def loss_whole(p):
            _, pi, value = net.apply(p, carry0, obs)
            return _objective(pi, value)

        def loss_split(p):
            # Detach at the midpoint, exactly as storing the carry does.
            mid_carry, pi1, v1 = net.apply(p, carry0, tuple(x[: T // 2] for x in obs))
            mid_carry = jax.lax.stop_gradient(mid_carry)
            _, pi2, v2 = net.apply(p, mid_carry, tuple(x[T // 2 :] for x in obs))
            return _objective(pi1, v1) + _objective(pi2, v2)

        g_whole = jax.grad(loss_whole)(params)
        g_split = jax.grad(loss_split)(params)

        # Forward values are identical either way; only the gradients differ.
        assert jnp.allclose(loss_whole(params), loss_split(params), atol=1e-5)

        for branch in ("actor_rtu", "critic_rtu"):
            diffs = [
                float(jnp.abs(a - b).max())
                for a, b in zip(
                    jax.tree_util.tree_leaves(g_whole["params"][branch]),
                    jax.tree_util.tree_leaves(g_split["params"][branch]),
                    strict=True,
                )
            ]
            assert max(diffs) > 1e-8, f"truncation had no effect on {branch}"
