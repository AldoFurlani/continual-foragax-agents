"""Tests for the T-BPTT actor-critic agents (Conv / MLP, plain and Stacked).

The properties that matter are:
  1. the module accepts the (seq_len, seq_batch, ...) shape contract;
  2. unrolling a sequence in one call agrees with stepping it one timestep at a
     time from the same carry -- this is what lets the rollout (which acts
     step-by-step) and the update (which unrolls chunks) see the same network;
  3. gradients actually cross timesteps inside a chunk, and stop at its start;
  4. for the stacked agents, each block's residual runs from the RTU's input to
     the MLP's output (LRU's SequenceLayer convention, one skip per block).
"""

import jax
import jax.numpy as jnp
import pytest
from flax.traverse_util import flatten_dict, unflatten_dict

from algorithms.nn.BPTTACConv import BPTTActorCriticConv
from algorithms.nn.BPTTACConvStacked import BPTTActorCriticConvStacked
from algorithms.nn.BPTTACMLP import BPTTActorCriticMLP
from algorithms.nn.BPTTACMLPStacked import BPTTActorCriticMLPStacked

ACTION_DIM = 4
D_HIDDEN = 8
HIDDEN_SIZE = 6
OBS_SHAPE = (5, 5, 3)
N_BLOCKS = 2

STACKED = (BPTTActorCriticConvStacked, BPTTActorCriticMLPStacked)


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
    if cls in STACKED:
        kwargs.setdefault("n_blocks", N_BLOCKS)
    return cls(
        action_dim=ACTION_DIM,
        d_hidden=D_HIDDEN,
        hidden_size=HIDDEN_SIZE,
        activation="tanh",
        rtu_type=rtu_type,
        **kwargs,
    )


def _memory(cls, batch_size):
    if cls in STACKED:
        return cls.initialize_memory(batch_size, D_HIDDEN, n_blocks=N_BLOCKS)
    return cls.initialize_memory(batch_size, D_HIDDEN)


_STACKED_RTUS = tuple(
    f"{branch}_rtu{blk}" for branch in ("actor", "critic") for blk in range(N_BLOCKS)
)

# (id, class, rtu_type, names of the RTU parameter subtrees)
AGENTS = [
    ("conv-linear", BPTTActorCriticConv, "linear_rtu", ("actor_rtu", "critic_rtu")),
    ("conv-nonlinear", BPTTActorCriticConv, "non_linear_rtu", ("actor_rtu", "critic_rtu")),
    ("mlp-linear", BPTTActorCriticMLP, "linear_rtu", ("actor_rtu", "critic_rtu")),
    ("mlp-nonlinear", BPTTActorCriticMLP, "non_linear_rtu", ("actor_rtu", "critic_rtu")),
    ("stacked-linear", BPTTActorCriticConvStacked, "linear_rtu", _STACKED_RTUS),
    ("stacked-nonlinear", BPTTActorCriticConvStacked, "non_linear_rtu", _STACKED_RTUS),
    ("mlp-stacked-linear", BPTTActorCriticMLPStacked, "linear_rtu", _STACKED_RTUS),
    ("mlp-stacked-nonlinear", BPTTActorCriticMLPStacked, "non_linear_rtu", _STACKED_RTUS),
]


class TestShapes:
    @pytest.mark.parametrize("name,cls,rtu_type,rtu_keys", AGENTS)
    def test_forward_shapes(self, name, cls, rtu_type, rtu_keys):
        T, B = 7, 3
        net = _build(cls, rtu_type)
        carry = _memory(cls, B)
        obs = _obs(T, B, jax.random.PRNGKey(0))
        params = net.init(jax.random.PRNGKey(1), carry, obs)

        new_carry, pi, value = net.apply(params, carry, obs)

        assert value.shape == (T, B)
        assert pi.logits.shape == (T, B, ACTION_DIM)
        # The returned carry is the state at the sequence's LAST step: one per
        # sequence, with no time axis. tree_leaves rather than a fixed nesting
        # depth, since the stacked agent carries one state per block.
        leaves = jax.tree_util.tree_leaves(new_carry)
        assert leaves, "no recurrent state returned"
        for leaf in leaves:
            assert leaf.shape == (B, D_HIDDEN)

    @pytest.mark.parametrize("name,cls,rtu_type,rtu_keys", AGENTS)
    def test_initialize_memory_ignores_d_input(self, name, cls, rtu_type, rtu_keys):
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

    @pytest.mark.parametrize("name,cls,rtu_type,rtu_keys", AGENTS)
    def test_sequential_matches_unrolled(self, name, cls, rtu_type, rtu_keys):
        T, B = 6, 1
        net = _build(cls, rtu_type)
        carry0 = _memory(cls, B)
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
    @pytest.mark.parametrize("name,cls,rtu_type,rtu_keys", AGENTS)
    def test_gradient_crosses_timesteps(self, name, cls, rtu_type, rtu_keys):
        """The value at the last step must depend on the observation at the
        first -- that dependence is the whole point of T-BPTT, and it is exactly
        what the real-time cell's stop_gradient removes."""
        T, B = 5, 1
        net = _build(cls, rtu_type)
        carry = _memory(cls, B)
        obs = _obs(T, B, jax.random.PRNGKey(4))
        params = net.init(jax.random.PRNGKey(5), carry, obs)

        def last_value(obs_img):
            full = (obs_img,) + obs[1:]
            _, _, value = net.apply(params, carry, full)
            return value[-1].sum()

        g = jax.grad(last_value)(obs[0])
        assert jnp.abs(g[0]).sum() > 0.0, "no gradient reached the first timestep"

    @pytest.mark.parametrize("name,cls,rtu_type,rtu_keys", AGENTS)
    def test_gradient_stops_at_chunk_start(self, name, cls, rtu_type, rtu_keys):
        """Splitting a chunk in two must change the parameter gradient: the
        second half no longer receives credit through the first. If these agreed,
        the truncation boundary would not exist."""
        T, B = 6, 1
        net = _build(cls, rtu_type)
        carry0 = _memory(cls, B)
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

        # For the stacked agent this covers rtu0 as well as rtu1: block 0's
        # gradient can only change under truncation if credit crosses the split,
        # which for the lower block means travelling through block 1's state
        # over time -- the term the layer-local RTRL traces drop.
        for branch in rtu_keys:
            diffs = [
                float(jnp.abs(a - b).max())
                for a, b in zip(
                    jax.tree_util.tree_leaves(g_whole["params"][branch]),
                    jax.tree_util.tree_leaves(g_split["params"][branch]),
                    strict=True,
                )
            ]
            assert max(diffs) > 1e-8, f"truncation had no effect on {branch}"


class TestStackedResidualTopology:
    """The stacked agent uses LRU's SequenceLayer convention: ONE skip, from the
    RTU's input to the MLP's output. The transformer convention (a separate
    residual around the RTU sublayer, a second around the MLP) is the thing
    these tests rule out."""

    CASES = [(c, r) for c in STACKED for r in ("linear_rtu", "non_linear_rtu")]

    @pytest.mark.parametrize("cls,rtu_type", CASES)
    def test_no_separate_rtu_residual_projection(self, cls, rtu_type):
        """A two-residual block needs its own projection to bring the RTU output
        back to the stream width. This one does not: the MLP is the only way
        down from 2*d_hidden."""
        net = _build(cls, rtu_type)
        carry = _memory(cls, 1)
        obs = _obs(3, 1, jax.random.PRNGKey(8))
        params = net.init(jax.random.PRNGKey(9), carry, obs)

        names = set(params["params"].keys())
        assert not any("rtu_proj" in n for n in names)
        for blk in range(N_BLOCKS):
            assert f"actor_mlp_up{blk}" in names
            assert f"actor_mlp_down{blk}" in names

    @pytest.mark.parametrize("cls,rtu_type", CASES)
    def test_mlp_is_the_only_path_out_of_the_rtu(self, cls, rtu_type):
        """Zeroing every block's MLP down-projection must sever all temporal
        dependence. With one skip the RTU reaches the stream only through the
        MLP, so killing the MLP leaves a purely feedforward network; with a
        second residual straight off the RTU the temporal path would survive."""
        T, B = 5, 1
        net = _build(cls, rtu_type)
        carry = _memory(cls, B)
        obs = _obs(T, B, jax.random.PRNGKey(10))
        params = net.init(jax.random.PRNGKey(11), carry, obs)

        flat = flatten_dict(params)
        blanked = unflatten_dict(
            {
                # flatten_dict's value type includes _EmptyNode, which pyright
                # cannot pass to zeros_like; a param tree never holds one here.
                k: (
                    jnp.zeros_like(v)  # pyright: ignore[reportArgumentType]
                    if any("mlp_down" in part for part in k)
                    else v
                )
                for k, v in flat.items()
            }
        )

        def last_value(obs_img, p):
            _, _, value = net.apply(p, carry, (obs_img,) + obs[1:])
            return value[-1].sum()

        g_live = jax.grad(last_value)(obs[0], params)
        g_dead = jax.grad(last_value)(obs[0], blanked)

        # Sanity: the intact network does carry credit back to step 0.
        assert jnp.abs(g_live[0]).sum() > 0.0
        # With the MLPs blanked, step 0 is unreachable from the last step.
        assert jnp.abs(g_dead[0]).sum() == 0.0, (
            "temporal path survived the MLP being zeroed -- the RTU has a "
            "residual route to the stream that bypasses the MLP"
        )
