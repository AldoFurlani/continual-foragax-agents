# T-BPTT counterpart of RealTimeActorCriticMLPStacked, with the conv vision
# front-end from BPTTActorCriticConv.
#
# Backbone is the LRU/S5 SequenceLayer of Orvieto et al. (2023), with RTUs in
# place of the LRU and the recurrence unrolled over a time axis (BPTTLRTUs /
# BPTTNonLRTUs) instead of stepped by the real-time RTRL cell.
#
# Block layout per branch (actor and critic are separate stacks):
#
#   x = Dense(embed of [conv_emb, last_action, last_reward...])   (stream, width W)
#   repeat n_blocks times:
#     y = RTU(x)                       # temporal mixing, output width 2*d_hidden
#     y = MLP(act(y))                  # channel mixing, back down to width W
#     x = x + y                        # ONE skip: RTU input -> MLP output
#     x = LayerNorm(x)
#   heads(LayerNorm(x))
#
# The single residual is the one structural difference from
# RealTimeActorCriticMLPStacked, which uses the transformer convention of two
# pre-norm residuals (one around the RTU sublayer, one around the MLP).  LRU's
# SequenceLayer instead wraps a single skip around the whole [recurrence -> MLP]
# block, which is what this reproduces.
#
# Why T-BPTT rather than the real-time cell: RealTimeLinearRTUs stop-gradients
# h_{t-1} and repairs it with an exact per-layer RTRL trace, but that trace only
# covers the cell's own parameters.  Stacking such cells therefore drops
# d(h^(l+1)_t)/d(theta^(l)) -- the lower layer never receives credit for how its
# output persists in the layer above.  Unrolling with nn.scan keeps every
# time-by-depth path inside the truncation window instead.
#
# Shape contract: obs arrays carry leading axes (seq_len, seq_batch, ...) and
# `hidden` holds each block's recurrent state at the sequence's FIRST step,
# shaped (seq_batch, d_hidden).
import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from algorithms.nn.activations import get_activation
from algorithms.nn.rtus.rtus import BPTTLRTUs, BPTTNonLRTUs


class BPTTActorCriticConvStacked(nn.Module):
    action_dim: int
    d_hidden: int = 192  # RTU units per block (RTU output is 2*d_hidden)
    hidden_size: int = 64  # residual stream width W
    n_blocks: int = 2
    activation: str = "tanh"
    cont: bool = False
    rtu_type: str = "linear_rtu"
    use_sinusoidal_encoding: bool = False
    use_reward_trace: bool = False
    use_layernorm: bool = True  # gates the conv front-end norms only
    conv: str = "Conv2D"
    mlp_expansion: int = 4
    prenorm: bool = False  # LRU's default is postnorm

    def _conv_stack(self, x, prefix, activation):
        """Vision front-end, shared with BPTTActorCriticConv. Runs on rows that
        already have time folded into the batch axis."""
        if self.conv in ("PConv2D", "PConv2DConv2D"):
            x = nn.Conv(16, 1, 1, kernel_init=orthogonal(np.sqrt(2)),
                        bias_init=constant(0.0), name=f"{prefix}_pconv1")(x)
            if self.use_layernorm:
                x = nn.LayerNorm(epsilon=1e-05, name=f"{prefix}_player_norm1")(x)
            x = activation(x)
        if self.conv in ("Conv2D", "PConv2DConv2D"):
            x = nn.Conv(16, 3, 1, kernel_init=orthogonal(np.sqrt(2)),
                        bias_init=constant(0.0), name=f"{prefix}_conv1")(x)
            if self.use_layernorm:
                x = nn.LayerNorm(epsilon=1e-05, name=f"{prefix}_layernorm1")(x)
            x = activation(x)
        # conv="none": skip conv layers, flatten raw obs directly
        return jnp.reshape(x, (x.shape[0], -1))

    def _block(self, x, carry, seq_model, activation, unfold, fold, prefix, blk):
        """One LRU-style [RTU -> MLP] block with a single residual.

        `x` arrives folded as (seq_len*seq_batch, W); only the RTU sees the time
        axis, so it is unfolded around that call and folded straight back.
        """
        W = self.hidden_size
        skip = x  # the RTU's input -- the source of the block's one skip
        if self.prenorm:
            x = nn.LayerNorm(name=f"{prefix}_layernorm_blk{blk}")(x)

        carry, y = seq_model(
            self.d_hidden,
            params_type="exp_exp",
            activation="relu",
            name=f"{prefix}_rtu{blk}",
        )(carry, unfold(x))
        y = fold(y)
        self.sow("intermediates", f"{prefix}_rtu_out{blk}", y)

        # No activation here: the RTU output is already post-nonlinearity
        # (LinearRTUs/NonLinearRTUs apply act_options[activation] to the
        # concatenated state).  LRU needs its gelu at this point because its
        # recurrence and readout are both linear; RTUs do not, and
        # RealTimeActorCriticMLPStacked likewise goes straight from the RTU into
        # the projection.  Adding one would stack relu and tanh back to back.
        y = nn.Dense(
            self.mlp_expansion * W,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name=f"{prefix}_mlp_up{blk}",
        )(y)
        y = activation(y)
        y = nn.Dense(
            W,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name=f"{prefix}_mlp_down{blk}",
        )(y)

        x = skip + y
        if not self.prenorm:
            x = nn.LayerNorm(name=f"{prefix}_layernorm_blk{blk}")(x)
        return x, carry

    def _branch(self, obs, extras, carries, seq_model, activation, unfold, fold, prefix):
        """Full stack for one branch (actor or critic)."""
        W = self.hidden_size
        emb = self._conv_stack(obs, prefix, activation)
        emb = nn.Dense(
            W,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name=f"{prefix}_dense1",
        )(emb)
        # Unconditional, as in RealTimeActorCriticMLPStacked: the residual
        # stream needs its norms regardless of `use_layernorm`, which here gates
        # only the conv front-end.
        emb = nn.LayerNorm(name=f"{prefix}_layernorm_embed")(emb)
        emb = activation(emb)
        # Fold action/reward into the stream; plays the role of the linear
        # encoder in LRU stacks and fixes every block's RTU input width to W.
        x = nn.Dense(
            W,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name=f"{prefix}_embed",
        )(jnp.concatenate((emb, *extras), axis=-1))

        new_carries = []
        for blk in range(self.n_blocks):
            x, carry = self._block(
                x, carries[blk], seq_model, activation, unfold, fold, prefix, blk
            )
            new_carries.append(carry)
        x = nn.LayerNorm(name=f"{prefix}_layernorm_out")(x)

        head = nn.Dense(
            W,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name=f"{prefix}_dense2",
        )(x)
        head = activation(head)
        return head, tuple(new_carries)

    @nn.compact
    def __call__(self, hidden, obs):
        """
        hidden: (actor_carries, critic_carries), each a tuple of n_blocks RTU
                carries ((seq_batch, d_hidden), (seq_batch, d_hidden))
        obs: ((seq_len, seq_batch, H, W, C), (seq_len, seq_batch, action_dim), ...)

        Returns (final_hidden, pi, value) with pi/value carrying leading
        (seq_len, seq_batch) axes.
        """
        activation = get_activation(self.activation)

        if self.rtu_type == "linear_rtu":
            seq_model = BPTTLRTUs
        elif self.rtu_type == "non_linear_rtu":
            seq_model = BPTTNonLRTUs
        else:
            raise NotImplementedError

        (actor_carries, critic_carries) = hidden
        (obs, last_action_encoded, last_reward, sine, cosine, reward_trace) = obs

        # Everything except the RTUs is per-timestep and order-independent, so
        # fold time into the batch axis and run those layers once over
        # seq_len*seq_batch rows.
        T, B = obs.shape[0], obs.shape[1]

        def fold(x):
            return x.reshape((T * B,) + x.shape[2:])

        def unfold(x):
            return x.reshape((T, B) + x.shape[1:])

        obs = fold(obs)
        last_action_encoded = fold(last_action_encoded)
        last_reward = fold(last_reward)
        sine = fold(sine)
        cosine = fold(cosine)
        reward_trace = fold(reward_trace)

        last_reward_plus = last_reward
        if self.use_sinusoidal_encoding:
            last_reward_plus = jnp.concatenate((last_reward_plus, sine, cosine), axis=-1)
        if self.use_reward_trace:
            last_reward_plus = jnp.concatenate((last_reward_plus, reward_trace), axis=-1)
        extras = (last_action_encoded, last_reward_plus)

        actor_out, actor_carries = self._branch(
            obs, extras, actor_carries, seq_model, activation, unfold, fold, "actor"
        )
        critic_out, critic_carries = self._branch(
            obs, extras, critic_carries, seq_model, activation, unfold, fold, "critic"
        )

        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_mean",
        )(actor_out)
        actor_mean = unfold(actor_mean)
        if self.cont:
            actor_logtstd = self.param(
                "log_std", nn.initializers.zeros, (self.action_dim,)
            )
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_value"
        )(critic_out)
        critic = unfold(jnp.squeeze(critic, axis=-1))

        hidden = (actor_carries, critic_carries)
        return hidden, pi, critic

    @staticmethod
    def initialize_memory(batch_size, d_hidden, d_input=None, n_blocks=2):
        """Recurrent state only -- no RTRL sensitivity carry, so `d_input` is
        accepted and ignored for call-site compatibility with the RealTime*
        classes."""
        del d_input

        def one_block():
            return (
                jnp.zeros((batch_size, d_hidden)),
                jnp.zeros((batch_size, d_hidden)),
            )

        actor_carries = tuple(one_block() for _ in range(n_blocks))
        critic_carries = tuple(one_block() for _ in range(n_blocks))
        return (actor_carries, critic_carries)
