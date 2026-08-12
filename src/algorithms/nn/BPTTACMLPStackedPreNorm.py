# T-BPTT agent using the RTRL stack's block topology (RealTimeActorCriticMLPStacked)
# rather than the LRU SequenceLayer topology of BPTTActorCriticMLPStacked.
#
# Existing agents pair one gradient scheme with one topology, so the two are
# confounded:
#
#   RealTimeActorCriticMLPStacked   RTRL   + transformer-convention block
#   BPTTActorCriticMLPStacked       T-BPTT + LRU SequenceLayer block
#
# This class is the missing cell of that 2x2 -- T-BPTT + transformer-convention
# block -- so the two axes can be separated. It exists because E142's stacked
# T-BPTT agent collapsed on 27-29/30 seeds on ForagaxSquareWaveTwoBiome-v11 while
# E140's stacked RTRL agent, at the same depth, collapsed on 18/30. Since the
# T-BPTT one uses the *better* gradient scheme (it keeps the cross-layer temporal
# paths the layer-local RTRL trace drops), the topology is the remaining suspect.
#
# Block layout per branch, matching RealTimeACMLPStacked exactly:
#
#   x = Dense(embed of [obs_emb, last_action, last_reward...])   (stream, width W)
#   repeat n_blocks times:
#     x = x + Proj(RTU(LayerNorm(x)))    # temporal mixing, pre-norm residual
#     x = x + MLP(LayerNorm(x))          # channel mixing,  pre-norm residual
#   heads(LayerNorm(x))
#
# THREE differences from BPTTActorCriticMLPStacked, and they travel together --
# this is not reachable by setting that class's `prenorm` flag, which moves the
# norm but leaves the other two alone:
#
#   1. Two residuals per block instead of one. The RTU gets its own skip, so it
#      reaches the stream without passing through the MLP.
#   2. Pre-norm instead of post-norm: LayerNorm sits inside each residual branch,
#      leaving an unbroken identity path from block 0 to the output. Post-norm
#      instead places a LayerNorm on that path once per block, which is the
#      arrangement whose gradients grow toward the output layers (Xiong et al.
#      2020) -- and E142's dead seeds show exactly that, grad_l2 on the deeper
#      RTU growing 7.2x while the shallower one shrinks.
#   3. A dedicated `rtu_proj` maps the RTU's 2*d_hidden output back to the width-W
#      stream, so `mlp_up` reads W rather than 2*d_hidden. That also removes the
#      2x parameter gap: 711,365 params here against 1,431,493 for the LRU-style
#      block at d_hidden=512 / W=64 / n_blocks=2, which makes this the
#      parameter-matched comparison against E140 as well as the topology-matched
#      one.
#
# use_gating replaces the plain residual adds with GTrXL-style GRU gates
# (Parisotto et al. 2020), biased toward the identity path at init -- carried
# over from RealTimeACMLPStacked for parity.
#
# Shape contract (from the BPTT family, NOT the RealTime one): obs arrays carry
# leading axes (seq_len, seq_batch, ...) and `hidden` holds each block's
# recurrent state at the sequence's FIRST step, shaped (seq_batch, d_hidden).
import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from algorithms.nn.activations import get_activation
from algorithms.nn.rtus.rtus import BPTTLRTUs, BPTTNonLRTUs


class _GRUGate(nn.Module):
    """GTrXL gating layer: out = (1-z)*x + z*h_hat. bg > 0 biases z toward 0
    (identity) at init so each block starts close to a skip connection."""

    width: int
    bg: float = 2.0

    @nn.compact
    def __call__(self, x, y):
        def dense(name):
            return nn.Dense(
                self.width,
                use_bias=False,
                kernel_init=orthogonal(1.0),
                name=name,
            )

        r = nn.sigmoid(dense("wr")(y) + dense("ur")(x))
        z = nn.sigmoid(dense("wz")(y) + dense("uz")(x) - self.bg)
        h_hat = nn.tanh(dense("wg")(y) + dense("ug")(r * x))
        return (1.0 - z) * x + z * h_hat


class BPTTActorCriticMLPStackedPreNorm(nn.Module):
    action_dim: int
    d_hidden: int = 192  # RTU units per block (RTU output is 2*d_hidden)
    hidden_size: int = 64  # residual stream width W
    n_blocks: int = 2
    activation: str = "tanh"  # MLP-sublayer nonlinearity
    cont: bool = False
    rtu_type: str = "linear_rtu"
    use_sinusoidal_encoding: bool = False
    use_reward_trace: bool = False
    use_layernorm: bool = True  # interface parity; blocks always pre-norm
    use_gating: bool = False  # GTrXL GRU gates instead of plain residual adds
    mlp_expansion: int = 4

    def _residual(self, x, y, name):
        if self.use_gating:
            return _GRUGate(self.hidden_size, name=name)(x, y)
        return x + y

    def _block(self, x, carry, seq_model, activation, unfold, fold, prefix, blk):
        """One [RTU -> MLP] pair with a pre-norm residual around each sublayer.

        `x` arrives folded as (seq_len*seq_batch, W); only the RTU sees the time
        axis, so it is unfolded around that call and folded straight back.
        LayerNorm and the Dense layers are per-timestep, so they operate on the
        folded array unchanged.
        """
        W = self.hidden_size
        # -- recurrent sublayer --
        y = nn.LayerNorm(name=f"{prefix}_layernorm_rtu{blk}")(x)
        carry, y = seq_model(
            self.d_hidden,
            params_type="exp_exp",
            activation="relu",
            name=f"{prefix}_rtu{blk}",
        )(carry, unfold(y))
        y = fold(y)
        self.sow("intermediates", f"{prefix}_rtu_out{blk}", y)
        y = nn.Dense(
            W,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name=f"{prefix}_rtu_proj{blk}",
        )(y)
        x = self._residual(x, y, f"{prefix}_rtu_gate{blk}")

        # -- MLP sublayer --
        y = nn.LayerNorm(name=f"{prefix}_layernorm_mlp{blk}")(x)
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
        x = self._residual(x, y, f"{prefix}_mlp_gate{blk}")
        return x, carry

    def _branch(self, obs, extras, carries, seq_model, activation, unfold, fold, prefix):
        """Full stack for one branch (actor or critic)."""
        W = self.hidden_size
        emb = nn.Dense(
            W,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name=f"{prefix}_dense1",
        )(obs)
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
        # Pre-norm leaves the accumulated stream unnormalised, so this final norm
        # is load-bearing here (unlike in the post-norm variant, where the last
        # block has already normalised).
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
        obs: ((seq_len, seq_batch, obs_dim), (seq_len, seq_batch, action_dim), ...)

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

        obs = jnp.reshape(obs, (obs.shape[0], -1))

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
