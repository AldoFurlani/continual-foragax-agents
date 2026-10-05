# Stacked [RTU -> MLP] residual blocks (LRU/S5-style backbone) for real-time PPO.
#
# Block layout per branch (actor and critic are separate stacks, following
# RealTimeACMLP):
#
#   x = Dense(embed of [obs_emb, last_action, last_reward...])      (stream, width W)
#   repeat n_blocks times:
#     x = x + Proj(RTU(LayerNorm(x)))        # temporal mixing, pre-norm residual
#     x = x + MLP(LayerNorm(x))              # channel mixing, pre-norm residual
#   heads(LayerNorm(x))
#
# Online-training semantics: RealTimeLinearRTUs wraps the cell in a custom_vjp
# that stop-gradients h_{t-1} and carries exact per-layer RTRL traces, passing
# only the instantaneous cotangent to its input. Stacking such cells therefore
# implements the layer-local scheme of Zucchet et al. (NeurIPS 2023): exact
# within-layer temporal credit, spatial backprop across blocks at the current
# step, cross-layer temporal sensitivities dropped.
#
# use_gating replaces the plain residual adds with GTrXL-style GRU gates
# (Parisotto et al. 2020), biased toward the identity path at init.
import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from algorithms.nn.activations import get_activation
from algorithms.nn.rtus.rtus import RTLRTUs, RTNLRTUs


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


class RealTimeActorCriticMLPStacked(nn.Module):
    action_dim: int
    d_hidden: int = 192  # RTU units per block (RTU output is 2*d_hidden)
    hidden_size: int = 64  # stream width W (also obs-embedding width)
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

    def _block(self, x, carry, seq_model, activation, prefix, blk):
        """One [RTU -> MLP] residual block on the width-W stream."""
        W = self.hidden_size
        # -- recurrent sublayer --
        y = nn.LayerNorm(name=f"{prefix}_layernorm_rtu{blk}")(x)
        carry, y = seq_model(
            self.d_hidden,
            params_type="exp_exp",
            activation="relu",
            name=f"{prefix}_rtu{blk}",
        )(carry, y)
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

    def _branch(self, obs, extras, carries, seq_model, activation, prefix):
        """Full stack for one branch (actor or critic). extras = concat of
        last_action / last_reward(+encodings); carries = tuple of RTU carries."""
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
            x, carry = self._block(x, carries[blk], seq_model, activation, prefix, blk)
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
                carries ((h_c1, h_c2), grad_memory)
        obs: (obs, last_action_encoded, last_reward, sine, cosine, reward_trace)
        """
        activation = get_activation(self.activation)
        if self.rtu_type == "linear_rtu":
            seq_model = RTLRTUs
        elif self.rtu_type == "non_linear_rtu":
            seq_model = RTNLRTUs
        else:
            raise NotImplementedError

        (actor_carries, critic_carries) = hidden
        (obs, last_action_encoded, last_reward, sine, cosine, reward_trace) = obs
        last_reward_plus = last_reward
        if self.use_sinusoidal_encoding:
            last_reward_plus = jnp.concatenate(
                (last_reward_plus, sine, cosine), axis=-1
            )
        if self.use_reward_trace:
            last_reward_plus = jnp.concatenate(
                (last_reward_plus, reward_trace), axis=-1
            )
        extras = (last_action_encoded, last_reward_plus)
        obs = jnp.reshape(obs, (obs.shape[0], -1))

        actor_out, actor_carries = self._branch(
            obs, extras, actor_carries, seq_model, activation, "actor"
        )
        critic_out, critic_carries = self._branch(
            obs, extras, critic_carries, seq_model, activation, "critic"
        )

        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_mean",
        )(actor_out)
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

        hidden = (actor_carries, critic_carries)
        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_memory(batch_size, d_hidden, d_input, n_blocks=2):
        """d_input is the stream width W (== hidden_size): every block's RTU
        reads the LayerNorm'd width-W stream, so all trace tensors share it."""

        def one_block():
            hidden_init = (
                jnp.zeros((batch_size, d_hidden)),
                jnp.zeros((batch_size, d_hidden)),
            )
            memory_grad_init = (
                jnp.zeros((batch_size, d_hidden)),
                jnp.zeros((batch_size, d_hidden)),
                jnp.zeros((batch_size, d_hidden)),
                jnp.zeros((batch_size, d_hidden)),
                jnp.zeros((batch_size, d_input, d_hidden)),
                jnp.zeros((batch_size, d_input, d_hidden)),
                jnp.zeros((batch_size, d_input, d_hidden)),
                jnp.zeros((batch_size, d_input, d_hidden)),
            )
            return (hidden_init, memory_grad_init)

        actor_carries = tuple(one_block() for _ in range(n_blocks))
        critic_carries = tuple(one_block() for _ in range(n_blocks))
        return (actor_carries, critic_carries)
