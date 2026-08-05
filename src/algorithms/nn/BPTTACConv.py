# T-BPTT counterpart of RealTimeACConv.
#
# Identical architecture to RealTimeActorCriticConv -- same conv stack, same
# skip connection, same heads -- but the RTU is unrolled over a time axis with
# nn.scan (BPTTLRTUs / BPTTNonLRTUs) instead of stepped by the real-time cell.
# Gradients therefore flow backwards across the whole sequence and stop at its
# first step, which is the truncation.
#
# Shape contract: obs arrays carry leading axes (seq_len, seq_batch, ...) and
# `hidden` holds the recurrent state at each sequence's FIRST step, shaped
# (seq_batch, d_hidden).  The real-time class instead reads its leading axis as
# a batch of independent timesteps with one stored carry each.
import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from algorithms.nn.activations import get_activation
from algorithms.nn.rtus.rtus import BPTTLRTUs, BPTTNonLRTUs


class BPTTActorCriticConv(nn.Module):
    action_dim: int
    d_hidden: int = 192
    hidden_size: int = 64
    activation: str = "tanh"
    cont: bool = False
    rtu_type: str = "linear_rtu"

    use_sinusoidal_encoding: bool = False
    use_reward_trace: bool = False
    use_layernorm: bool = False
    conv: str = "Conv2D"

    @nn.compact
    def __call__(self, hidden, obs):
        """
        hidden: (((seq_batch, d_hidden), (seq_batch, d_hidden)),  # actor
                 ((seq_batch, d_hidden), (seq_batch, d_hidden)))  # critic
        obs: ((seq_len, seq_batch, H, W, C), (seq_len, seq_batch, action_dim), ...)

        Returns (final_hidden, pi, value) where pi/value carry leading
        (seq_len, seq_batch) axes.
        """
        activation = get_activation(self.activation)

        if self.rtu_type == "linear_rtu":
            seq_model = BPTTLRTUs
        elif self.rtu_type == "non_linear_rtu":
            seq_model = BPTTNonLRTUs
        else:
            raise NotImplementedError
        rtu_activation = (
            "crelu"
            if self.activation == "crelu" and self.rtu_type == "linear_rtu"
            else "relu"
        )

        (actor_hidden, critic_hidden) = hidden

        (obs, last_action_encoded, last_reward, sine, cosine, reward_trace) = obs

        # The conv/dense stacks are per-timestep and order-independent, so fold
        # time into the batch axis and run them once over seq_len*seq_batch
        # rows.  Only the RTU needs to see the time axis.
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

        # Actor conv stack
        actor_embedding = obs
        if self.conv in ("PConv2D", "PConv2DConv2D"):
            actor_embedding = nn.Conv(16, 1, 1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_pconv1")(actor_embedding)
            if self.use_layernorm:
                actor_embedding = nn.LayerNorm(epsilon=1e-05, name="actor_player_norm1")(actor_embedding)
            actor_embedding = activation(actor_embedding)
        if self.conv in ("Conv2D", "PConv2DConv2D"):
            actor_embedding = nn.Conv(16, 3, 1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_conv1")(actor_embedding)
            if self.use_layernorm:
                actor_embedding = nn.LayerNorm(epsilon=1e-05, name="actor_layernorm1")(actor_embedding)
            actor_embedding = activation(actor_embedding)
        # conv="none": skip all conv layers, flatten raw obs directly
        actor_embedding = jnp.reshape(actor_embedding, (actor_embedding.shape[0], -1))
        actor_embedding = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense2")(actor_embedding)
        if self.use_layernorm:
            actor_embedding = nn.LayerNorm(epsilon=1e-05, name="actor_layernorm2",
        )(actor_embedding)
        actor_embedding = activation(actor_embedding)
        actor_embedding = jnp.concatenate((
            actor_embedding,
            last_action_encoded, last_reward_plus),
            axis=-1,
        )
        actor_embedding_skip = actor_embedding

        # Critic conv stack
        critic_embedding = obs
        if self.conv in ("PConv2D", "PConv2DConv2D"):
            critic_embedding = nn.Conv(16, 1, 1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_pconv1")(critic_embedding)
            if self.use_layernorm:
                critic_embedding = nn.LayerNorm(epsilon=1e-05, name="critic_player_norm1")(critic_embedding)
            critic_embedding = activation(critic_embedding)
        if self.conv in ("Conv2D", "PConv2DConv2D"):
            critic_embedding = nn.Conv(16, 3, 1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_conv1")(critic_embedding)
            if self.use_layernorm:
                critic_embedding = nn.LayerNorm(epsilon=1e-05, name="critic_layernorm1")(critic_embedding)
            critic_embedding = activation(critic_embedding)
        # conv="none": skip all conv layers, flatten raw obs directly
        critic_embedding = jnp.reshape(
            critic_embedding, (critic_embedding.shape[0], -1)
        )
        critic_embedding = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense2")(critic_embedding)
        if self.use_layernorm:
            critic_embedding = nn.LayerNorm(epsilon=1e-05, name="critic_layernorm2",
        )(critic_embedding)
        critic_embedding = activation(critic_embedding)
        critic_embedding = jnp.concatenate((
            critic_embedding,
            last_action_encoded, last_reward_plus),
            axis=-1,
        )
        critic_embedding_skip = critic_embedding

        # Recurrence: restore the time axis, unroll, then fold it away again.
        # The carry entering the scan is a plain array, so the gradient path
        # through it terminates here -- this boundary is the T in T-BPTT.
        actor_hidden, actor_embedding = seq_model(
            self.d_hidden,
            params_type="exp_exp",
            activation=rtu_activation,
            name="actor_rtu",
        )(actor_hidden, unfold(actor_embedding))
        critic_hidden, critic_embedding = seq_model(
            self.d_hidden,
            params_type="exp_exp",
            activation=rtu_activation,
            name="critic_rtu",
        )(critic_hidden, unfold(critic_embedding))
        actor_embedding = fold(actor_embedding)
        critic_embedding = fold(critic_embedding)

        actor_embedding = jnp.concatenate((actor_embedding, actor_embedding_skip), axis=-1)
        critic_embedding = jnp.concatenate((critic_embedding, critic_embedding_skip), axis=-1)

        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor_dense3")(actor_embedding)
        if self.use_layernorm:
            actor_mean = nn.LayerNorm(epsilon=1e-05, name="actor_layernorm3")(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_mean",
        )(actor_mean)
        # actor_mean: (seq_len, seq_batch, action_dim)
        actor_mean = unfold(actor_mean)
        if self.cont:
            actor_logtstd = self.param(
                "log_std", nn.initializers.zeros, (self.action_dim,)
            )
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="critic_dense3",
        )(critic_embedding)
        if self.use_layernorm:
            critic = nn.LayerNorm(epsilon=1e-05, name="critic_layernorm3")(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_value"
        )(critic)
        # critic: (seq_len, seq_batch)
        critic = unfold(jnp.squeeze(critic, axis=-1))
        hidden = (actor_hidden, critic_hidden)
        return hidden, pi, critic

    @staticmethod
    def initialize_memory(batch_size, d_hidden, d_input=None):
        """Recurrent state only -- unlike the real-time cell there is no RTRL
        sensitivity carry, so `d_input` is accepted and ignored for call-site
        compatibility with the RealTime* classes."""
        del d_input

        def branch():
            return (
                jnp.zeros((batch_size, d_hidden)),
                jnp.zeros((batch_size, d_hidden)),
            )

        return (branch(), branch())
