# Modified from esraaelelimy/continuing_ppo
import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from algorithms.nn.activations import get_activation
from algorithms.nn.rtus.rtus import RTLRTUs, RTNLRTUs


class RealTimeActorCriticConv(nn.Module):
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
        hidden: ((batch_size, d_hidden), (batch_size, d_hidden))
        obs: ((batch_size, H, W, C), (batch_size, action_dim), (batch_size, 1), ...)
        """
        activation = get_activation(self.activation)

        if self.rtu_type == "linear_rtu":
            seq_model = RTLRTUs
        elif self.rtu_type == "non_linear_rtu":
            seq_model = RTNLRTUs
        else:
            raise NotImplementedError
        rtu_activation = (
            "crelu"
            if self.activation == "crelu" and self.rtu_type == "linear_rtu"
            else "relu"
        )

        (actor_hidden, critic_hidden) = hidden

        (obs, last_action_encoded, last_reward, sine, cosine, reward_trace) = obs
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

        actor_hidden, actor_embedding = seq_model(
            self.d_hidden,
            params_type="exp_exp",
            activation=rtu_activation,
            name="actor_rtu",
        )(actor_hidden, actor_embedding)
        critic_hidden, critic_embedding = seq_model(
            self.d_hidden,
            params_type="exp_exp",
            activation=rtu_activation,
            name="critic_rtu",
        )(critic_hidden, critic_embedding)
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
        # actor_mean: (batch_size, action_dim)
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
        # critic: (batch_size, 1)
        hidden = (actor_hidden, critic_hidden)
        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_memory(batch_size, d_hidden, d_input):
        actor_hidden_init = (
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_hidden)),
        )
        actor_memory_grad_init = (
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_input, d_hidden)),
            jnp.zeros((batch_size, d_input, d_hidden)),
            jnp.zeros((batch_size, d_input, d_hidden)),
            jnp.zeros((batch_size, d_input, d_hidden)),
        )

        critic_hidden_init = (
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_hidden)),
        )
        critic_memory_grad_init = (
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_hidden)),
            jnp.zeros((batch_size, d_input, d_hidden)),
            jnp.zeros((batch_size, d_input, d_hidden)),
            jnp.zeros((batch_size, d_input, d_hidden)),
            jnp.zeros((batch_size, d_input, d_hidden)),
        )

        carry_init = (
            (actor_hidden_init, actor_memory_grad_init),
            (critic_hidden_init, critic_memory_grad_init),
        )
        return carry_init
