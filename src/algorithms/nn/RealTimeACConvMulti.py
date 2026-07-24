# Conv variant of RealTimeActorCriticMLPMulti: the two-RTU "concat idiom" body
# from RealTimeACMLPMulti with the convolutional vision front-end from
# RealTimeACConv. Vision -> Conv stack -> Dense(obs_hidden_size) -> concat
# (action, last_reward[+hint]) -> RTU1 -> concat-skip -> Dense -> RTU2 ->
# concat-skip -> heads. Both RTUs read a width-hidden_size vector (uniform RTRL
# trace shapes), so initialize_memory matches RealTimeACMLPMulti exactly.
#
# The hint (ForagaxBig cue) rides in via last_reward, same as plain Conv.
import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from algorithms.nn.rtus.rtus import RTLRTUs, RTNLRTUs


class RealTimeActorCriticConvMulti(nn.Module):
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

    def _conv_stack(self, x, prefix):
        """Vision front-end matching RealTimeACConv (shared actor/critic code)."""
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
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

    @nn.compact
    def __call__(self, hidden, obs):
        """
        hidden: (actor1, critic1, actor2, critic2) RTU carries
        obs: ((batch, H, W, C), (batch, action_dim), (batch, 1[+hint]), ...)
        """
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        if self.rtu_type == "linear_rtu":
            seq_model = RTLRTUs
        elif self.rtu_type == "non_linear_rtu":
            seq_model = RTNLRTUs
        else:
            raise NotImplementedError

        (actor_hidden1, critic_hidden1, actor_hidden2, critic_hidden2) = hidden

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

        # Size the vision embedding so that after re-concatenating action/reward
        # the RTU input width == hidden_size (uniform trace shapes across RTUs).
        obs_hidden_size = (
            self.hidden_size
            - last_action_encoded.shape[-1]
            - last_reward_plus.shape[-1]
        )

        actor_embedding = self._conv_stack(obs, "actor")
        actor_embedding = nn.Dense(
            obs_hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense1",
        )(actor_embedding)
        if self.use_layernorm:
            actor_embedding = nn.LayerNorm(name="actor_layernorm_d1")(actor_embedding)
        actor_embedding = activation(actor_embedding)
        actor_embedding = jnp.concatenate(
            (actor_embedding, last_action_encoded, last_reward_plus), axis=-1
        )
        actor_embedding_skip = actor_embedding

        critic_embedding = self._conv_stack(obs, "critic")
        critic_embedding = nn.Dense(
            obs_hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense1",
        )(critic_embedding)
        if self.use_layernorm:
            critic_embedding = nn.LayerNorm(name="critic_layernorm_d1")(critic_embedding)
        critic_embedding = activation(critic_embedding)
        critic_embedding = jnp.concatenate(
            (critic_embedding, last_action_encoded, last_reward_plus), axis=-1
        )
        critic_embedding_skip = critic_embedding

        actor_hidden1, actor_embedding = seq_model(
            self.d_hidden, params_type="exp_exp", name="actor_rtu1"
        )(actor_hidden1, actor_embedding)
        critic_hidden1, critic_embedding = seq_model(
            self.d_hidden, params_type="exp_exp", name="critic_rtu1"
        )(critic_hidden1, critic_embedding)
        actor_embedding = jnp.concatenate(
            (actor_embedding, actor_embedding_skip), axis=-1
        )
        critic_embedding = jnp.concatenate(
            (critic_embedding, critic_embedding_skip), axis=-1
        )

        actor_embedding = nn.Dense(
            obs_hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense2",
        )(actor_embedding)
        if self.use_layernorm:
            actor_embedding = nn.LayerNorm(name="actor_layernorm2")(actor_embedding)
        actor_embedding = activation(actor_embedding)
        # Re-concat action/reward (as before RTU1) so RTU2's input width matches
        # RTU1's (== hidden_size); keeps the per-layer RTRL trace shapes uniform.
        actor_embedding = jnp.concatenate(
            (actor_embedding, last_action_encoded, last_reward_plus), axis=-1
        )
        actor_embedding_skip = actor_embedding
        critic_embedding = nn.Dense(
            obs_hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense2",
        )(critic_embedding)
        if self.use_layernorm:
            critic_embedding = nn.LayerNorm(name="critic_layernorm2")(critic_embedding)
        critic_embedding = activation(critic_embedding)
        critic_embedding = jnp.concatenate(
            (critic_embedding, last_action_encoded, last_reward_plus), axis=-1
        )
        critic_embedding_skip = critic_embedding

        actor_hidden2, actor_embedding = seq_model(
            self.d_hidden, params_type="exp_exp", name="actor_rtu2"
        )(actor_hidden2, actor_embedding)
        critic_hidden2, critic_embedding = seq_model(
            self.d_hidden, params_type="exp_exp", name="critic_rtu2"
        )(critic_hidden2, critic_embedding)
        actor_embedding = jnp.concatenate(
            (actor_embedding, actor_embedding_skip), axis=-1
        )
        critic_embedding = jnp.concatenate(
            (critic_embedding, critic_embedding_skip), axis=-1
        )

        actor_mean = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="actor_dense3",
        )(actor_embedding)
        if self.use_layernorm:
            actor_mean = nn.LayerNorm(name="actor_layernorm3")(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_mean",
        )(actor_mean)
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
            critic = nn.LayerNorm(name="critic_layernorm3")(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_value"
        )(critic)
        hidden = (actor_hidden1, critic_hidden1, actor_hidden2, critic_hidden2)
        return hidden, pi, jnp.squeeze(critic, axis=-1)

    # Identical to RealTimeActorCriticMLPMulti: four RTU carries, d_input == hidden_size.
    @staticmethod
    def initialize_memory(batch_size, d_hidden, d_input):
        def _rtu_carry():
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

        return (_rtu_carry(), _rtu_carry(), _rtu_carry(), _rtu_carry())
