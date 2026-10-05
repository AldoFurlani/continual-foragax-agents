# Modified from esraaelelimy/continuing_ppo
import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from algorithms.nn.activations import get_activation
from algorithms.nn.rtus.rtus import RTLRTUs, RTNLRTUs, ExpRTLRTUs


class RealTimeActorCriticMLP(nn.Module):
    action_dim: int
    d_hidden: int = 192
    hidden_size: int = 64
    activation: str = "tanh"
    cont: bool = False
    rtu_type: str = "linear_rtu"
    alpha: float = 0.9
    use_sinusoidal_encoding: bool = False
    use_reward_trace: bool = False
    use_layernorm: bool = False

    def _sow_act(self, x, name, activation):
        """Apply the layer activation and sow the plasticity probe at the site
        that matches the nonlinearity:
          - tanh: sow PRE-activation (saturation is a pre-tanh notion; the
            saturation metric re-applies tanh to this), then activate.
          - relu: activate first, then sow POST-activation (Sokar dormancy is
            measured on the actual unit outputs).
        Returns the activated tensor. This makes the single class reproduce the
        former tanh and ReLU variants exactly."""
        if self.activation == "relu":
            x = activation(x)
            self.sow("intermediates", name, x)
        else:
            self.sow("intermediates", name, x)
            x = activation(x)
        return x

    @nn.compact
    def __call__(self, hidden, obs):
        """
        hidden: ((batch_size, d_hidden), (batch_size, d_hidden))
        obs: ((batch_size, obs_dim), (batch_size, action_dim), (batch_size, 1))
        """
        activation = get_activation(self.activation)

        seq_kwargs = {}
        if self.rtu_type == "linear_rtu":
            seq_model = RTLRTUs
        elif self.rtu_type == "non_linear_rtu":
            seq_model = RTNLRTUs
        elif self.rtu_type == "exp_rtu":
            seq_model = ExpRTLRTUs
            seq_kwargs["alpha"] = self.alpha  #need this because only exp_rtu class has an alpha config
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
            last_reward_plus = jnp.concatenate(
                (last_reward_plus, sine, cosine), axis=-1
            )
        if self.use_reward_trace:
            last_reward_plus = jnp.concatenate(
                (last_reward_plus, reward_trace), axis=-1
            )

        obs = jnp.reshape(obs, (obs.shape[0], -1))

        actor_embedding = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_dense1",
        )(obs)
        if self.use_layernorm:
            actor_embedding = nn.LayerNorm(name="actor_layernorm1")(actor_embedding)
        # Plasticity-metric probe (pre-tanh for tanh, post-ReLU for relu).
        # No-op unless apply() is called with mutable=['intermediates'].
        actor_embedding = self._sow_act(actor_embedding, "actor_pre1", activation)
        actor_embedding = jnp.concatenate(
            (actor_embedding, last_action_encoded, last_reward_plus), axis=-1
        )
        actor_embedding_skip = actor_embedding

        critic_embedding = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_dense1",
        )(obs)
        if self.use_layernorm:
            critic_embedding = nn.LayerNorm(name="critic_layernorm1")(critic_embedding)
        critic_embedding = self._sow_act(critic_embedding, "critic_pre1", activation)
        critic_embedding = jnp.concatenate(
            (critic_embedding, last_action_encoded, last_reward_plus), axis=-1
        )
        critic_embedding_skip = critic_embedding

        actor_hidden, actor_embedding = seq_model(
            self.d_hidden,
            params_type="exp_exp",
            activation=rtu_activation,
            name="actor_rtu",
            **seq_kwargs
        )(actor_hidden, actor_embedding)
        critic_hidden, critic_embedding = seq_model(
            self.d_hidden,
            params_type="exp_exp",
            activation=rtu_activation,
            name="critic_rtu",
            **seq_kwargs
        )(critic_hidden, critic_embedding)
        # RTU output is post-nonlinearity: RTLRTUs/RTNLRTUs apply
        # `act_options[self.activation]` to the concatenated recurrent state
        # at the end of __call__, and we instantiate them with the explicit
        # `activation=rtu_activation` parameter to control which nonlinearity
        # is applied.
        # Effective rank measures how many independent recurrent feature
        # dimensions are in use; Sokar τ-dormancy applies here in its
        # original sense because the signal is post-ReLU. The "_out" suffix
        # avoids colliding with the "actor_rtu"/"critic_rtu" submodule names
        # flax uses for scope tracking.
        self.sow("intermediates", "actor_rtu_out", actor_embedding)
        self.sow("intermediates", "critic_rtu_out", critic_embedding)
        actor_embedding = jnp.concatenate((actor_embedding, actor_embedding_skip), axis=-1)
        critic_embedding = jnp.concatenate((critic_embedding, critic_embedding_skip), axis=-1)

        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor_dense2")(actor_embedding)
        if self.use_layernorm:
            actor_mean = nn.LayerNorm(epsilon=1e-05, name="actor_layernorm2")(actor_mean)
        actor_mean = self._sow_act(actor_mean, "actor_pre2", activation)
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
            name="critic_dense2",
        )(critic_embedding)
        if self.use_layernorm:
            critic = nn.LayerNorm(name="critic_layernorm2")(critic)
        critic = self._sow_act(critic, "critic_pre2", activation)
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
