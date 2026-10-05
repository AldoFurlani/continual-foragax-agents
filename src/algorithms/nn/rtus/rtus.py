# From esraaelelimy/continuing_ppo
from flax import linen as nn
import jax.numpy as jnp
from typing import Any, Tuple

from algorithms.nn.rtus.linear_rtus import LinearRTUs, RealTimeLinearRTUs
from algorithms.nn.rtus.non_linear_rtus import NonLinearRTUs, RealTimeNonLinearRTUs
from algorithms.nn.rtus.exp_linear_rtus import ExpRealTimeLinearRTUs
from algorithms.nn.rtus.rtus_utils import act_options


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

class ExpRTLRTUs(nn.Module):
    n_hidden: int  # number of hidden features
    params_type: str = "exp_exp"  # direct, exp, exp_exp_nu, exp_exp
    stable_r: bool = False  # if True, clip r to be \in (eps,1]
    d_input: int = 1
    activation: str = "relu"
    alpha: float = 0.9
    @nn.compact
    def __call__(self, carry, x_t):
        update_gate = ExpRealTimeLinearRTUs(
            self.n_hidden,
            self.params_type,
            self.stable_r,
            self.alpha,
        )
        carry, (h_t_c1, h_t_c2) = update_gate(carry, x_t)
        h_t = act_options[self.activation](jnp.concatenate((h_t_c1, h_t_c2), axis=-1))
        return carry, h_t  # carry, output

    def initialize_state(self, batch_size=1):
        hidden_init = (
            jnp.zeros((batch_size, self.n_hidden)),
            jnp.zeros((batch_size, self.n_hidden)),
        )
        memory_grad_init = (
            jnp.zeros((batch_size, self.n_hidden)),
            jnp.zeros((batch_size, self.n_hidden)),
            jnp.zeros((batch_size, self.n_hidden)),
            jnp.zeros((batch_size, self.n_hidden)),
            jnp.zeros((batch_size, self.d_input, self.n_hidden)),
            jnp.zeros((batch_size, self.d_input, self.n_hidden)),
            jnp.zeros((batch_size, self.d_input, self.n_hidden)),
            jnp.zeros((batch_size, self.d_input, self.n_hidden)),
        )
        return (hidden_init, memory_grad_init)


## BPTT for Non-Linear RTUs expect inputs of shape (batch_size, n_timesteps, n_features)
## real-time rtus expect inputs of shape (batch_size, n_features)

"""
A Consice interface to Real-Time Non-Linear RTUs
Non-linear recurrence
"""


class RTNLRTUs(nn.Module):
    n_hidden: int  # number of hidden features
    params_type: str = "exp_exp"  # direct, exp, exp_exp_nu, exp_exp
    stable_r: bool = False  # if True, clip r to be \in (eps,1]
    d_input: int = 1
    activation: str = "relu"

    @nn.compact
    def __call__(self, carry, x_t):
        update_gate = RealTimeNonLinearRTUs(
            self.n_hidden,
            self.params_type,
            self.stable_r,
            self.d_input,
            self.activation,
        )
        carry, h_t = update_gate(carry, x_t)
        return carry, h_t  # carry, output

    def initialize_state(self, batch_size=1):
        hidden_init = (
            jnp.zeros((batch_size, self.n_hidden)),
            jnp.zeros((batch_size, self.n_hidden)),
        )
        memory_grad_init = (
            jnp.zeros((batch_size, self.n_hidden)),
            jnp.zeros((batch_size, self.n_hidden)),
            jnp.zeros((batch_size, self.n_hidden)),
            jnp.zeros((batch_size, self.n_hidden)),
            jnp.zeros((batch_size, self.d_input, self.n_hidden)),
            jnp.zeros((batch_size, self.d_input, self.n_hidden)),
            jnp.zeros((batch_size, self.d_input, self.n_hidden)),
            jnp.zeros((batch_size, self.d_input, self.n_hidden)),
        )
        return (hidden_init, memory_grad_init)


# BPTT for Non-Linear RTUs
class BPTTNonLRTUs(nn.Module):
    n_hidden: int  # number of hidden features
    params_type: str = "direct"  # direct, exp, exp_exp_nu, exp_exp
    stable_r: bool = False  # if True, clip r to be \in (eps,1]
    activation: str = "relu"

    @nn.compact
    def __call__(self, c, xs):
        # xs.shape = (n_timesteps,batch_size,  n_hiddens)
        # c.shape = ((batch_size, n_hiddens),(batch_size, n_hiddens))
        rtu_scan = nn.scan(
            NonLinearRTUs,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )
        return rtu_scan(
            self.n_hidden, self.params_type, self.stable_r, self.activation
        )(c, xs)

    def initialize_state(self, batch_size=1):
        hidden_init = (
            jnp.zeros(
                (
                    batch_size,
                    self.n_hidden,
                )
            ),
            jnp.zeros((batch_size, self.n_hidden)),
        )
        return hidden_init


"""
A Consice interface to Real-Time Linear RTUs
Linear recurrence + non-linear output
"""


class RTLRTUs(nn.Module):
    n_hidden: int  # number of hidden features
    params_type: str = "exp_exp"  # direct, exp, exp_exp_nu, exp_exp
    stable_r: bool = False  # if True, clip r to be \in (eps,1]
    d_input: int = 1
    activation: str = "relu"

    @nn.compact
    def __call__(self, carry, x_t):
        update_gate = RealTimeLinearRTUs(self.n_hidden, self.params_type, self.stable_r)
        carry, (h_t_c1, h_t_c2) = update_gate(carry, x_t)
        h_t = act_options[self.activation](jnp.concatenate((h_t_c1, h_t_c2), axis=-1))
        return carry, h_t  # carry, output

    def initialize_state(self, batch_size=1):
        hidden_init = (
            jnp.zeros((batch_size, self.n_hidden)),
            jnp.zeros((batch_size, self.n_hidden)),
        )
        memory_grad_init = (
            jnp.zeros((batch_size, self.n_hidden)),
            jnp.zeros((batch_size, self.n_hidden)),
            jnp.zeros((batch_size, self.n_hidden)),
            jnp.zeros((batch_size, self.n_hidden)),
            jnp.zeros((batch_size, self.d_input, self.n_hidden)),
            jnp.zeros((batch_size, self.d_input, self.n_hidden)),
            jnp.zeros((batch_size, self.d_input, self.n_hidden)),
            jnp.zeros((batch_size, self.d_input, self.n_hidden)),
        )
        return (hidden_init, memory_grad_init)


# BPTT for linear RTUs
class BPTTLRTUs(nn.Module):
    n_hidden: int  # number of hidden features
    params_type: str = "direct"  # direct, exp, exp_exp_nu, exp_exp
    stable_r: bool = False  # if True, clip r to be \in (eps,1]
    activation: str = "relu"

    @nn.compact
    def __call__(self, c, xs):
        # xs.shape = (n_timesteps,batch_size,n_hiddens)
        # c.shape = ((batch_size, n_hiddens),(batch_size, n_hiddens))
        rtu_scan = nn.scan(
            LinearRTUs,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )
        return rtu_scan(
            self.n_hidden, self.params_type, self.stable_r, self.activation
        )(c, xs)

    def initialize_state(self, batch_size=1):
        hidden_init = (
            jnp.zeros(
                (
                    batch_size,
                    self.n_hidden,
                )
            ),
            jnp.zeros((batch_size, self.n_hidden)),
        )
        return hidden_init
