# From esraaelelimy/continuing_ppo
import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn

## Different Parameterization for r and theta
# 1. Direct parameterization: Learn r and theta directly. Options: clipping r to be positive, clipping r to be \in (0,1], no clipping.
# 2. Exp Re-parametrization with nu: r = exp(-nu), theta = theta. Options: clipping r to be \in (0,1], no clipping.
# 3. ExpExp Re-parameteization with nu_log: r = exp(-exp(nu_log)), theta = theta.
# 4. ExpExp Re-parameteization with nu_log and theta_log: r = exp(-exp(nu_log)), theta = exp(theta_log).
# 5. Sigmoid reparmetrization: r = sigmoid(nu), theta = theta.
# g = r cos(theta), phi = r sin(theta)

"""
Direct parameterization: Learn r and theta directly. Options: clipping r to be positive, clipping r to be \\in (0,1], no clipping.
stable_r: if True, clip r to be \\in (eps,1]
r.shape = (batch_size, n_hidden)
theta.shape = (batch_size, n_hidden)
"""


@jax.jit
def g_phi_direct_params(r, theta, stable_r=False, eps=1e-8):
    r = stable_r * jnp.clip(r, eps, 1) + (1 - stable_r) * r
    g = r * jnp.cos(theta)
    phi = r * jnp.sin(theta)
    norm = jnp.sqrt(1 - r**2) + eps
    return g, phi, norm


"""
Exp parameterization: Learn nu and theta. r = exp(-nu). Options: clipping r to be positive, clipping r to be \\in (0,1], no clipping.
"""


@jax.jit
def g_phi_exp_params(nu, theta, stable_r=True, eps=1e-8):
    r = jnp.exp(-nu)
    g, phi, norm = g_phi_direct_params(r, theta, stable_r, eps)
    return g, phi, norm


"""
ExpExp_nu parameterization: Learn nu_log and theta. r = exp(-exp(nu_log)). By design r is always less than 1
"""


@jax.jit
def g_phi_exp_exp_nu_params(nu_log, theta, stable_r=False, eps=1e-8):
    nu = jnp.exp(nu_log)
    g, phi, norm = g_phi_exp_params(
        nu, theta, stable_r=stable_r, eps=1e-8
    )  # by design r will always be less than 1
    return g, phi, norm


"""
ExpExp parameterization: Learn nu_log and theta_log. r = exp(-exp(nu_log)), theta = exp(theta_log). By design r is always less than 1
"""


@jax.jit
def g_phi_exp_exp_params(nu_log, theta_log, stable_r=False, eps=1e-8):
    nu = jnp.exp(nu_log)
    theta = jnp.exp(theta_log)
    g, phi, norm = g_phi_exp_params(
        nu, theta, stable_r=stable_r, eps=1e-8
    )  # by design r will always be less than 1
    return g, phi, norm


"""
Sigmoid parametrization: Learn nu and theta. r = sigmoid(nu)
"""


@jax.jit
def g_phi_sigmoid_params(nu, theta, stable_r=False, eps=1e-8):
    r = jax.nn.sigmoid(nu)
    g, phi, norm = g_phi_direct_params(r, theta, stable_r, eps)
    return g, phi, norm


## Methods for initializing r and theta
def initialize_direct_r(key, shape, r_max=1, r_min=0):
    u1 = jax.random.uniform(key, shape=shape)
    nu = -0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2)
    r = jnp.exp(-nu)
    return r


def initialize_exp_r(key, shape, r_max=1, r_min=0):
    u1 = jax.random.uniform(key, shape=shape)
    nu = -0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2)
    return nu


def initialize_exp_exp_r(key, shape, r_max=1, r_min=0):
    u1 = jax.random.uniform(key, shape=shape)
    nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
    return nu_log


def initialize_sigmoid_r(key, shape, r_max=1, r_min=0):
    u1 = jax.random.uniform(key, shape=shape)
    nu = -0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2)
    r = jnp.exp(-nu)
    return jax.nn.sigmoid(r)


def initialize_theta_log(key, shape, max_phase=6.28):
    u2 = jax.random.uniform(key, shape=shape)
    theta_log = jnp.log(max_phase * u2)
    return theta_log


def initialize_theta(key, shape, max_phase=6.28):
    u2 = jax.random.uniform(key, shape=shape)
    theta = max_phase * u2
    return theta


## Derivatives of g and phi w.r.t w_r and w_theta
@jax.jit
def d_g_phi_direct_params(w_r, w_theta, g, phi, norm):
    d_g_w_r = jnp.cos(w_theta)
    d_g_w_theta = -phi
    d_phi_w_r = jnp.sin(w_theta)
    d_phi_w_theta = g
    d_norm_w_r = -w_r / norm
    return d_g_w_r, d_g_w_theta, d_phi_w_r, d_phi_w_theta, d_norm_w_r


@jax.jit
def d_g_phi_exp_params(w_r, w_theta, g, phi, norm):
    d_g_w_r = -g
    d_g_w_theta = -phi
    d_phi_w_r = -phi
    d_phi_w_theta = g
    d_norm_w_r = jnp.exp(-2 * w_r) / norm
    return d_g_w_r, d_g_w_theta, d_phi_w_r, d_phi_w_theta, d_norm_w_r


@jax.jit
def d_g_phi_exp_exp_nu_params(w_r, w_theta, g, phi, norm):
    d_g_w_r = -jnp.exp(w_r) * g
    d_g_w_theta = -phi
    d_phi_w_r = -jnp.exp(w_r) * phi
    d_phi_w_theta = g
    d_norm_w_r = jnp.exp(w_r) * jnp.exp(-2 * jnp.exp(w_r)) / norm
    return d_g_w_r, d_g_w_theta, d_phi_w_r, d_phi_w_theta, d_norm_w_r


@jax.jit
def d_g_phi_exp_exp_params(w_r, w_theta, g, phi, norm):
    d_g_w_r = -jnp.exp(w_r) * g
    d_g_w_theta = -phi * jnp.exp(w_theta)
    d_phi_w_r = -jnp.exp(w_r) * phi
    d_phi_w_theta = g * jnp.exp(w_theta)
    d_norm_w_r = jnp.exp(w_r) * jnp.exp(-2 * jnp.exp(w_r)) / norm
    return d_g_w_r, d_g_w_theta, d_phi_w_r, d_phi_w_theta, d_norm_w_r


@jax.jit
def d_g_phi_sigmoid_params(w_r, w_theta, g, phi, norm):
    d_g_w_r = (1 - jax.nn.sigmoid(w_r)) * g
    d_g_w_theta = -phi
    d_phi_w_r = (1 - jax.nn.sigmoid(w_r)) * phi
    d_phi_w_theta = g
    d_norm_w_r = -jax.nn.sigmoid(w_r) * (1 - jax.nn.sigmoid(w_r)) / norm
    return d_g_w_r, d_g_w_theta, d_phi_w_r, d_phi_w_theta, d_norm_w_r


## Different options for g and phi
g_phi_options = {
    "direct": g_phi_direct_params,
    "exp": g_phi_exp_params,
    "exp_exp_nu": g_phi_exp_exp_nu_params,
    "exp_exp": g_phi_exp_exp_params,
    "sigmoid": g_phi_sigmoid_params,
}

init_options = {
    "direct": [initialize_direct_r, initialize_theta],
    "exp": [initialize_exp_r, initialize_theta],
    "exp_exp_nu": [initialize_exp_exp_r, initialize_theta],
    "exp_exp": [initialize_exp_exp_r, initialize_theta_log],
    "sigmoid": [initialize_sigmoid_r, initialize_theta],
}

d_g_phi = {
    "direct": d_g_phi_direct_params,
    "exp": d_g_phi_exp_params,
    "exp_exp_nu": d_g_phi_exp_exp_nu_params,
    "exp_exp": d_g_phi_exp_exp_params,
    "sigmoid": d_g_phi_sigmoid_params,
}


### custom activation functions
@jax.jit
def l2_norm(x, eps=1e-12):
    return x * jax.lax.rsqrt((x * x).sum(keepdims=True) + eps)


@jax.jit
def linear_act(x):
    return x


@jax.jit
def crelu_act(x):
    return jnp.concatenate((nn.relu(x), nn.relu(-x)), axis=-1)


@jax.jit
def drelu(x):
    return lax.select(x > 0, lax.full_like(x, 1), lax.full_like(x, 0))


@jax.jit
def dtanh(x):
    return 1 - jnp.tanh(x) ** 2


@jax.jit
def dlinear(x):
    return lax.full_like(x, 1)


@jax.jit
def d_l2_norm(x, eps=1e-12):
    return jax.jacfwd(l2_norm)(x)


act_options = {
    "relu": nn.relu,
    "tanh": nn.tanh,
    "linear": linear_act,
    "crelu": crelu_act,
}

d_act = {"relu": drelu, "tanh": dtanh, "linear": dlinear}
