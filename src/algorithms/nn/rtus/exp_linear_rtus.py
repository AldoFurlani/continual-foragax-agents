"""
RTUs variations
1. Linear RTUs without gradient corrections in forward pass
3. BPTT for linear RTUs
5. Linear RTUs with gradient corrections in forward pass
6. Real-Time Linear RTUs
7. A Consice interface to Real-Time Linear RTUs
"""

from flax import linen as nn
import jax
import jax.numpy as jnp
import flax
from typing import Callable, Any, Tuple, Iterable, Optional
from algorithms.nn.rtus.rtus_utils import *

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


"""
# Linear RTUs with gradient corrections in forward pass
grad memory: dh_c1_{t-1}/d w_r #0, dh_c2_{t-1}/d w_r #1,  
             dh_c1_{t-1}/d w_theta #2, dh_c2_{t-1}/d w_theta #3,
             dh_c1_{t-1}/d w_c1_x #4, dh_c2_{t-1}/d w_c1_x #5,  
             dh_c1_{t-1}/d w_c2_x #6, dh_c2_{t-1}/d w_c2_x #7, 

dh_c*/d w_r.shape = (batch_size, n_hidden)
dh_c*/d w_theta.shape = (batch_size, n_hidden)
dh_c*/d w_c1_x.shape = (batch_size,d_input,n_hidden)
dh_c*/d w_c2_x.shape = (batch_size,d_input,n_hidden)

With batch of data, the gradient correction is calculated for the whole batch and summed over during the backward pass.
"""


class ExpFwdRealTimeLinearRTUs(nn.Module):
    n_hidden: int  # number of hidden features
    params_type: str = "exp_exp"  # direct, exp, exp_exp_nu, exp_exp
    stable_r: bool = False  # if True, clip r to be \in (eps,1]
    alpha: float = 0.9

    @nn.compact
    def __call__(self, carry, x_t):
        ## x_t.shape = (batch_size, n_features), h_tminus1.shape = ((batch_size, n_hiddens),(batch_size, n_hiddens))
        h_tminus1, grad_memory = carry
        h_tminus1_c1, h_tminus1_c2 = h_tminus1
        h_tminus1_c1, h_tminus1_c2 = (
            jax.lax.stop_gradient(h_tminus1_c1),
            jax.lax.stop_gradient(h_tminus1_c2),
        )

        # these params might not be the actual r and theta, but a transformed version of them based on the param_type
        r_param = self.param(
            "r_param", init_options[self.params_type][0], (1, self.n_hidden)
        )
        theta_param = self.param(
            "theta_param", init_options[self.params_type][1], (1, self.n_hidden)
        )

        mlp_xc1 = nn.Dense(
            self.n_hidden, name="wx1", use_bias=False
        )  # use ones for init to gaurntee same inital satates
        mlp_xc2 = nn.Dense(
            self.n_hidden, name="wx2", use_bias=False
        )  # use ones for init to gaurntee same inital satates
        g, phi, norm = g_phi_options[self.params_type](r_param, theta_param)

        w_c1_x_t = mlp_xc1(x_t)
        w_c2_x_t = mlp_xc2(x_t)

        h_t_c1 = (
            jnp.multiply(g, h_tminus1_c1)
            - jnp.multiply(phi, h_tminus1_c2)
            + jnp.multiply(norm, w_c1_x_t)
        )
        h_t_c2 = (
            jnp.multiply(g, h_tminus1_c2)
            + jnp.multiply(phi, h_tminus1_c1)
            + jnp.multiply(norm, w_c2_x_t)
        )

        ### Needed Gradient information for gradient corrections
        d_g_w_r, d_g_w_theta, d_phi_w_r, d_phi_w_theta, d_norm_w_r = d_g_phi[
            self.params_type
        ](r_param, theta_param, g, phi, norm)

        new_grad_memory_hc1_w_r = (
            self.alpha * d_g_w_r * h_tminus1_c1
            + (1 - self.alpha) * g * grad_memory[0]
            - self.alpha * d_phi_w_r * h_tminus1_c2
            - (1 - self.alpha) * phi * grad_memory[1]
            + self.alpha * jnp.multiply(d_norm_w_r, w_c1_x_t)
        )
        new_grad_memory_hc2_w_r = (
            self.alpha * d_g_w_r * h_tminus1_c2
            + (1 - self.alpha) * g * grad_memory[1]
            + self.alpha * d_phi_w_r * h_tminus1_c1
            + (1 - self.alpha) * phi * grad_memory[0]
            + self.alpha * jnp.multiply(d_norm_w_r, w_c2_x_t)
        )

        new_grad_memory_hc1_w_theta = (
            self.alpha * d_g_w_theta * h_tminus1_c1
            + (1 - self.alpha) * g * grad_memory[2]
            - self.alpha * d_phi_w_theta * h_tminus1_c2
            - (1 - self.alpha) * phi * grad_memory[3]
        )
        new_grad_memory_hc2_w_theta = (
            self.alpha * d_g_w_theta * h_tminus1_c2
            + (1 - self.alpha) * g * grad_memory[3]
            + self.alpha * d_phi_w_theta * h_tminus1_c1
            + (1 - self.alpha) * phi * grad_memory[2]
        )

        new_grad_c1_wx1 = (
            (1 - self.alpha) * jnp.multiply(g, grad_memory[4])
            - (1 - self.alpha) * jnp.multiply(phi, grad_memory[5])
            + self.alpha
            * jnp.multiply(
                norm, jnp.repeat(jnp.expand_dims(x_t, 2), h_t_c1.shape[-1], axis=2)
            )
        )
        new_grad_c1_wx2 = (1 - self.alpha) * jnp.multiply(g, grad_memory[6]) - (
            1 - self.alpha
        ) * jnp.multiply(phi, grad_memory[7])

        new_grad_c2_wx1 = (1 - self.alpha) * jnp.multiply(g, grad_memory[5]) + (
            1 - self.alpha
        ) * jnp.multiply(phi, grad_memory[4])
        new_grad_c2_wx2 = (
            (1 - self.alpha) * jnp.multiply(g, grad_memory[7])
            + (1 - self.alpha) * jnp.multiply(phi, grad_memory[6])
            + self.alpha
            * jnp.multiply(
                norm, jnp.repeat(jnp.expand_dims(x_t, 2), h_t_c2.shape[-1], axis=2)
            )
        )

        new_grad_memory = (
            new_grad_memory_hc1_w_r,
            new_grad_memory_hc2_w_r,
            new_grad_memory_hc1_w_theta,
            new_grad_memory_hc2_w_theta,
            new_grad_c1_wx1,
            new_grad_c2_wx1,
            new_grad_c1_wx2,
            new_grad_c2_wx2,
        )

        return ((h_t_c1, h_t_c2), jax.lax.stop_gradient(new_grad_memory)), (
            (h_t_c1, h_t_c2),
            jax.lax.stop_gradient(new_grad_memory),
        )


class ExpRealTimeLinearRTUs(nn.Module):
    n_hidden: int  # number of hidden features
    params_type: str = "exp_exp"  # direct, exp, exp_exp_nu, exp_exp
    stable_r: bool = False  # if True, clip r to be \in (eps,1]
    alpha: float = 0.9

    @nn.compact
    def __call__(self, carry, x_t):
        def f(mdl, carry, x_t):
            return mdl(carry, x_t)

        def fwd(mdl, carry, x_t):
            f_out, vjp_func = nn.vjp(f, mdl, carry, x_t)
            return f_out, (vjp_func, f_out[1][1])

        def bwd(residuals, y_t):
            # y_t =(partial{output}/partial{h_{t,c1}},partial{output}/partial{h_{t,c2}}),ignore the rest
            # grad_memory = \partial{h_{t-1},c1} \partial{lambda},\partial{h_{t-1,c2}} \partial{lambda},
            # \partial{h_{t-1},c1} \partial{theta},\partial{h_{t-1,c2}} \partial{theta}
            vjp_func, new_grad_memory = residuals
            params_t, *inputs_t = vjp_func(y_t)
            d_output_d_h1 = y_t[1][0][0]
            d_output_d_h2 = y_t[1][0][1]
            correct_w_r = (
                d_output_d_h1 * new_grad_memory[0] + d_output_d_h2 * new_grad_memory[1]
            )
            correct_w_theta = (
                d_output_d_h1 * new_grad_memory[2] + d_output_d_h2 * new_grad_memory[3]
            )
            correct_w1x = (
                jnp.expand_dims(d_output_d_h1, 1) * new_grad_memory[4]
                + jnp.expand_dims(d_output_d_h2, 1) * new_grad_memory[5]
            )
            correct_w2x = (
                jnp.expand_dims(d_output_d_h1, 1) * new_grad_memory[6]
                + jnp.expand_dims(d_output_d_h2, 1) * new_grad_memory[7]
            )
            params_t1 = flax.core.unfreeze(params_t)
            params_t1["params"]["r_param"] = jnp.expand_dims(jnp.sum(correct_w_r, 0), 0)
            params_t1["params"]["theta_param"] = jnp.expand_dims(
                jnp.sum(correct_w_theta, 0), 0
            )
            params_t1["params"]["wx1"]["kernel"] = jnp.sum(correct_w1x, 0)
            params_t1["params"]["wx2"]["kernel"] = jnp.sum(correct_w2x, 0)
            return (params_t1, *inputs_t)

        rt_cell_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)
        model_fn = ExpFwdRealTimeLinearRTUs(
            n_hidden=self.n_hidden,
            params_type=self.params_type,
            stable_r=self.stable_r,
            alpha=self.alpha,
        )
        ((h_t_c1, h_t_c2), new_grad_memory), ((h_t_c1, h_t_c2), new_grad_memory) = (
            rt_cell_grad(model_fn, carry, x_t)
        )
        return ((h_t_c1, h_t_c2), new_grad_memory), (h_t_c1, h_t_c2)  # carry, output
