import flax.linen as nn
import jax.numpy as jnp


def crelu(x):
    return jnp.concatenate((nn.relu(x), nn.relu(-x)), axis=-1)


def get_activation(name: str):
    if name == "relu":
        return nn.relu
    if name == "crelu":
        return crelu
    return nn.tanh
