import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map


class SavedKey:
    """Raw data of a typed PRNG key array, which some jax versions cannot pickle."""

    def __init__(self, key):
        self.data = np.asarray(jax.random.key_data(key))
        self.impl = str(jax.random.key_impl(key))


def _is_key(x) -> bool:
    return hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jax.dtypes.prng_key)


def keys_to_data(tree):
    """Replace typed PRNG keys in a pytree with picklable SavedKey leaves."""
    return tree_map(lambda x: SavedKey(x) if _is_key(x) else x, tree)


def data_to_keys(tree):
    """Inverse of keys_to_data."""
    return tree_map(
        lambda x: jax.random.wrap_key_data(jnp.asarray(x.data), impl=x.impl) if isinstance(x, SavedKey) else x,
        tree,
        is_leaf=lambda x: isinstance(x, SavedKey),
    )
