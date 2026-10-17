"""Tests for pickling pytrees that hold typed PRNG keys."""

import pickle

import jax
import jax.numpy as jnp

from utils.prng_pickle import data_to_keys, keys_to_data


def _has_key_leaf(tree):
    return any(
        hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jax.dtypes.prng_key)
        for x in jax.tree_util.tree_leaves(tree)
    )


def test_round_trip_restores_keys():
    tree = {
        "key": jax.random.key(3),
        "keys": jax.random.split(jax.random.key(4), 5),
        "params": {"w": jnp.arange(6.0).reshape(2, 3)},
        "step": 7,
    }
    saved = keys_to_data(tree)
    assert not _has_key_leaf(saved)
    restored = data_to_keys(pickle.loads(pickle.dumps(saved)))
    assert jnp.issubdtype(restored["keys"].dtype, jax.dtypes.prng_key)
    assert jnp.array_equal(jax.random.key_data(restored["key"]), jax.random.key_data(tree["key"]))
    assert jnp.array_equal(jax.random.key_data(restored["keys"]), jax.random.key_data(tree["keys"]))
    assert jax.random.normal(restored["keys"][2]) == jax.random.normal(tree["keys"][2])
    assert jnp.array_equal(restored["params"]["w"], tree["params"]["w"])
    assert restored["step"] == 7


def test_trees_without_keys_are_unchanged():
    tree = {"w": jnp.ones(3), "n": 1}
    out = data_to_keys(keys_to_data(tree))
    assert jnp.array_equal(out["w"], tree["w"]) and out["n"] == 1
