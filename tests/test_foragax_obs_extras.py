"""Tests for the extra Foragax observation features (agent channel, patch offsets)."""

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from gymnax.environments import spaces

from environments.ForagaxObsExtras import make


@pytest.mark.parametrize("hint", [True, False])
def test_agent_channel_marks_position(hint):
    kwargs: dict[str, Any] = {"hint_duration": 100} if hint else {"return_hint": False}
    env = make(
        "ForagaxBig-v5",
        aperture_size=-1,
        observation_type="rgb",
        agent_channel=True,
        **kwargs,
    )
    space = env.observation_space(env.default_params)
    key = jax.random.key(0)
    obs, state = env.reset(key, env.default_params)
    step = jax.jit(env.step)
    for _ in range(20):
        key, act_key, step_key = jax.random.split(key, 3)
        action = jax.random.randint(act_key, (), 0, 4)
        obs, state, *_ = step(step_key, state, action, env.default_params)
        img = obs["image"] if hint else obs
        image_space = space.spaces["image"] if isinstance(space, spaces.Dict) else space
        expected_shape = image_space.shape
        assert img.shape == expected_shape == (28, 28, 4)
        assert img[..., -1].sum() == 1
        assert img[state.pos[1], state.pos[0], -1] == 1
        if hint:
            assert obs["hint"].sum() == 1


def test_default_is_unchanged():
    env = make("ForagaxBig-v5", aperture_size=-1, observation_type="rgb")
    obs, _ = env.reset(jax.random.key(0), env.default_params)
    assert obs["image"].shape == (28, 28, 3)


@pytest.mark.parametrize(
    "aperture_size, observation_type", [(9, "rgb"), (-1, "object")]
)
def test_rejects_unsupported_views(aperture_size, observation_type):
    with pytest.raises(ValueError):
        make(
            "ForagaxBig-v5",
            aperture_size=aperture_size,
            observation_type=observation_type,
            agent_channel=True,
        )


def _offsets_env():
    return make(
        "ForagaxBig-v5",
        aperture_size=9,
        observation_type="rgb",
        hint_duration=100,
        patch_offsets=True,
    )


def test_patch_offsets_shape_and_hint_unchanged():
    env = _offsets_env()
    plain = make("ForagaxBig-v5", aperture_size=9, observation_type="rgb", hint_duration=100)
    key = jax.random.key(0)
    obs, state = env.reset(key, env.default_params)
    plain_obs, _ = plain.reset(key, plain.default_params)
    space = env.observation_space(env.default_params)
    assert isinstance(space, spaces.Dict)
    assert obs["hint"].shape == space.spaces["hint"].shape == (4 + 2 * 4,)
    assert obs["image"].shape == plain_obs["image"].shape == (9, 9, 3)
    assert jnp.array_equal(obs["hint"][:4], plain_obs["hint"])
    assert jnp.all(jnp.abs(obs["hint"][4:]) <= 1)


@pytest.mark.parametrize("k, centre", [(0, (7, 7)), (1, (21, 7)), (2, (7, 21)), (3, (21, 21))])
def test_patch_offsets_zero_at_patch_centre(k, centre):
    env = _offsets_env()
    _, state = env.reset(jax.random.key(0), env.default_params)
    state = state.replace(pos=jnp.array(centre, dtype=state.pos.dtype))
    offsets = env.get_obs(state, env.default_params)["hint"][4:].reshape(4, 2)
    assert jnp.allclose(offsets[k], 0.0)


def test_patch_offsets_take_shortest_way_around():
    env = _offsets_env()
    _, state = env.reset(jax.random.key(0), env.default_params)
    state = state.replace(pos=jnp.array((0, 0), dtype=state.pos.dtype))
    offsets = env.get_obs(state, env.default_params)["hint"][4:].reshape(4, 2)
    # From (0, 0): patch 0 at (7, 7) is +7 each way; patch 3 at (21, 21) is
    # -7 each way across the wrapped edge, not +21. Half the world is 14.
    assert jnp.allclose(offsets[0], jnp.array([0.5, 0.5]))
    assert jnp.allclose(offsets[3], jnp.array([-0.5, -0.5]))
    assert jnp.allclose(offsets[1], jnp.array([-0.5, 0.5]))


def test_patch_offsets_render_matches_plain():
    env = _offsets_env()
    plain = make("ForagaxBig-v5", aperture_size=9, observation_type="rgb", hint_duration=100)
    _, state = env.reset(jax.random.key(0), env.default_params)
    frame = env.render(state, env.default_params, render_mode="world_reward")
    plain_frame = plain.render(state, plain.default_params, render_mode="world_reward")
    assert jnp.array_equal(frame, plain_frame)
    # get_obs still carries the offsets after a render.
    assert env.get_obs(state, env.default_params)["hint"].shape == (12,)


def test_patch_offsets_need_hint():
    with pytest.raises(ValueError):
        make(
            "ForagaxBig-v5",
            aperture_size=9,
            observation_type="rgb",
            return_hint=False,
            patch_offsets=True,
        )
