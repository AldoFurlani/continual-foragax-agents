"""Tests for the full-world agent-position channel."""

import jax
import pytest

from gymnax.environments import spaces

from environments.ForagaxAgentChannel import make


@pytest.mark.parametrize("hint", [True, False])
def test_agent_channel_marks_position(hint):
    kwargs = {"hint_duration": 100} if hint else {"return_hint": False}
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
