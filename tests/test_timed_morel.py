# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false, reportIncompatibleMethodOverride=false
# (kept identical to the continual-foragax original, which is not type-checked
# with pyright: flax struct dataclasses and gymnax annotations trip it up)
# Copy of continual-foragax's tests/test_timed_morel.py for the local copy of the
# environment in src/environments/TimedMorel.py. Delete together with it.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foragax.env import Actions
from environments.ForagaxObsExtras import make
from environments.TimedMorel import (
    MOREL_DEATHCAP_ID,
    MOREL_ID,
    OYSTER_DEATHCAP_ID,
    OYSTER_ID,
    WALL_ID,
    TimedMorelForagaxEnv,
)

MOREL_COLUMNS = slice(0, 7)
OYSTER_COLUMNS = slice(8, 15)


def rollout(env, actions, seed=0):
    """Run the actions and return the reset state and the stacked transitions."""
    params = env.default_params
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key, params)

    def step(carry, action):
        key, state = carry
        key, step_key = jax.random.split(key)
        obs, state, reward, _, info = env.step(step_key, state, action, params)
        return (key, state), (obs, state, reward, info)

    _, out = jax.lax.scan(step, (key, state), actions)
    return (obs, state), out


def pacing(num_steps):
    """Walk down column 8 of the oyster biome forever (the world wraps)."""
    return jnp.full((num_steps,), Actions.DOWN, dtype=jnp.int32)


def band_rows(top):
    return {(int(top) + i) % 9 for i in range(3)}


def test_layout():
    env = TimedMorelForagaxEnv()
    params = env.default_params
    assert env.size == (16, 9)
    assert env.observation_space(params).shape == (5, 5, 4)

    obs, state = env.reset(jax.random.key(0), params)
    assert obs.shape == (5, 5, 4)
    grid = np.array(state.object_state.object_id)

    walls = grid == WALL_ID
    expected = np.zeros((9, 16), dtype=bool)
    expected[:, [7, 15]] = True
    expected[1, 7] = False
    expected[7, 15] = False
    np.testing.assert_array_equal(walls, expected)

    assert (grid[:, MOREL_COLUMNS] == 0).all()
    assert (grid[:, OYSTER_COLUMNS] == OYSTER_ID).sum() == 8
    assert (grid[:, OYSTER_COLUMNS] == OYSTER_DEATHCAP_ID).sum() == 4
    assert (grid > 0).sum() == walls.sum() + 12
    assert not band_rows(state.morel_band) & band_rows(state.deathcap_band)


def test_walls_and_doors():
    env = TimedMorelForagaxEnv()
    params = env.default_params
    key = jax.random.key(0)
    _, state = env.reset(key, params)

    def move(pos, action):
        s = state.replace(pos=jnp.array(pos, dtype=jnp.int32))
        _, s, _, _, _ = env.step(key, s, action, params)
        return tuple(int(v) for v in s.pos)

    # A move into a wall does not move the agent
    assert move((8, 4), Actions.LEFT) == (8, 4)
    assert move((6, 4), Actions.RIGHT) == (6, 4)
    # The world wraps, and column 15 is a wall too
    assert move((0, 4), Actions.LEFT) == (0, 4)
    assert move((3, 0), Actions.UP) == (3, 8)
    # Doors: column 7 at row 1, column 15 at row 7
    assert move((8, 1), Actions.LEFT) == (7, 1)
    assert move((7, 1), Actions.LEFT) == (6, 1)
    assert move((0, 7), Actions.LEFT) == (15, 7)
    assert move((15, 7), Actions.LEFT) == (14, 7)
    # No way up or down inside a wall column
    assert move((7, 1), Actions.UP) == (7, 1)
    assert move((7, 1), Actions.DOWN) == (7, 1)


def test_morel_clock_and_bands():
    env = TimedMorelForagaxEnv(band_move_every=None)
    _, (_, states, _, info) = rollout(env, pacing(1500))

    grid = np.array(states.object_state.object_id)[:, :, MOREL_COLUMNS]
    morels = (grid == MOREL_ID).sum(axis=(1, 2))
    deathcaps = (grid == MOREL_DEATHCAP_ID).sum(axis=(1, 2))
    # All mushrooms of the biome appear and go away together
    assert set(np.unique(morels)) == {0, 6}
    np.testing.assert_array_equal(morels, deathcaps)

    present = morels > 0
    starts = np.flatnonzero(present[1:] & ~present[:-1]) + 1
    stops = np.flatnonzero(~present[1:] & present[:-1]) + 1
    assert len(starts) >= 6
    assert ((np.diff(starts) >= 195) & (np.diff(starts) <= 205)).all()
    assert len(np.unique(np.diff(starts))) > 1
    # Not collected: gone 50 steps after they appear
    np.testing.assert_array_equal(stops - starts[: len(stops)], 50)
    np.testing.assert_array_equal(
        np.array(info["appearance_count"])[starts], np.arange(1, len(starts) + 1)
    )
    # The clock in info: mushrooms are out exactly while steps_to_expiry > 0
    np.testing.assert_array_equal(np.array(info["steps_to_expiry"]) > 0, present)
    np.testing.assert_array_equal(np.array(info["steps_to_expiry"])[starts], 50)
    to_appearance = np.array(info["steps_to_appearance"])
    np.testing.assert_array_equal(to_appearance[starts - 1], 1)
    assert ((to_appearance[starts] >= 195) & (to_appearance[starts] <= 205)).all()

    # Morels only in the morel band, deathcaps only in the deathcap band
    morel_rows = band_rows(states.morel_band[0])
    deathcap_rows = band_rows(states.deathcap_band[0])
    assert not morel_rows & deathcap_rows
    assert set(np.nonzero(grid == MOREL_ID)[1]) <= morel_rows
    assert set(np.nonzero(grid == MOREL_DEATHCAP_ID)[1]) <= deathcap_rows
    # Cells are drawn again at each appearance
    assert (grid[starts[0]] != grid[starts[1]]).any()
    # The bands never move
    assert len(np.unique(np.array(states.morel_band))) == 1
    assert len(np.unique(np.array(states.deathcap_band))) == 1


@pytest.mark.parametrize("k", [1, 2])
def test_band_move(k):
    env = TimedMorelForagaxEnv(band_move_every=k)
    _, (_, states, _, _) = rollout(env, pacing(1500))

    count = np.array(states.appearance_count)
    morel_band = np.array(states.morel_band)
    deathcap_band = np.array(states.deathcap_band)
    appearances = np.flatnonzero(np.diff(count)) + 1
    assert len(appearances) >= 6
    for step in appearances:
        moved = count[step] > 1 and (count[step] - 1) % k == 0
        assert (morel_band[step] != morel_band[step - 1]) == moved
        assert (deathcap_band[step] != deathcap_band[step - 1]) == moved
        assert not band_rows(morel_band[step]) & band_rows(deathcap_band[step])
    # The bands only change at an appearance
    changed = np.flatnonzero(np.diff(morel_band) | np.diff(deathcap_band)) + 1
    assert set(changed) <= set(appearances)

    # The mushrooms follow the bands
    grid = np.array(states.object_state.object_id)[:, :, MOREL_COLUMNS]
    for step in appearances:
        assert set(np.nonzero(grid[step] == MOREL_ID)[0]) <= band_rows(morel_band[step])
        assert set(np.nonzero(grid[step] == MOREL_DEATHCAP_ID)[0]) <= band_rows(
            deathcap_band[step]
        )


def test_fog():
    actions = pacing(450)
    fogged = TimedMorelForagaxEnv(aperture_size=-1, fog=True)
    clear = TimedMorelForagaxEnv(aperture_size=-1, fog=False)
    (obs0, _), (obs, states, _, _) = rollout(fogged, actions)
    _, (clear_obs, clear_states, _, _) = rollout(clear, actions)

    # The fog changes the observation only
    np.testing.assert_array_equal(
        np.array(states.object_state.object_id),
        np.array(clear_states.object_state.object_id),
    )
    grid = np.array(states.object_state.object_id)
    assert (grid[:, :, MOREL_COLUMNS] > 0).any()

    # Channels: morel, oyster, deathcap, wall
    assert obs0.shape == (9, 16, 4)
    assert (np.array(obs)[:, :, MOREL_COLUMNS] == 0).all()
    np.testing.assert_array_equal(
        np.array(obs)[:, :, 7:], np.array(clear_obs)[:, :, 7:]
    )
    clear_obs = np.array(clear_obs)
    np.testing.assert_array_equal(clear_obs[..., 0], grid == MOREL_ID)
    np.testing.assert_array_equal(clear_obs[..., 1], grid == OYSTER_ID)
    np.testing.assert_array_equal(
        clear_obs[..., 2], (grid == MOREL_DEATHCAP_ID) | (grid == OYSTER_DEATHCAP_ID)
    )
    np.testing.assert_array_equal(clear_obs[..., 3], grid == WALL_ID)


def test_aperture_fog():
    env = TimedMorelForagaxEnv(aperture_size=5)
    params = env.default_params
    key = jax.random.key(0)
    _, state = env.reset(key, params)
    grid = state.object_state.object_id.at[4, 3].set(MOREL_ID)
    grid = grid.at[4, 5].set(MOREL_DEATHCAP_ID)
    state = state.replace(
        pos=jnp.array([4, 4], dtype=jnp.int32),
        object_state=state.object_state.replace(object_id=grid),
    )
    obs = env.get_obs(state, params)
    assert obs.sum() == 0
    # From column 6 the wall and the oyster biome behind it are visible
    obs = env.get_obs(state.replace(pos=jnp.array([6, 4], dtype=jnp.int32)), params)
    assert (obs[:, 3, 3] == 1).all()
    assert obs[..., :3].sum() == (state.object_state.object_id[2:7, 8] > 0).sum()


def test_collect_hidden_mushrooms():
    env = TimedMorelForagaxEnv()
    params = env.default_params
    key = jax.random.key(0)
    _, state = env.reset(key, params)
    grid = state.object_state.object_id.at[4, 3].set(MOREL_ID)
    grid = grid.at[4, 5].set(MOREL_DEATHCAP_ID)
    # Keep the clock out of the way
    state = state.replace(
        pos=jnp.array([4, 4], dtype=jnp.int32),
        object_state=state.object_state.replace(object_id=grid),
        expiry=jnp.array(1000, dtype=jnp.int32),
        next_appearance=jnp.array(1000, dtype=jnp.int32),
    )

    _, after, reward, _, info = env.step(key, state, Actions.LEFT, params)
    assert reward == 10.0
    assert info["object_collected_id"] == MOREL_ID
    assert after.object_state.object_id[4, 3] == 0

    _, after, reward, _, info = env.step(key, state, Actions.RIGHT, params)
    assert reward == -5.0
    assert info["object_collected_id"] == MOREL_DEATHCAP_ID

    # A collected mushroom does not come back before the next appearance
    for _ in range(30):
        key, step_key = jax.random.split(key)
        _, after, reward, _, _ = env.step(step_key, after, Actions.UP, params)
        assert reward == 0.0
    grid = np.array(after.object_state.object_id)[:, MOREL_COLUMNS]
    assert (grid == MOREL_DEATHCAP_ID).sum() == 0
    assert (grid == MOREL_ID).sum() == 1
    assert (np.array(after.object_state.respawn_timer)[:, MOREL_COLUMNS] == 0).all()


def test_oyster_biome_respawn():
    env = TimedMorelForagaxEnv()
    actions = jax.random.randint(jax.random.key(1), (3000,), 0, 4)
    _, (_, states, rewards, _) = rollout(env, actions)

    grid = np.array(states.object_state.object_id)
    timer = np.array(states.object_state.respawn_timer)
    pending = np.array(states.object_state.respawn_object_id)
    assert {-1.0, 1.0} <= set(np.unique(np.array(rewards)))

    for object_id, count in ((OYSTER_ID, 8), (OYSTER_DEATHCAP_ID, 4)):
        here = grid == object_id
        coming = (pending == object_id) & (timer > 0)
        # Always in the oyster biome, and none is lost
        assert not here[:, :, :8].any() and not here[:, :, 15].any()
        assert not coming[:, :, :8].any() and not coming[:, :, 15].any()
        np.testing.assert_array_equal((here | coming).sum(axis=(1, 2)), count)

    # A collected object is away for 9-11 steps, at a new random cell
    new_timers = timer[1:][(timer[1:] > 0) & (timer[:-1] == 0)]
    assert set(np.unique(new_timers)) == {9, 10, 11}
    assert (timer[0][timer[0] > 0] >= 9).all()
    respawn_columns = np.nonzero(timer > 0)[2]
    assert len(np.unique(respawn_columns)) == 7


def test_registry():
    env = make("ForagaxTimedMorel-v1")
    assert isinstance(env, TimedMorelForagaxEnv)
    assert env.name == "ForagaxTimedMorel-v1"
    assert env.fog
    assert env.band_move_every == 1250
    assert env.observation_space(env.default_params).shape == (5, 5, 4)

    env = make("ForagaxTimedMorel-v1", fog=False, aperture_size=17)
    assert not env.fog
    assert env.observation_space(env.default_params).shape == (17, 17, 4)

    assert make("ForagaxTimedMorel-v1", band_move_every=1).band_move_every == 1
    assert make("ForagaxTimedMorel-v1", band_move_every=None).band_move_every == 0
    assert (
        make("ForagaxTimedMorel-v1", morel_deathcap_reward=-10.0)
        .objects[MOREL_DEATHCAP_ID]
        .reward_val
        == -10.0
    )
