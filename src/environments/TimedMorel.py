# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false, reportIncompatibleMethodOverride=false
# (kept identical to the continual-foragax original, which is not type-checked
# with pyright: flax struct dataclasses and gymnax annotations trip it up)
# LOCAL COPY of continual-foragax's foragax/timed_morel.py (branch
# feat/timed-morel-forager), so E146 can run before that branch is released.
# Once the installed continual-foragax has ForagaxTimedMorel-v1, delete this
# file and its hook in ForagaxObsExtras.make, and import from foragax.timed_morel.
# Differences from the original: NoRegenForagaxObject is defined here (the
# released foragax.objects does not have it yet) and make() is added.
"""Timed-Morel Forager.

A 16 x 9 torus with two biomes of 7 columns, separated by two wall columns
with one door each.

Morel biome (columns 0-6): all of its mushrooms appear together on a clock
(every 195-205 steps) and the ones that are not collected go away 50 steps
later. Morels grow only in one band of 3 rows and deathcaps only in another
band of 3 rows. Every ``band_move_every`` appearances both bands move to new
rows, with no signal. With ``fog`` the whole biome looks empty in the
observation, so the agent finds mushrooms only by walking into them.

Oyster biome (columns 8-14): oysters and deathcaps that respawn a few steps
after collection at a random free cell of the biome. No fog.
"""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from foragax.env import (
    BIOME_ID_DTYPE,
    COLOR_DTYPE,
    ID_DTYPE,
    REWARD_DTYPE,
    Biome,
    BiomeState,
    EnvParams,
    EnvState,
    ForagaxEnv,
    ObjectState,
)
from foragax.objects import WALL, DefaultForagaxObject


class NoRegenForagaxObject(DefaultForagaxObject):
    """Collectable object that does not come back on its own after collection.

    The environment decides when (and where) it appears again.
    """

    def regen_delay(self, clock: int, rng: jax.Array) -> int:
        """No individual regeneration - returns infinity."""
        return jnp.iinfo(jnp.int32).max

WIDTH, HEIGHT = 16, 9
BIOME_WIDTH = 7
MOREL_BIOME_COLUMNS = (0, 7)  # [start, stop)
OYSTER_BIOME_COLUMNS = (8, 15)  # [start, stop)
# Wall column -> row of its door
DOORS = {7: 1, 15: 7}

MOREL_COLOR = (63, 30, 25)
OYSTER_COLOR = (124, 61, 81)
DEATHCAP_COLOR = (193, 178, 30)

# Object IDs (0 is EMPTY)
MOREL_ID = 1
OYSTER_ID = 2
MOREL_DEATHCAP_ID = 3
OYSTER_DEATHCAP_ID = 4
WALL_ID = 5

MOREL_BIOME_ID = 0
OYSTER_BIOME_ID = 1


@struct.dataclass
class TimedMorelEnvState(EnvState):
    next_appearance: jax.Array  # time of the next morel appearance
    expiry: jax.Array  # time at which the current morel-biome mushrooms go away
    appearance_count: jax.Array  # morel appearances so far
    morel_band: jax.Array  # top row of the morel growth band
    deathcap_band: jax.Array  # top row of the deathcap growth band


class TimedMorelForagaxEnv(ForagaxEnv):
    """Two-biome forager with a morel clock, walls, fog and hidden growth bands."""

    def __init__(
        self,
        name: str = "ForagaxTimedMorel-v1",
        aperture_size: Union[Tuple[int, int], int] = (5, 5),
        observation_type: str = "color",
        fog: bool = True,
        band_move_every: Optional[int] = 1250,
        morel_reward: float = 10.0,
        morel_deathcap_reward: float = -5.0,
        num_morels: int = 6,
        num_morel_deathcaps: int = 6,
        appearance_interval: Tuple[int, int] = (195, 205),
        expiry_time: int = 50,
        band_height: int = 3,
        oyster_reward: float = 1.0,
        oyster_deathcap_reward: float = -1.0,
        num_oysters: int = 8,
        num_oyster_deathcaps: int = 4,
        oyster_regen_delay: Tuple[int, int] = (9, 11),
    ):
        """
        Args:
            fog: Hide every object of the morel biome from the observation.
            band_move_every: Number of morel appearances between band moves.
                None or 0 means the bands never move.
            appearance_interval: Inclusive range of steps between appearances.
            expiry_time: Steps a morel-biome mushroom stays before it goes away.
            oyster_regen_delay: Inclusive range of steps an oyster-biome object
                is away after collection.
        """
        band_cells = band_height * BIOME_WIDTH
        if max(num_morels, num_morel_deathcaps) > band_cells:
            raise ValueError("A growth band cannot hold that many mushrooms")
        if 2 * band_height > HEIGHT:
            raise ValueError("The two growth bands do not fit in the biome")
        if num_oysters + num_oyster_deathcaps > BIOME_WIDTH * HEIGHT:
            raise ValueError("The oyster biome cannot hold that many objects")
        if expiry_time > appearance_interval[0]:
            raise ValueError("Mushrooms must expire before the next appearance")

        self.fog = fog
        self.band_move_every = band_move_every or 0
        self.num_morels = num_morels
        self.num_morel_deathcaps = num_morel_deathcaps
        self.appearance_interval = appearance_interval
        self.expiry_time = expiry_time
        self.band_height = band_height
        self.num_oysters = num_oysters
        self.num_oyster_deathcaps = num_oyster_deathcaps

        # An object with regen_delay d is away for d + 1 steps (see _respawn_logic),
        # and randint excludes its upper bound.
        regen = (oyster_regen_delay[0] - 1, oyster_regen_delay[1])
        objects = (
            NoRegenForagaxObject(
                name="morel", reward=morel_reward, collectable=True, color=MOREL_COLOR
            ),
            DefaultForagaxObject(
                name="oyster",
                reward=oyster_reward,
                collectable=True,
                color=OYSTER_COLOR,
                regen_delay=regen,
                random_respawn=True,
            ),
            NoRegenForagaxObject(
                name="morel_deathcap",
                reward=morel_deathcap_reward,
                collectable=True,
                color=DEATHCAP_COLOR,
            ),
            DefaultForagaxObject(
                name="oyster_deathcap",
                reward=oyster_deathcap_reward,
                collectable=True,
                color=DEATHCAP_COLOR,
                regen_delay=regen,
                random_respawn=True,
            ),
            WALL,
        )
        # The frequencies are nominal: they only mark both biomes as food biomes
        # for the biome metrics. reset_env places exact counts itself.
        biome_cells = BIOME_WIDTH * HEIGHT
        biomes = (
            Biome(
                start=(MOREL_BIOME_COLUMNS[0], 0),
                stop=(MOREL_BIOME_COLUMNS[1], HEIGHT),
                object_frequencies=(
                    num_morels / biome_cells,
                    0.0,
                    num_morel_deathcaps / biome_cells,
                    0.0,
                    0.0,
                ),
            ),
            Biome(
                start=(OYSTER_BIOME_COLUMNS[0], 0),
                stop=(OYSTER_BIOME_COLUMNS[1], HEIGHT),
                object_frequencies=(
                    0.0,
                    num_oysters / biome_cells,
                    0.0,
                    num_oyster_deathcaps / biome_cells,
                    0.0,
                ),
            ),
        )
        super().__init__(
            name=name,
            size=(WIDTH, HEIGHT),
            aperture_size=aperture_size,
            objects=objects,
            biomes=biomes,
            nowrap=False,
            observation_type=observation_type,
        )

        wall_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        for column, door_row in DOORS.items():
            wall_mask[:, column] = True
            wall_mask[door_row, column] = False
        self.wall_mask = jnp.array(wall_mask)
        self.morel_biome_mask = self.biome_masks_array[MOREL_BIOME_ID]
        self.oyster_biome_mask = self.biome_masks_array[OYSTER_BIOME_ID]

        # Every placement of the two bands (top rows), each a run of band_height
        # rows that may wrap around the torus, with no row in common.
        placements = []
        for morel_top in range(HEIGHT):
            morel_rows = {(morel_top + i) % HEIGHT for i in range(band_height)}
            for deathcap_top in range(HEIGHT):
                rows = {(deathcap_top + i) % HEIGHT for i in range(band_height)}
                if not (morel_rows & rows):
                    placements.append((morel_top, deathcap_top))
        self.band_placements = jnp.array(placements, dtype=jnp.int32)

    def _sample_bands(
        self, key: jax.Array, morel_band: jax.Array, deathcap_band: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Sample a band placement in which both bands differ from the given ones."""
        allowed = (self.band_placements[:, 0] != morel_band) & (
            self.band_placements[:, 1] != deathcap_band
        )
        idx = jax.random.categorical(key, jnp.where(allowed, 0.0, -jnp.inf))
        return self.band_placements[idx, 0], self.band_placements[idx, 1]

    def _grow(
        self, key: jax.Array, band: jax.Array, num: int, object_id: int
    ) -> jax.Array:
        """Grid with ``num`` objects at random cells of the band, 0 elsewhere."""
        cells = jax.random.permutation(key, self.band_height * BIOME_WIDTH)[:num]
        rows = (band + cells // BIOME_WIDTH) % HEIGHT
        columns = MOREL_BIOME_COLUMNS[0] + cells % BIOME_WIDTH
        grid = jnp.zeros((HEIGHT, WIDTH), dtype=ID_DTYPE)
        return grid.at[rows, columns].set(jnp.array(object_id, dtype=ID_DTYPE))

    def _place(self, object_state: ObjectState, grown: jax.Array) -> ObjectState:
        """Put the non-zero cells of ``grown`` into the object state."""
        new = grown > 0
        return object_state.replace(
            object_id=jnp.where(new, grown, object_state.object_id),
            color=jnp.where(
                new[..., None],
                self.object_colors[grown].astype(COLOR_DTYPE),
                object_state.color,
            ),
        )

    def _sample_interval(self, key: jax.Array) -> jax.Array:
        low, high = self.appearance_interval
        return jax.random.randint(key, (), low, high + 1, dtype=jnp.int32)

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> Tuple[jax.Array, TimedMorelEnvState]:
        """Reset environment state."""
        band_key, interval_key, oyster_key = jax.random.split(key, 3)

        object_state = ObjectState.create_empty(self.size, 3)
        biome_id = jnp.full((HEIGHT, WIDTH), -1, dtype=BIOME_ID_DTYPE)
        biome_id = jnp.where(self.morel_biome_mask, MOREL_BIOME_ID, biome_id)
        biome_id = jnp.where(self.oyster_biome_mask, OYSTER_BIOME_ID, biome_id)
        object_state = object_state.replace(biome_id=biome_id.astype(BIOME_ID_DTYPE))

        # Walls
        walls = jnp.where(self.wall_mask, WALL_ID, 0).astype(ID_DTYPE)
        object_state = self._place(object_state, walls)

        # Oyster biome: exact counts at random cells
        num_objects = self.num_oysters + self.num_oyster_deathcaps
        cells = jax.random.permutation(oyster_key, BIOME_WIDTH * HEIGHT)[:num_objects]
        ids = jnp.where(
            jnp.arange(num_objects) < self.num_oysters, OYSTER_ID, OYSTER_DEATHCAP_ID
        ).astype(ID_DTYPE)
        oyster_grid = jnp.zeros((HEIGHT, WIDTH), dtype=ID_DTYPE)
        oyster_grid = oyster_grid.at[
            cells // BIOME_WIDTH, OYSTER_BIOME_COLUMNS[0] + cells % BIOME_WIDTH
        ].set(ids)
        object_state = self._place(object_state, oyster_grid)

        # The morel biome starts empty; the bands are already in place.
        none = jnp.array(-1, dtype=jnp.int32)
        morel_band, deathcap_band = self._sample_bands(band_key, none, none)

        state = TimedMorelEnvState(
            pos=jnp.array([WIDTH // 2, HEIGHT // 2], dtype=jnp.int32),
            time=jnp.array(0, dtype=jnp.int32),
            offset=jnp.array(0, dtype=jnp.int32),
            digestion_buffer=jnp.zeros((self.max_reward_delay,), dtype=REWARD_DTYPE),
            object_state=object_state,
            biome_state=BiomeState(
                consumption_count=jnp.zeros(2, dtype=ID_DTYPE),
                total_objects=jnp.array(
                    [self.num_morels + self.num_morel_deathcaps, num_objects],
                    dtype=ID_DTYPE,
                ),
                generation=jnp.zeros(2, dtype=ID_DTYPE),
            ),
            next_appearance=self._sample_interval(interval_key),
            expiry=jnp.array(0, dtype=jnp.int32),
            appearance_count=jnp.array(0, dtype=jnp.int32),
            morel_band=morel_band.astype(jnp.int32),
            deathcap_band=deathcap_band.astype(jnp.int32),
        )
        return jax.lax.stop_gradient(
            self.get_obs(state, params)
        ), jax.lax.stop_gradient(state)

    def step_env(
        self,
        key: jax.Array,
        state: TimedMorelEnvState,
        action: Union[int, float, jax.Array],
        params: EnvParams,
    ) -> Tuple[jax.Array, TimedMorelEnvState, jax.Array, jax.Array, Dict[Any, Any]]:
        """Perform single timestep state transition."""
        key, clock_key = jax.random.split(key)
        # Movement, collection and oyster-biome respawn
        _, base, reward, done, info = super().step_env(key, state, action, params)
        state = self._advance_morel_clock(clock_key, state, base)

        reward_grid = self._reward_grid(state, state.object_state)
        if not self.full_world:
            reward_grid = self._get_aperture(reward_grid, state.pos)
        info["rewards"] = reward_grid.astype(jnp.float16)
        # The morel clock, for scripted baselines that are given it
        info["steps_to_appearance"] = state.next_appearance - state.time
        info["steps_to_expiry"] = jnp.maximum(state.expiry - state.time, 0)
        info["appearance_count"] = state.appearance_count
        info["morel_band"] = state.morel_band
        info["deathcap_band"] = state.deathcap_band

        return (
            jax.lax.stop_gradient(self.get_obs(state, params)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def _advance_morel_clock(
        self, key: jax.Array, prev: TimedMorelEnvState, base: EnvState
    ) -> TimedMorelEnvState:
        """Expire and grow the mushrooms of the morel biome, and move the bands."""
        interval_key, band_key, morel_key, deathcap_key = jax.random.split(key, 4)
        time = base.time
        appear = time >= prev.next_appearance
        expire = time >= prev.expiry

        object_state = base.object_state
        object_state = object_state.replace(
            object_id=jnp.where(
                (appear | expire) & self.morel_biome_mask,
                jnp.array(0, dtype=ID_DTYPE),
                object_state.object_id,
            )
        )

        # The bands move after every band_move_every appearances.
        morel_band, deathcap_band = prev.morel_band, prev.deathcap_band
        if self.band_move_every > 0:
            move = (
                appear
                & (prev.appearance_count > 0)
                & (prev.appearance_count % self.band_move_every == 0)
            )
            new_morel_band, new_deathcap_band = self._sample_bands(
                band_key, morel_band, deathcap_band
            )
            morel_band = jnp.where(move, new_morel_band, morel_band)
            deathcap_band = jnp.where(move, new_deathcap_band, deathcap_band)

        grown = self._grow(morel_key, morel_band, self.num_morels, MOREL_ID)
        grown = grown + self._grow(
            deathcap_key, deathcap_band, self.num_morel_deathcaps, MOREL_DEATHCAP_ID
        )
        grown = jnp.where(appear, grown, jnp.array(0, dtype=ID_DTYPE))
        object_state = self._place(object_state, grown)

        return TimedMorelEnvState(
            pos=base.pos,
            time=base.time,
            offset=base.offset,
            digestion_buffer=base.digestion_buffer,
            object_state=object_state,
            biome_state=base.biome_state,
            next_appearance=jnp.where(
                appear, time + self._sample_interval(interval_key), prev.next_appearance
            ).astype(jnp.int32),
            expiry=jnp.where(appear, time + self.expiry_time, prev.expiry).astype(
                jnp.int32
            ),
            appearance_count=(prev.appearance_count + appear).astype(jnp.int32),
            morel_band=morel_band.astype(jnp.int32),
            deathcap_band=deathcap_band.astype(jnp.int32),
        )

    def get_obs(
        self, state: EnvState, params: EnvParams, key=None
    ) -> Union[jax.Array, Dict[str, jax.Array]]:
        """Observation; with fog every cell of the morel biome looks empty."""
        if self.fog:
            object_state = state.object_state
            state = state.replace(
                object_state=object_state.replace(
                    object_id=jnp.where(
                        self.morel_biome_mask,
                        jnp.array(0, dtype=ID_DTYPE),
                        object_state.object_id,
                    )
                )
            )
        return super().get_obs(state, params, key)


ENV_ID = "ForagaxTimedMorel-v1"


def make(
    env_id: str = ENV_ID,
    observation_type: str = "color",
    aperture_size: Union[Tuple[int, int], int] = (5, 5),
    **kwargs: Any,
) -> TimedMorelForagaxEnv:
    """Stand-in for foragax.registry.make(ENV_ID, ...), with the same defaults."""
    if observation_type not in ("object", "rgb", "color"):
        raise ValueError(f"Unknown observation_type: {observation_type}")
    kwargs = {"fog": True, "band_move_every": 1250, **kwargs}
    return TimedMorelForagaxEnv(
        name=env_id,
        observation_type=observation_type,
        aperture_size=aperture_size,
        **kwargs,
    )
