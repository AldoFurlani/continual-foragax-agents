# Scripted baselines for ForagaxTimedMorel-v1, chosen by the "mode" parameter:
#
# oysters_only  Stays in the oyster biome and walks to the nearest oyster.
# clock_oracle  Knows the morel clock and the doors, but not the growth bands
#               and not what is in the fog. Enters the morel biome as the
#               mushrooms appear and walks to the nearest cell of it that it
#               has not visited, until they expire.
# oracle        Sees through the fog and knows the morel clock. Enters the
#               morel band as the mushrooms appear and collects the morels.
#
# Between visits both oracles forage like oysters_only.
#
# All modes need the true world as the observation: aperture_size=-1,
# observation_type="object" and fog=false. The clock oracle then ignores the
# objects of the morel biome. The clock and the morel band come from the
# environment's info.
from dataclasses import replace
from functools import partial
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from ml_instrumentation.Collector import Collector

import utils.chex as cxu
from algorithms.BaseAgent import BaseAgent
from environments.TimedMorel import (
    HEIGHT,
    MOREL_BIOME_COLUMNS,
    OYSTER_BIOME_COLUMNS,
    WIDTH,
)

MODES = ("oysters_only", "clock_oracle", "oracle")
UNREACHABLE = 1e4
# (rows, columns) moved by each action: down, right, up, left
MOVES = ((1, 0), (0, 1), (-1, 0), (0, -1))


@cxu.dataclass
class AgentState:
    key: jax.Array
    visited: jax.Array  # cells of the morel biome walked during this appearance
    appearance_count: jax.Array
    steps: int = 0


class TimedMorelAgent(BaseAgent):
    def __init__(
        self,
        observations: Tuple[int, ...],
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        self.mode = params["mode"]
        if self.mode not in MODES:
            raise ValueError(f"Unknown mode: {self.mode}")
        if tuple(observations) != (HEIGHT, WIDTH, 6):
            raise ValueError(
                "TimedMorelAgent needs the true world: aperture_size=-1, "
                'observation_type="object" and fog=false'
            )

        columns = jnp.arange(WIDTH)[None, :]
        in_rows = jnp.ones((HEIGHT, 1), dtype=bool)
        self.morel_biome = in_rows & (
            (columns >= MOREL_BIOME_COLUMNS[0]) & (columns < MOREL_BIOME_COLUMNS[1])
        )
        self.oyster_biome = in_rows & (
            (columns >= OYSTER_BIOME_COLUMNS[0]) & (columns < OYSTER_BIOME_COLUMNS[1])
        )
        self.state = AgentState(
            key=self.key,
            visited=jnp.zeros((HEIGHT, WIDTH), dtype=bool),
            appearance_count=jnp.array(0, dtype=jnp.int32),
        )

    def _after_move(self, dist: jax.Array) -> jax.Array:
        """(4, H, W): the value of the cell that each action leads to."""
        return jnp.stack([jnp.roll(dist, (-dy, -dx), axis=(0, 1)) for dy, dx in MOVES])

    def _plan(
        self, pos: jax.Array, targets: jax.Array, blocked: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Steps to the nearest target, and the steps left after each action."""
        dist = jnp.where(targets & ~blocked, 0.0, UNREACHABLE)

        def relax(_, dist):
            closer = jnp.minimum(dist, self._after_move(dist).min(axis=0) + 1)
            return jnp.where(blocked, UNREACHABLE, closer)

        dist = jax.lax.fori_loop(0, HEIGHT * WIDTH // 2, relax, dist)
        after = self._after_move(dist)[:, pos[0], pos[1]]
        return jnp.minimum(dist[pos[0], pos[1]], after.min() + 1), after

    @partial(jax.jit, static_argnums=0)
    def act(
        self,
        state: AgentState,
        obs: jax.Array,
        extra: Optional[Dict[str, jax.Array]] = None,
    ) -> tuple[AgentState, jax.Array]:
        morel, oyster, morel_deathcap, oyster_deathcap, wall = (
            obs[:, :, channel] > 0 for channel in range(5)
        )
        pos = jnp.argwhere(obs[:, :, 5] > 0, size=1)[0]
        key, noise_key = jax.random.split(state.key)
        state = replace(state, key=key, steps=state.steps + 1)

        # Forage: the nearest oyster, or else the way (back) into the oyster biome
        blocked = wall | morel_deathcap | oyster_deathcap
        if self.mode == "oysters_only":
            blocked = blocked | ~self.oyster_biome
        steps, after = self._plan(pos, oyster, blocked)
        _, after_return = self._plan(pos, self.oyster_biome, blocked)
        after = jnp.where(steps < UNREACHABLE, after, after_return)
        # Break ties at random
        after = after + 0.5 * jax.random.uniform(noise_key, (4,))

        if self.mode != "oysters_only" and extra is not None:
            out = extra["steps_to_expiry"] > 0
            if self.mode == "oracle":
                rows = (jnp.arange(HEIGHT) - extra["morel_band"]) % HEIGHT < 3
                band = rows[:, None] & self.morel_biome
                targets = jnp.where(out, morel, band)
            else:
                # It sees the deathcaps of the oyster biome only
                blocked = wall | oyster_deathcap
                new_appearance = extra["appearance_count"] != state.appearance_count
                visited = jnp.where(new_appearance, False, state.visited)
                visited = jnp.where(out, visited.at[pos[0], pos[1]].set(True), visited)
                state = replace(
                    state,
                    visited=visited,
                    appearance_count=extra["appearance_count"].astype(jnp.int32),
                )
                targets = self.morel_biome & jnp.where(out, ~visited, True)

            steps, after_visit = self._plan(pos, targets, blocked)
            # While the mushrooms are out, go if there is time to get there.
            # Before they appear, leave so as to arrive as they do.
            go = jnp.where(
                out,
                steps <= extra["steps_to_expiry"],
                extra["steps_to_appearance"] <= steps,
            )
            # Ties go to the first action, which keeps the walk through the
            # morel biome a tidy sweep: random ties leave single cells behind.
            after = jnp.where(go & (steps < UNREACHABLE), after_visit, after)

        return state, jnp.argmin(after)

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, obs: jax.Array):
        self.state, a = self._start(self.state, obs)
        return a

    @partial(jax.jit, static_argnums=0)
    def _start(self, state: AgentState, obs: jax.Array):
        return self.act(state, obs)

    def step(self, reward: jax.Array, obs: jax.Array, extra: Dict[str, jax.Array]):
        self.state, a = self._step(self.state, reward, obs, extra)
        return a

    @partial(jax.jit, static_argnums=0)
    def _step(
        self,
        state: AgentState,
        reward: jax.Array,
        obs: jax.Array,
        extra: Dict[str, jax.Array],
    ):
        return self.act(state, obs, extra)

    def end(self, reward: jax.Array, extra: Dict[str, jax.Array]):
        self.state = self._end(self.state, reward, extra)

    @partial(jax.jit, static_argnums=0)
    def _end(self, state, reward: jax.Array, extra: Dict[str, jax.Array]):
        return state
