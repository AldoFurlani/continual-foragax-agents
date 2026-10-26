"""Tests for DRQN's R2D2-style Q-value discrepancy metric (delta_q_metric)."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_instrumentation.Collector import Collector
from ml_instrumentation.Sampler import Ignore

from algorithms.nn.DRQN import DRQN
from experiment import ExperimentModel
from problems.registry import getProblem

EXP = "experiments/E145-buffer-size/foragax-sweep/ForagaxBig-v5/9/DRQN_LN_HINT_B1K.json"


def _filled_agent(burn_in_steps) -> DRQN:
    exp = ExperimentModel.load(EXP)
    problem = getProblem(exp.problem)(exp, 0, Collector(config={}, default=Ignore()))
    # No updates, so every stored state comes from the parameters the loss sees.
    problem.params["freeze_steps"] = 0
    problem.params["burn_in_steps"] = burn_in_steps
    agent = cast(DRQN, problem.getAgent())

    shapes = cast(dict, problem.observations)
    image_shape, hint_shape = shapes["image"], shapes["hint"]
    rng = np.random.default_rng(0)

    def obs():
        return {
            "image": jnp.asarray(rng.random(image_shape), jnp.float32),
            "hint": jnp.asarray(rng.random(hint_shape), jnp.float32),
        }

    agent.start(obs())
    for _ in range(300):
        agent.step(jnp.float32(rng.random()), obs(), {})
    return agent


def _delta_q(agent: DRQN, params):
    batch = agent.buffer.sample(agent.state.buffer_state, jax.random.key(1))
    _, metrics = agent._loss(params, agent.state.target_params, batch.experience, batch.probabilities)
    return float(metrics["delta_q_first"]), float(metrics["delta_q_last"])


@pytest.mark.parametrize("burn_in_steps", [0, 16])
def test_delta_q_is_zero_when_params_match_the_stored_states(burn_in_steps):
    agent = _filled_agent(burn_in_steps)
    first, last = _delta_q(agent, agent.state.params)
    assert first == pytest.approx(0.0, abs=1e-5)
    assert last == pytest.approx(0.0, abs=1e-5)


def test_delta_q_grows_once_params_drift_from_the_stored_states():
    agent = _filled_agent(16)
    leaves, treedef = jax.tree.flatten(agent.state.params)
    keys = jax.random.split(jax.random.key(2), len(leaves))
    drifted = jax.tree.unflatten(
        treedef, [x + 0.1 * jax.random.normal(k, x.shape) for x, k in zip(leaves, keys, strict=True)]
    )
    # Every stored state came from the old parameters, so both ends move.
    first, last = _delta_q(agent, drifted)
    assert first > 1e-3
    assert last > 1e-3
