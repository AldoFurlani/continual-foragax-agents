import os
import sys

sys.path.append(os.getcwd())
sys.path.insert(0, os.path.abspath("/tmp/src"))
import argparse
import gc
import logging
import lzma
import pickle
import socket
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.utils.save_video import save_video
from jax.tree_util import tree_map
from ml_instrumentation.Collector import Collector
from ml_instrumentation.metadata import attach_metadata
from ml_instrumentation.Sampler import Ignore
from PyExpUtils.results.tools import getParamsAsDict
from tqdm.auto import tqdm

from environments.Foragax import Foragax
from experiment import ExperimentModel
from problems.registry import getProblem
from utils.checkpoint import Checkpoint
from utils.dqn_metrics import DQNMetricsComputer
from utils.preempt import TimeoutHandler
from utils.rlglue import RlGlue


UNROLL = 1


# ------------------
# -- Command Args --
# ------------------
def parse_indices(index_specs: list[str], total: int | None = None) -> list[int]:
    indices = []
    for spec in index_specs:
        if ":" not in spec:
            indices.append(int(spec))
            continue

        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid index slice '{spec}', expected START:STOP")

        start_s, stop_s = parts
        start = int(start_s) if start_s else 0
        if stop_s:
            stop = int(stop_s)
        elif total is not None:
            stop = total
        else:
            raise ValueError(f"Open-ended index slice '{spec}' requires total runs")

        indices.extend(range(start, stop))

    return indices


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp", type=str, required=True)
parser.add_argument("-i", "--idxs", nargs="+", type=str, required=True)
parser.add_argument("--save_path", type=str, default="./")
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/")
parser.add_argument("--silent", action="store_true", default=False)
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--max_steps", type=int, default=None)

args = parser.parse_args()

# ---------------------------
# -- Library Configuration --
# ---------------------------
if not args.gpu:
    jax.config.update("jax_platform_name", "cpu")

logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)
logger = logging.getLogger("exp")
prod = "cdr" in socket.gethostname() or args.silent
if not prod:
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)


# ----------------------
# -- Experiment Def'n --
# ----------------------
timeout_handler = TimeoutHandler()

exp = ExperimentModel.load(args.exp)
try:
    indices = parse_indices(args.idxs, exp.numPermutations())
except ValueError as e:
    parser.error(str(e))

Problem = getProblem(exp.problem)

# --------------------
# -- Batch Set-up --
# --------------------
start_time = time.time()

collectors = []
glues = []
chks = []
first_hypers = None
for idx in indices:
    chk = Checkpoint(exp, idx, base_path=args.checkpoint_path)
    chk.load_if_exists()
    timeout_handler.before_cancel(chk.save)
    chks.append(chk)

    collector = chk.build(
        "collector",
        lambda: Collector(
            # specify which keys to actually store and ultimately save
            # Options are:
            #  - Identity() (save everything)
            #  - Window(n)  take a window average of size n
            #  - Subsample(n) save one of every n elements
            config={},
            # by default, ignore keys that are not explicitly listed above
            default=Ignore(),
        ),
    )
    collector.set_experiment_id(idx)
    collectors.append(collector)

    run = exp.getRun(idx)

    # set random seeds accordingly
    hypers = exp.get_hypers(idx)
    if not first_hypers:
        first_hypers = hypers

    # validate that shape changing hypers are static.
    assert hypers.get("batch") == first_hypers.get("batch")
    assert hypers.get("buffer_size") == first_hypers.get("buffer_size")
    assert hypers.get("buffer_min_size") == first_hypers.get("buffer_min_size")
    assert hypers.get("environment", {}).get("aperture_size") == first_hypers.get(
        "environment", {}
    ).get("aperture_size")
    assert hypers.get("n_step") == first_hypers.get("n_step")
    assert hypers.get("optimizer", {}).get("name") == first_hypers.get(
        "optimizer", {}
    ).get("name")
    assert hypers.get("representation", {}).get("type") == first_hypers.get(
        "representation", {}
    ).get("type")
    assert hypers.get("representation", {}).get("hidden") == first_hypers.get(
        "representation", {}
    ).get("hidden")

    # build stateful things and attach to checkpoint
    problem = chk.build("p", lambda: Problem(exp, idx, collector))
    agent = chk.build("a", problem.getAgent)
    env = chk.build("e", problem.getEnvironment)

    glue = chk.build("glue", lambda: RlGlue(agent, env))
    glues.append(glue)

assert first_hypers is not None

render_mode = first_hypers.get("experiment", {}).get("render_mode", "world_reward")


def render(carry):
    return glues[0].environment.env.render(
        carry.env_state.state, None, render_mode=render_mode
    )


if hasattr(glues[0].agent, "_step_without_update"):
    v_start = jax.vmap(glues[0]._start)
    v_step = jax.vmap(glues[0]._step)
    v_step_skip = jax.vmap(glues[0]._step_without_update)
    v_update_agent = jax.vmap(glues[0].agent._update_state_with_metrics_if_can_sample)
    v_advance_agent = jax.vmap(glues[0].agent._advance_update_clock)

    @jax.jit
    def v_step_no_update(state):
        state, interaction = v_step_skip(state)
        agent_state = v_advance_agent(state.agent_state)
        state = replace(state, agent_state=agent_state)
        return state, interaction

    # Unlike v_step_no_update, this does not advance the agent's update
    # clock -- once state.steps >= freeze_steps, NNAgent._maybe_update_if_not_frozen
    # returns the state completely unchanged (no clock advance either), so
    # this matches that behavior exactly for steps that are already frozen.
    v_step_frozen = jax.jit(v_step_skip)

    @jax.jit
    def v_step_update(state):
        state, interaction = v_step_skip(state)
        agent_state = v_update_agent(state.agent_state)
        state = replace(state, agent_state=agent_state)
        agent_state = v_advance_agent(state.agent_state)
        state = replace(state, agent_state=agent_state)
        return state, interaction

    v_render = jax.vmap(render)
    use_explicit_update_steps = True
elif len(glues) > 1:
    # vmap glue methods
    v_start = jax.vmap(glues[0]._start)
    v_step = jax.vmap(glues[0]._step)
    v_render = jax.vmap(render)
    use_explicit_update_steps = False
else:
    v_start = glues[0]._start
    v_step = glues[0]._step
    v_render = render
    use_explicit_update_steps = False

total_setup_time = time.time() - start_time
num_indices = len(indices)
logger.debug("--- Batch Set-up Timings ---")
logger.debug(
    f"Total setup time: {total_setup_time:.4f}s | Average: {total_setup_time / num_indices:.4f}s"
)

n = exp.total_steps
if args.max_steps is not None:
    n = args.max_steps

video_length = first_hypers.get("experiment", {}).get("video_length", 0)
video_idxs = first_hypers.get("experiment", {}).get("video_idxs", [0])
if not any(idx in video_idxs for idx in indices) and num_indices > 1:
    video_length = 0

# --------------------
# -- Batch Execution --
# --------------------

start_step = None
save_every = first_hypers.get("experiment", {}).get("save_every", 10_001_000)
video_every = first_hypers.get("experiment", {}).get("video_every", save_every)
ntk_freq = int(first_hypers.get("experiment", {}).get("ntk_freq", 0))
agent_metrics_obj = getattr(glues[0].agent.state, "metrics", None)
METRIC_NAMES: tuple[str, ...] = (
    tuple(f.name for f in fields(agent_metrics_obj))
    if agent_metrics_obj is not None
    else ()
)


def get_agent_metrics(agent_state):
    metrics = getattr(agent_state, "metrics", None)
    if metrics is None:
        return {}
    return {k: getattr(metrics, k) for k in METRIC_NAMES}


def reset_datas():
    fresh_datas = {"rewards": np.empty((len(indices), n), dtype=np.float16)}
    for k in METRIC_NAMES:
        fresh_datas[k] = np.empty((len(indices), n), dtype=np.float16)
    if ntk_freq > 0:
        fresh_datas["churn_norm"] = np.full((len(indices), n), np.nan, dtype=np.float16)
        fresh_datas["ntk_rank"] = np.full((len(indices), n), np.nan, dtype=np.float16)
        fresh_datas["ntk_cond"] = np.full((len(indices), n), np.nan, dtype=np.float32)
    if isinstance(glues[0].environment, Foragax):
        fresh_datas["pos"] = np.empty((len(indices), n, 2), dtype=np.int32)
        fresh_datas["biome_id"] = np.empty((len(indices), n), dtype=np.int32)
        fresh_datas["biome_regret"] = np.empty((len(indices), n), dtype=np.float16)
        fresh_datas["biome_rank"] = np.empty((len(indices), n), dtype=np.int32)
        fresh_datas["object_collected_id"] = np.empty((len(indices), n), dtype=np.int32)
        fresh_datas["hint"] = np.full((len(indices), n), -1, dtype=np.int32)
        if "Weather" in glues[0].environment.env.name:
            fresh_datas["temperatures"] = np.empty(
                (len(indices), n, len(glues[0].environment.env.objects)),
                dtype=np.float16,
            )
    return fresh_datas


datas = reset_datas()


if isinstance(glues[0].environment, Foragax):

    def get_data(carry, interaction):
        data = {
            "rewards": interaction.reward,
            "pos": carry.env_state.state.pos,
            "biome_id": interaction.extra["biome_id"],
            "biome_regret": interaction.extra["biome_regret"],
            "biome_rank": interaction.extra["biome_rank"],
            "object_collected_id": interaction.extra["object_collected_id"],
            **get_agent_metrics(carry.agent_state),
        }

        if isinstance(interaction.obs, Mapping) and "hint" in interaction.obs:
            hint = interaction.obs["hint"]
            # hint is a one hot vector, store -1 if it is all zero.
            # store index of 1 if it is one-hot.
            hint_val = jnp.where(
                jnp.any(hint > 0, axis=-1), jnp.argmax(hint, axis=-1), -1
            )
            data["hint"] = hint_val
        elif "hint" in datas:
            data["hint"] = jnp.full(interaction.reward.shape, -1, dtype=jnp.int32)

        if "Weather" in glues[0].environment.env.name:
            data["temperatures"] = interaction.extra["temperatures"]
        return data
else:

    def get_data(carry, interaction):
        return {
            "rewards": interaction.reward,
            **get_agent_metrics(carry.agent_state),
        }


initial_glue_states = [g.state for g in glues]
loaded_checkpoints = [False] * len(indices)
checkpoint_load_failed = False


for i, idx in enumerate(indices):
    context = exp.buildSaveContext(idx, base=args.checkpoint_path)
    step_path = context.resolve(f"{idx}/step.txt")
    glue_state_path = context.resolve(f"{idx}/glue_state.pkl.xz")
    data_path = context.resolve(f"{idx}/data.npz")
    if os.path.exists(step_path):
        with open(step_path, "r") as f:
            try:
                start_step_idx = int(f.read())
                if start_step is None:
                    start_step = start_step_idx
                else:
                    assert start_step == start_step_idx

                with lzma.open(glue_state_path, "rb") as f:
                    glues[i].state = pickle.load(f)

                with np.load(data_path) as data_idx:
                    for key in data_idx.keys():
                        datas[key][i, : len(data_idx[key])] = data_idx[key]
                loaded_checkpoints[i] = True
            except Exception:
                traceback.print_exc()
                logger.warning(
                    f"Failed to load checkpoint for index {idx}, starting fresh"
                )
                start_step = None
                checkpoint_load_failed = True
                datas = reset_datas()

if checkpoint_load_failed or (any(loaded_checkpoints) and not all(loaded_checkpoints)):
    missing = [idx for idx, loaded in zip(indices, loaded_checkpoints) if not loaded]
    logger.warning(
        "Not all selected indices had valid checkpoints; starting the whole batch fresh. "
        f"Missing checkpoint indices: {missing}"
    )
    start_step = None
    for glue, initial_state in zip(glues, initial_glue_states):
        glue.state = initial_state
    datas = reset_datas()

if len(glues) > 1:
    # combine states
    glue_states = tree_map(lambda *leaves: jnp.stack(leaves), *[g.state for g in glues])
elif use_explicit_update_steps:
    glue_states = tree_map(lambda leaf: jnp.expand_dims(leaf, 0), glues[0].state)
else:
    glue_states = glues[0].state

if start_step is None:
    glue_states, _ = v_start(glue_states)
    start_step = 0
else:
    logger.debug(f"Loaded checkpoints, resuming from step {start_step}")

# Initialize DQN metrics computer and collect reference observations
metrics_computer = DQNMetricsComputer(glues, datas, ntk_freq)
if ntk_freq > 0:
    x_ref_steps = first_hypers.get("experiment", {}).get("x_ref_steps", 500)
    metrics_computer.collect_x_ref(glue_states, v_step, x_ref_steps)

current_step = start_step
training_pbar = tqdm(
    total=n,
    initial=current_step,
    desc="Training",
    unit="step",
    disable=args.silent,
)


def update_training_progress(step_count):
    step_count = int(step_count)
    remaining = training_pbar.total - training_pbar.n
    if remaining > 0:
        training_pbar.update(min(step_count, remaining))


def scan_progress(num_iters: int, unit_scale: int = 1):
    if args.silent:
        return lambda scan_fn: scan_fn

    print_rate = max(min(num_iters // 100, 1000), 1)

    def wrap(scan_fn):
        def wrapped(carry, x):
            result = scan_fn(carry, x)
            iter_num = x[0] if isinstance(x, tuple) else x
            next_iter = iter_num + 1
            should_update = (next_iter % print_rate == 0) | (next_iter == num_iters)
            iter_delta = jnp.where(
                next_iter == num_iters,
                ((num_iters - 1) % print_rate) + 1,
                print_rate,
            )
            step_delta = iter_delta * unit_scale

            def do_update(_):
                jax.debug.callback(
                    update_training_progress,
                    step_delta,
                    ordered=True,
                )
                return None

            jax.lax.cond(should_update, do_update, lambda _: None, operand=None)
            return result

        return wrapped

    return wrap


# Defined outside the outer milestone loop so trace overhead isn't paid
# on every save/video iteration.
update_freq = int(first_hypers.get("update_freq", 1))
freeze_steps = first_hypers.get("freeze_steps", np.inf)
if freeze_steps is None:
    freeze_steps = np.inf
freeze_steps = float(freeze_steps)


def step(carry, _):
    carry, interaction = v_step(carry)
    data = get_data(carry, interaction)
    return carry, data


def video_step(carry, _):
    frame = v_render(carry)
    carry, interaction = v_step(carry)
    data = get_data(carry, interaction)
    return carry, (data, frame)


no_update_step: Optional[Callable[..., Any]] = None
frozen_step: Optional[Callable[..., Any]] = None
update_fire_step: Optional[Callable[..., Any]] = None
update_block: Optional[Callable[..., Any]] = None
macro_block: Optional[Callable[..., Any]] = None
periodic_freq: Optional[int] = None
blocks_per_macro: Optional[int] = None

if use_explicit_update_steps:

    def _no_update_step(carry, _):
        carry, interaction = v_step_no_update(carry)
        data = get_data(carry, interaction)
        return carry, data

    no_update_step = _no_update_step

    def _frozen_step(carry, _):
        carry, interaction = v_step_frozen(carry)
        data = get_data(carry, interaction)
        return carry, data

    frozen_step = _frozen_step

    def _update_fire_step(carry, _):
        carry, interaction = v_step_update(carry)
        data = get_data(carry, interaction)
        return carry, data

    update_fire_step = _update_fire_step

    def _update_block(carry, _):
        carry, interaction = v_step_update(carry)
        update_data = get_data(carry, interaction)
        carry, skip_data = jax.lax.scan(
            _no_update_step,
            carry,
            jnp.arange(update_freq - 1),
            unroll=UNROLL,
        )
        block_data = tree_map(
            lambda first, rest: jnp.concatenate(
                [jnp.expand_dims(first, 0), rest], axis=0
            ),
            update_data,
            skip_data,
        )
        return carry, block_data

    update_block = _update_block

    periodic_freq = getattr(glues[0].agent, "periodic_freq", None)
    if periodic_freq is not None:
        assert periodic_freq % update_freq == 0, (
            f"agent.periodic_freq ({periodic_freq}) must be divisible "
            f"by update_freq ({update_freq})"
        )
        blocks_per_macro = periodic_freq // update_freq
        _v_periodic = jax.vmap(glues[0].agent._periodic_step)
        _blocks_per_macro = blocks_per_macro

        def _macro_block(carry, _):
            carry, blocks_data = jax.lax.scan(
                _update_block,
                carry,
                jnp.arange(_blocks_per_macro),
                unroll=UNROLL,
            )
            new_agent_state = _v_periodic(carry.agent_state)
            carry = replace(carry, agent_state=new_agent_state)
            return carry, blocks_data

        macro_block = _macro_block


def run_explicit_steps(
    glue_states, agent_step_count, step_count, current_step, fire_trailing=False
):
    assert no_update_step is not None and update_block is not None
    data_chunks = []
    steps_remaining = step_count

    steps_mod = agent_step_count % update_freq
    prefix_count = min((update_freq - steps_mod) % update_freq, steps_remaining)
    if prefix_count > 0:
        prefix_step = scan_progress(prefix_count)(no_update_step)
        glue_states, prefix_data = jax.lax.scan(
            prefix_step, glue_states, jnp.arange(prefix_count), unroll=UNROLL
        )
        data_chunks.append(prefix_data)
        steps_remaining -= prefix_count

    block_count = steps_remaining // update_freq
    if block_count > 0:
        if blocks_per_macro is not None and block_count >= blocks_per_macro:
            assert periodic_freq is not None and macro_block is not None
            assert current_step % periodic_freq == 0, (
                f"current_step ({current_step}) is not aligned to "
                f"periodic_freq ({periodic_freq}); set save_every and "
                f"video_every to multiples of periodic_freq, and only "
                f"resume from checkpoints at multiples of periodic_freq"
            )
            n_macro = block_count // blocks_per_macro
            macro_steps_per_iter = blocks_per_macro * update_freq
            macro_steps = n_macro * macro_steps_per_iter
            glue_states, macro_data = jax.lax.scan(
                scan_progress(n_macro, unit_scale=macro_steps_per_iter)(
                    macro_block
                ),
                glue_states,
                jnp.arange(n_macro),
            )
            macro_data = tree_map(
                lambda x: x.reshape((macro_steps, *x.shape[3:])),
                macro_data,
            )
            data_chunks.append(macro_data)
            steps_remaining -= macro_steps

            trailing_blocks = block_count - n_macro * blocks_per_macro
            if trailing_blocks > 0:
                glue_states, trailing_data = jax.lax.scan(
                    scan_progress(trailing_blocks, unit_scale=update_freq)(
                        update_block
                    ),
                    glue_states,
                    jnp.arange(trailing_blocks),
                )
                trailing_data = tree_map(
                    lambda x: x.reshape(
                        (trailing_blocks * update_freq, *x.shape[2:])
                    ),
                    trailing_data,
                )
                data_chunks.append(trailing_data)
                steps_remaining -= trailing_blocks * update_freq
        else:
            glue_states, block_data = jax.lax.scan(
                scan_progress(block_count, unit_scale=update_freq)(
                    update_block
                ),
                glue_states,
                jnp.arange(block_count),
            )
            block_data = tree_map(
                lambda x: x.reshape((block_count * update_freq, *x.shape[2:])),
                block_data,
            )
            data_chunks.append(block_data)
            steps_remaining -= block_count * update_freq

    if steps_remaining > 0 and fire_trailing:
        # The trailing partial block's first step is update-aligned (the
        # prefix + full blocks above preserve alignment), and this segment
        # butts right up against the freeze boundary -- there is no later
        # chunk left to catch this update up in, so if we don't fire it here
        # it is permanently lost.
        assert update_fire_step is not None
        glue_states, fire_data = jax.lax.scan(
            scan_progress(1)(update_fire_step),
            glue_states,
            jnp.arange(1),
            unroll=UNROLL,
        )
        data_chunks.append(fire_data)
        steps_remaining -= 1

    if steps_remaining > 0:
        glue_states, tail_data = jax.lax.scan(
            scan_progress(steps_remaining)(no_update_step),
            glue_states,
            jnp.arange(steps_remaining),
            unroll=UNROLL,
        )
        data_chunks.append(tail_data)

    if not data_chunks:
        return glue_states, None

    data_chunk = (
        data_chunks[0]
        if len(data_chunks) == 1
        else tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *data_chunks)
    )
    return glue_states, data_chunk


while current_step < n:
    next_save = ((current_step // save_every) + 1) * save_every
    next_video = ((current_step // video_every) + 1) * video_every
    # Add ntk_freq as a milestone so metrics are computed with exact per-step states
    next_ntk = ((current_step // ntk_freq) + 1) * ntk_freq if ntk_freq > 0 else n + 1
    next_milestone = min(next_save, next_video, next_ntk, n)
    steps_in_iter = next_milestone - current_step

    if steps_in_iter <= 0:
        break

    # We only record video if this milestone reaches a video frequency
    is_video_milestone = next_milestone % video_every == 0
    if is_video_milestone and video_length > 0:
        no_video_steps_count = max(steps_in_iter - video_length, 0)
        video_steps_count = min(steps_in_iter, video_length)
    else:
        no_video_steps_count = steps_in_iter
        video_steps_count = 0

    data_chunk = None
    if no_video_steps_count > 0:
        agent_step_count = int(np.asarray(glue_states.agent_state.steps).reshape(-1)[0])
        # A single outer-loop chunk can straddle the freeze boundary (e.g.
        # whenever save_every/video_every are left at their defaults, the
        # entire run is one chunk). Split it into a pre-freeze part (explicit
        # update-skipping path) and a post-freeze part (cheap frozen path) --
        # treating the chunk as all-or-nothing here would silently disable
        # the explicit-update fast path for the *entire* run, not just the
        # steps after freezing.
        pre_freeze_count = int(min(no_video_steps_count, max(0.0, freeze_steps - agent_step_count)))
        post_freeze_count = no_video_steps_count - pre_freeze_count

        pre_chunk = None
        if pre_freeze_count > 0:
            if use_explicit_update_steps and update_freq > 1:
                glue_states, pre_chunk = run_explicit_steps(
                    glue_states,
                    agent_step_count,
                    pre_freeze_count,
                    current_step,
                    fire_trailing=post_freeze_count > 0,
                )
            else:
                glue_states, pre_chunk = jax.lax.scan(
                    scan_progress(pre_freeze_count)(step),
                    glue_states,
                    jnp.arange(pre_freeze_count),
                    unroll=UNROLL,
                )

        post_chunk = None
        if post_freeze_count > 0:
            # Once frozen, no update ever happens again (see
            # NNAgent._maybe_update_if_not_frozen), so this can always use the
            # cheap frozen step regardless of update_freq. Without this, the
            # loop would fall back to the generic `step()` path, whose freeze
            # check is a `lax.cond` that degrades to `lax.select` under vmap
            # -- it computes (and discards) a full update on every step.
            if use_explicit_update_steps:
                assert frozen_step is not None
                glue_states, post_chunk = jax.lax.scan(
                    scan_progress(post_freeze_count)(frozen_step),
                    glue_states,
                    jnp.arange(post_freeze_count),
                    unroll=UNROLL,
                )
            else:
                glue_states, post_chunk = jax.lax.scan(
                    scan_progress(post_freeze_count)(step),
                    glue_states,
                    jnp.arange(post_freeze_count),
                    unroll=UNROLL,
                )

        if pre_chunk is not None and post_chunk is not None:
            data_chunk = tree_map(
                lambda a, b: jnp.concatenate([a, b], axis=0), pre_chunk, post_chunk
            )
        else:
            data_chunk = pre_chunk if pre_chunk is not None else post_chunk

    frames = None
    data_chunk_video = None
    if video_steps_count > 0:
        video_steps = jnp.arange(video_steps_count)
        glue_states, (data_chunk_video, frames) = jax.lax.scan(
            scan_progress(video_steps_count)(video_step),
            glue_states,
            video_steps,
            unroll=UNROLL,
        )
        if data_chunk is None:
            data_chunk = data_chunk_video
        else:
            data_chunk = tree_map(
                lambda a, b: jnp.concatenate([a, b], axis=0),
                data_chunk,
                data_chunk_video,
            )
        frames = np.asarray(frames)
        if len(glues) < 2 and use_explicit_update_steps:
            frames = frames[:, 0]

    # checkpointing
    if len(glues) < 2 and not use_explicit_update_steps:
        data_chunk = tree_map(lambda x: np.expand_dims(x, 1), data_chunk)

    for i, idx in enumerate(indices):
        data_idx = tree_map(lambda x: x[:, i], data_chunk)
        for key in datas:
            if key in data_idx:
                datas[key][i, current_step : current_step + steps_in_iter] = data_idx[key]

        # Compute NTK metrics at regular intervals throughout the chunk
        if len(glues) > 1 or use_explicit_update_steps:
            glue_state_idx = tree_map(lambda x: x[i], glue_states)
        else:
            glue_state_idx = glue_states

        # Compute metrics at ntk_freq boundaries
        if metrics_computer.should_compute(next_milestone):
            metrics_computer.compute_and_store(i, next_milestone, glue_state_idx)

        if next_milestone % save_every == 0 and next_milestone < n:
            checkpoint_start_time = time.time()
            context = exp.buildSaveContext(idx, base=args.checkpoint_path)

            # Write to temporary files first for atomic checkpointing
            glue_state_path_tmp = context.resolve(f"{idx}/glue_state.pkl.xz.tmp")
            data_path_tmp = context.resolve(f"{idx}/data.tmp.npz")
            step_path_tmp = context.resolve(f"{idx}/step.txt.tmp")

            # Ensure directories exist
            context.ensureExists(glue_state_path_tmp, is_file=True)
            context.ensureExists(data_path_tmp, is_file=True)
            context.ensureExists(step_path_tmp, is_file=True)

            # Prepare checkpoint data
            if len(glues) > 1 or use_explicit_update_steps:
                glue_state_idx = tree_map(lambda x: x[i], glue_states)
            else:
                glue_state_idx = glue_states

            data_to_save = tree_map(lambda d: d[i, :next_milestone], datas)

            # Write temporary files
            with lzma.open(glue_state_path_tmp, "wb") as f:
                pickle.dump(glue_state_idx, f)

            np.savez_compressed(data_path_tmp, **data_to_save)

            with open(step_path_tmp, "w") as f:
                f.write(str(next_milestone))

            # Atomically rename temporary files to final names
            final_glue_state_path = context.resolve(f"{idx}/glue_state.pkl.xz")
            final_data_path = context.resolve(f"{idx}/data.npz")
            final_step_path = context.resolve(f"{idx}/step.txt")

            os.rename(glue_state_path_tmp, final_glue_state_path)
            os.rename(data_path_tmp, final_data_path)
            os.rename(step_path_tmp, final_step_path)
            checkpoint_time = time.time() - checkpoint_start_time
            logger.debug(
                f"Checkpointed {idx} at {next_milestone} in {checkpoint_time:.4f}s"
            )

        # Save video
        if (
            (next_milestone % video_every == 0 or next_milestone == n)
            and video_length
            and frames is not None
        ):
            if idx in video_idxs:
                video_context = exp.buildSaveContext(idx, base=args.save_path)
                video_path = video_context.resolve(f"videos/{idx}")
                video_context.ensureExists(video_path, is_file=True)

                frames_list = list(frames)
                logger.debug(
                    f"Saving {next_milestone - len(frames_list)}_{next_milestone} video to {video_path}"
                )
                save_video(
                    frames_list,
                    video_path,
                    name_prefix=f"{next_milestone - len(frames_list)}_{next_milestone}",
                    fps=8,
                )

    current_step = next_milestone
    if "frames" in locals():
        del frames
    if "data_chunk" in locals():
        del data_chunk
    if "data_chunk_video" in locals():
        del data_chunk_video
    gc.collect()

training_pbar.close()

# --------------------
# -- Saving --
# --------------------
total_numpy_time = 0
total_db_time = 0
num_indices = len(indices)
for i, idx in enumerate(indices):
    collector = collectors[i]

    # ------------
    # -- Saving --
    # ------------
    context = exp.buildSaveContext(idx, base=args.save_path)
    save_path = context.resolve("results.db")
    data_path = context.resolve(f"data/{idx}.npz")
    context.ensureExists(data_path, is_file=True)

    start_time = time.time()
    data_to_save = tree_map(lambda d: d[i], datas)
    np.savez_compressed(data_path, **data_to_save)
    total_numpy_time += time.time() - start_time

    meta = getParamsAsDict(exp, idx)
    meta |= {"seed": exp.getRun(idx)}
    attach_metadata(save_path, idx, meta)

    start_time = time.time()
    collector.merge(context.resolve("results.db"))
    total_db_time += time.time() - start_time

    collector.close()

logger.debug("--- Saving Timings ---")
logger.debug(
    f"Total numpy save time: {total_numpy_time:.4f}s | Average: {total_numpy_time / num_indices:.4f}s"
)
logger.debug(
    f"Total db save time: {total_db_time:.4f}s | Average: {total_db_time / num_indices:.4f}s"
)
total_save_time = total_numpy_time + total_db_time
logger.debug(
    f"Total save time: {total_save_time:.4f}s | Average: {total_save_time / num_indices:.4f}s"
)
