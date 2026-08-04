"""NTK and churn plasticity metrics for PPO agents.

The PPO network ``apply_fn`` has the signature::

    hidden, pi, value = apply_fn(params, hidden, obs)

where ``obs`` is the 6-tuple
``(image, action_encoded, last_reward, sine, cosine, reward_trace)`` (each with
a leading batch axis), ``pi`` is a ``distrax.Categorical`` over ``action_dim``
actions, and ``value`` has shape ``(batch,)``.

Two heads are measured separately:

* **value (critic)** -- scalar value output, the direct analogue of the DQN
  Q-value metrics.
* **policy (actor)** -- the ``action_dim`` policy logits.

For each head we report the NTK Gram-matrix discrete (hard) rank and effective
(stable) rank, plus the per-update *churn*: the norm of the change in the head's
predictions on a fixed reference batch from immediately before to immediately
after one PPO update.  The Gram is built in row-chunks (see ``_gram_chunked``) so
peak memory is bounded by the chunk size rather than the reference-set size.
"""

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp


def build_ref_obs_tuple(x: jnp.ndarray, action_dim: int, reward_dim: int) -> Tuple:
    """Build the 6-tuple obs the PPO network expects for a single reference obs.

    Scalar / context features (action encoding, last reward, sinusoidal time
    encoding, reward trace) are zeroed, mirroring the DQN reference-metric setup
    in ``utils.metrics.compute_ntk_metrics`` which feeds zero scalars.  A leading
    batch axis of size 1 is added so the convolutional / dense layers see the
    rank they expect.

    Args:
        x: A single reference observation image (no batch axis).
        action_dim: Number of discrete actions (size of the action encoding).
        reward_dim: Width of the ``last_reward`` feature (1 for plain envs,
            ``1 + hint_dim`` for hint envs).

    Returns:
        The 6-tuple ``(image, action_encoded, last_reward, sine, cosine,
        reward_trace)``, each with a leading batch axis of size 1.
    """
    image = jnp.expand_dims(x, 0)
    action_encoded = jnp.zeros((1, action_dim))
    last_reward = jnp.zeros((1, reward_dim))
    sine = jnp.zeros((1, 1))
    cosine = jnp.zeros((1, 1))
    reward_trace = jnp.zeros((1, 1))
    return (image, action_encoded, last_reward, sine, cosine, reward_trace)


def _flatten_jacobian(jac_tree: Any, n_rows: int) -> jnp.ndarray:
    """Flatten a Jacobian pytree to a dense ``[n_rows, n_params]`` matrix."""
    leaves = jax.tree_util.tree_leaves(jac_tree)
    flat_leaves = [leaf.reshape(n_rows, -1) for leaf in leaves]
    return jnp.concatenate(flat_leaves, axis=1)


def weight_norm(params: Any) -> jnp.ndarray:
    """Global L2 norm of every parameter leaf in ``params``.

    Returns:
        A scalar array holding ``||theta||_2``.
    """
    leaves = jax.tree_util.tree_leaves(params)
    sq_sum = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
    return jnp.sqrt(sq_sum)


def nan_weight_norm() -> jnp.ndarray:
    """The ``NaN`` weight-norm scalar emitted on non-metric / disabled updates."""
    return jnp.float32(jnp.nan)


def weight_drift(
    params: Any, init_params: Any, labels: Any
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """L2 distance ``||theta - theta_0||`` between current and initial params.

    The drift is reported split by the actor and critic
    trunks -- which in this RTU-PPO architecture share no weights and are trained
    by different objectives -- plus a global total over every leaf.

    Args:
        params: Current (post-update) parameters.
        init_params: The frozen theta_0 snapshot taken right after init.
        labels: Per-leaf label tree ("pi" / "vf" / ...), same structure
            as params.

    Returns:
        (drift_pi, drift_vf, drift_total) as scalar arrays.
    """
    sqdiff = jax.tree_util.tree_map(
        lambda p, p0: jnp.sum(jnp.square(p - p0)), params, init_params
    )
    sq_leaves = jax.tree_util.tree_leaves(sqdiff)
    lbl_leaves = jax.tree_util.tree_leaves(labels)
    pi_sq = sum(
        (s for s, lbl in zip(sq_leaves, lbl_leaves, strict=False) if lbl == "pi"), 0.0
    )
    vf_sq = sum(
        (s for s, lbl in zip(sq_leaves, lbl_leaves, strict=False) if lbl == "vf"), 0.0
    )
    total_sq = sum(sq_leaves, 0.0)
    return jnp.sqrt(pi_sq), jnp.sqrt(vf_sq), jnp.sqrt(total_sq)


def nan_weight_drift() -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The ``NaN`` weight-drift triple emitted on non-metric / disabled updates."""
    nan = jnp.float32(jnp.nan)
    return nan, nan, nan


def _gram_chunked(
    f_single: Callable, params: Any, x_ref: jnp.ndarray, chunk: int, m: int
) -> jnp.ndarray:
    """NTK Gram matrix ``J Jᵀ`` built in row-chunks, never materializing ``J``.

    ``f_single(params, x)`` maps one reference obs to an ``m``-vector head output
    (``m = 1`` for the scalar value head, ``m = action_dim`` for the policy
    logits).  The full Jacobian ``J`` therefore has ``n_ref * m`` rows and one
    column per parameter -- the object whose ``[n_ref*m, n_params]`` size causes
    the OOM when formed all at once.

    Instead we tile the reference set into chunks of ``chunk`` samples and
    compute the Gram blockwise: for each pair of chunks ``(i, j)`` we materialize
    only the two chunk-Jacobians ``J_i, J_j`` (shape ``[chunk*m, n_params]``) and
    their product ``J_i J_jᵀ``.  Peak memory is ``O(chunk * m * n_params +
    (n_ref*m)²)`` -- bounded by ``chunk`` rather than ``n_ref`` -- at the cost of
    recomputing each chunk-Jacobian ``n_chunks`` times.  The result is bit-for-bit
    the dense ``J Jᵀ`` (modulo float reassociation).

    ``chunk`` need not divide ``n_ref``: the reference set is padded up to a whole
    number of chunks and the padded rows are zeroed out of every chunk-Jacobian,
    so they contribute nothing to the Gram (zero rows add only zero eigenvalues,
    leaving rank and effective rank untouched).
    """
    n = x_ref.shape[0]
    chunk = min(chunk, n)
    n_chunks = -(-n // chunk)  # ceil
    pad = n_chunks * chunk - n
    if pad:
        x_pad = jnp.concatenate(
            [x_ref, jnp.zeros((pad, *x_ref.shape[1:]), x_ref.dtype)]
        )
        valid = jnp.concatenate([jnp.ones(n), jnp.zeros(pad)])
    else:
        x_pad, valid = x_ref, jnp.ones(n)
    x_chunks = x_pad.reshape(n_chunks, chunk, *x_ref.shape[1:])
    valid_chunks = valid.reshape(n_chunks, chunk)

    def chunk_jac(xc: jnp.ndarray, vc: jnp.ndarray) -> jnp.ndarray:
        """Flat ``[chunk*m, n_params]`` Jacobian for one chunk, padded rows zeroed."""
        jac = jax.vmap(jax.jacrev(f_single, argnums=0), in_axes=(None, 0))(params, xc)
        flat = _flatten_jacobian(jac, chunk * m)
        row_mask = jnp.repeat(vc, m)  # each sample owns m consecutive rows
        return flat * row_mask[:, None]

    def row_of_blocks(carry_i):
        xi, vi = carry_i
        jac_i = chunk_jac(xi, vi)  # [chunk*m, n_params]

        def block(carry_j):
            xj, vj = carry_j
            jac_j = chunk_jac(xj, vj)
            return jac_i @ jac_j.T  # [chunk*m, chunk*m]

        return jax.lax.map(block, (x_chunks, valid_chunks))  # [n_chunks, cm, cm]

    blocks = jax.lax.map(row_of_blocks, (x_chunks, valid_chunks))
    # blocks[i, j] is the (i, j) Gram block -> reorder to a dense [n_rows, n_rows].
    cm = chunk * m
    n_rows = n_chunks * cm
    return blocks.transpose(0, 2, 1, 3).reshape(n_rows, n_rows)


def _ntk_rank_eff_cond(ntk: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Discrete (hard) NTK rank, effective (stable) rank, and condition number from the Gram matrix.

    ``ntk = J Jᵀ`` is symmetric positive-semidefinite, so a single symmetric
    eigendecomposition (``eigvalsh``) gives every spectral quantity we need --
    cheaper and more numerically appropriate than the two separate SVDs the old
    ``matrix_rank`` + ``cond`` path used.

    * **hard rank** -- count of eigenvalues above ``λ_max * n_rows * eps``, the
      same tolerance ``jnp.linalg.matrix_rank`` applies to the Gram, so this
      matches the previously logged rank semantics exactly.
    * **effective rank** -- the *stable rank* ``(Σλ) / λ_max = ‖J‖_F² / ‖J‖₂²``,
      a smooth, threshold-free measure of how many directions actually carry
      energy.  It equals the hard rank only for a perfectly flat spectrum and is
      otherwise smaller; the gap reveals spectral concentration that the integer
      rank cannot.
    * **condition number** -- the ratio ``λ_max / λ_min`` of the largest to
      smallest nonzero eigenvalue, measuring how well-conditioned the Gram is.
    """
    eigvals = jnp.clip(jnp.linalg.eigvalsh(ntk), a_min=0.0)  # ascending, PSD
    lam_max = eigvals[-1]
    n_rows = ntk.shape[0]
    tol = lam_max * n_rows * jnp.finfo(ntk.dtype).eps
    rank = jnp.sum(eigvals > tol).astype(jnp.float32)
    eff_rank = jnp.where(lam_max > 0, jnp.sum(eigvals) / lam_max, 0.0).astype(
        jnp.float32
    )
    above_tol = eigvals > tol
    lam_min = jnp.where(jnp.any(above_tol), jnp.min(jnp.where(above_tol, eigvals, jnp.inf)), 1.0)
    cond = jnp.where(
        jnp.logical_and(lam_max > 0, lam_min > 0), lam_max / lam_min, 0.0
    ).astype(jnp.float32)
    return rank, eff_rank, cond


def value_metrics(
    apply_fn: Callable,
    params_before: Any,
    params_after: Any,
    init_hstate: Any,
    x_ref: jnp.ndarray,
    action_dim: int,
    reward_dim: int,
    chunk: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """NTK rank / effective rank / condition number and per-update churn for the value head.

    NTK is measured on the post-update parameters (the network's current state,
    matching the DQN convention).  Churn is the norm of the value change on
    ``x_ref`` from ``params_before`` to ``params_after``.  ``chunk`` is the
    row-chunk size for the memory-bounded Gram build; it changes only peak memory
    / speed, not the result.

    Returns:
        ``(rank, eff_rank, cond, churn)`` as scalar arrays.
    """

    def value_of(params, x):
        obs = build_ref_obs_tuple(x, action_dim, reward_dim)
        _, _, value = apply_fn(params, init_hstate, obs)
        return value[0:1]  # (1,) so the Gram builder sees a uniform m-vector head

    # Per-sample predictions before / after the update -> churn numerator.
    pred_before = jax.vmap(value_of, in_axes=(None, 0))(params_before, x_ref)
    pred_after = jax.vmap(value_of, in_axes=(None, 0))(params_after, x_ref)

    # NTK on the current (post-update) params; Gram built in row-chunks.
    ntk = _gram_chunked(value_of, params_after, x_ref, chunk, m=1)
    rank, eff_rank, cond = _ntk_rank_eff_cond(ntk)
    churn_numerator = jnp.linalg.norm(pred_after - pred_before)

    # Gradient-norm aggregate ||J||_F = sqrt(trace(J Jᵀ)) -- read straight off the
    # Gram diagonal, so the full Jacobian is never materialized.
    grad_norm_aggregate = jnp.sqrt(jnp.trace(ntk))

    # Weight update norm: ||ΔΘ|| = ||params_after - params_before||
    leaves_before = jax.tree_util.tree_leaves(params_before)
    leaves_after = jax.tree_util.tree_leaves(params_after)
    delta_leaves = [
        jnp.sum(jnp.square(a - b))
        for a, b in zip(leaves_after, leaves_before, strict=False)
    ]
    weight_update_norm = jnp.sqrt(sum(delta_leaves, 0.0))

    # Relative churn: bounded [0, 1] by Cauchy-Schwarz.
    # Add small epsilon to avoid division by zero when network is completely plastic.
    eps = 1e-8
    churn = churn_numerator / (grad_norm_aggregate * weight_update_norm + eps)

    return rank, eff_rank, cond, churn


def policy_metrics(
    apply_fn: Callable,
    params_before: Any,
    params_after: Any,
    init_hstate: Any,
    x_ref: jnp.ndarray,
    action_dim: int,
    reward_dim: int,
    chunk: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """NTK rank / effective rank / condition number and per-update churn for the policy head.

    The policy output is the ``action_dim`` logit vector, so the Jacobian has
    ``n_ref * action_dim`` rows (and the rank ceiling is correspondingly
    ``action_dim`` times the value head's).  Churn is measured on the action
    probabilities (softmax of the logits), the interpretable notion of how much
    the policy moved on the reference set in one update.  ``chunk`` is the
    row-chunk size for the memory-bounded Gram build; it changes only peak memory
    / speed, not the result.

    Returns:
        ``(rank, eff_rank, cond, churn)`` as scalar arrays.
    """

    def logits_of(params, x):
        obs = build_ref_obs_tuple(x, action_dim, reward_dim)
        _, pi, _ = apply_fn(params, init_hstate, obs)
        return pi.logits[0]  # (action_dim,)

    def probs_of(params, x):
        obs = build_ref_obs_tuple(x, action_dim, reward_dim)
        _, pi, _ = apply_fn(params, init_hstate, obs)
        return pi.probs[0]  # (action_dim,)

    pred_before = jax.vmap(probs_of, in_axes=(None, 0))(params_before, x_ref)
    pred_after = jax.vmap(probs_of, in_axes=(None, 0))(params_after, x_ref)
    churn_numerator = jnp.linalg.norm(pred_after - pred_before)

    # NTK on the current (post-update) params; Gram built in row-chunks, each
    # sample contributing action_dim rows.
    ntk = _gram_chunked(logits_of, params_after, x_ref, chunk, m=action_dim)
    rank, eff_rank, cond = _ntk_rank_eff_cond(ntk)

    # Gradient-norm aggregate ||J||_F = sqrt(trace(J Jᵀ)) -- read straight off the
    # Gram diagonal, so the full Jacobian is never materialized.
    grad_norm_aggregate = jnp.sqrt(jnp.trace(ntk))

    # Weight update norm: ||ΔΘ|| = ||params_after - params_before||
    leaves_before = jax.tree_util.tree_leaves(params_before)
    leaves_after = jax.tree_util.tree_leaves(params_after)
    delta_leaves = [
        jnp.sum(jnp.square(a - b))
        for a, b in zip(leaves_after, leaves_before, strict=False)
    ]
    weight_update_norm = jnp.sqrt(sum(delta_leaves, 0.0))

    # Relative churn: bounded [0, 1] by Cauchy-Schwarz.
    # Add small epsilon to avoid division by zero.
    eps = 1e-8
    churn = churn_numerator / (grad_norm_aggregate * weight_update_norm + eps)

    return rank, eff_rank, cond, churn


def compute_ppo_metrics(
    apply_fn: Callable,
    params_before: Any,
    params_after: Any,
    init_hstate: Any,
    x_ref: jnp.ndarray,
    action_dim: int,
    reward_dim: int,
    chunk: int,
    compute_value: bool = True,
    compute_policy: bool = True,
) -> Tuple[jnp.ndarray, ...]:
    """Compute value- and policy-head NTK + churn metrics.

    Pure JAX and statically shaped so it can be traced inside ``jax.lax.scan``
    / ``jax.lax.cond`` and vmapped across runs.  Heads that are disabled (or
    when this is called on a non-metric step via ``lax.cond``) report ``NaN``.

    The weight norm is intentionally *not* computed here -- it has its own
    cadence and config flag; see ``weight_norm`` / ``nan_weight_norm``.

    Args:
        apply_fn: The network ``apply`` function.
        params_before: Parameters before the current PPO update (for churn).
        params_after: Parameters after the current PPO update (NTK + churn).
        init_hstate: Initial hidden state sized for batch 1 (zeros for RTUs).
        x_ref: Reference observation images, shape ``[n_ref, ...]``.
        action_dim: Number of discrete actions.
        reward_dim: Width of the ``last_reward`` feature.
        chunk: Row-chunk size for the memory-bounded Gram build (result-invariant).
        compute_value: Whether to measure the value head.
        compute_policy: Whether to measure the policy head.

    Returns:
        ``(value_rank, value_eff_rank, value_cond, value_churn, policy_rank,
        policy_eff_rank, policy_cond, policy_churn)`` as scalar arrays.
    """
    nan = jnp.float32(jnp.nan)

    if compute_value:
        v_rank, v_eff_rank, v_cond, v_churn = value_metrics(
            apply_fn,
            params_before,
            params_after,
            init_hstate,
            x_ref,
            action_dim,
            reward_dim,
            chunk,
        )
    else:
        v_rank, v_eff_rank, v_cond, v_churn = nan, nan, nan, nan

    if compute_policy:
        p_rank, p_eff_rank, p_cond, p_churn = policy_metrics(
            apply_fn,
            params_before,
            params_after,
            init_hstate,
            x_ref,
            action_dim,
            reward_dim,
            chunk,
        )
    else:
        p_rank, p_eff_rank, p_cond, p_churn = nan, nan, nan, nan

    return v_rank, v_eff_rank, v_cond, v_churn, p_rank, p_eff_rank, p_cond, p_churn


def nan_ppo_metrics() -> Tuple[jnp.ndarray, ...]:
    """The all-``NaN`` NTK / churn metric tuple emitted on non-metric updates."""
    nan = jnp.float32(jnp.nan)
    return nan, nan, nan, nan, nan, nan, nan, nan


def rtu_r_stats(params: Any, params_type: str = "exp_exp") -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Mean and max spectral radius ``r`` over every RTU diagonal in ``params``.

    Each RTU unit's recurrence is a scaled rotation with ``g = r*cos(theta)`` and
    ``phi = r*sin(theta)``, so ``r = hypot(g, phi)`` recovers the pole magnitude
    regardless of which reparameterization (``direct`` / ``exp`` / ``exp_exp``
    / ...) produced it.  ``r`` sets the unit's memory horizon: the effective time
    constant is ``tau = -1 / ln(r)`` steps, so ``r -> 1`` is long memory and
    ``r -> 0`` is one-step memory.

    Every ``r_param`` leaf in the tree is included, so this aggregates over all
    blocks and over both the actor and critic branches -- the depth of the stack
    does not change the shape of the output.

    Args:
        params: A Flax parameter tree containing one or more RTU cells.
        params_type: The RTU reparameterization, used to map the raw
            ``r_param`` / ``theta_param`` leaves to ``(g, phi)``.

    Returns:
        ``(r_mean, r_max)`` as scalar arrays.  Both are ``NaN`` when the tree
        holds no RTU cell (e.g. a non-recurrent baseline agent).
    """
    from algorithms.nn.rtus.rtus_utils import g_phi_options

    g_phi = g_phi_options[params_type]
    radii = []

    def walk(node):
        if isinstance(node, dict):
            if "r_param" in node and "theta_param" in node:
                g, phi, _ = g_phi(node["r_param"], node["theta_param"])
                radii.append(jnp.ravel(jnp.hypot(g, phi)))
            for child in node.values():
                walk(child)

    walk(params)
    if not radii:
        return jnp.float32(jnp.nan), jnp.float32(jnp.nan)
    all_r = jnp.concatenate(radii)
    return jnp.mean(all_r), jnp.max(all_r)


def nan_rtu_r_stats() -> Tuple[jnp.ndarray, jnp.ndarray]:
    """The ``NaN`` pair emitted on non-metric updates / when disabled."""
    return jnp.float32(jnp.nan), jnp.float32(jnp.nan)
