import jax
import jax.numpy as jnp
import polars as pl


def compute_churn(agent, x_ref, pred_before):
    """Compute churn: norm of output change on reference data after training update.

    Args:
        agent: The agent with updated state
        x_ref: Reference batch of observations
        pred_before: Predictions on x_ref before the update

    Returns:
        Churn norm (scalar)
    """
    # Compute predictions with updated agent state
    pred_after = agent._values(agent.state, x_ref)

    # Handle dict output (multi-head agents)
    if isinstance(pred_after, dict):
        pred_after = pred_after.get("values", pred_after.get("v", list(pred_after.values())[0]))
    if isinstance(pred_before, dict):
        pred_before = pred_before.get("values", pred_before.get("v", list(pred_before.values())[0]))

    # Compute norm of output change
    churn_norm = jnp.linalg.norm(pred_after - pred_before)
    return churn_norm


def compute_ntk_metrics(agent, x_ref_sample, scalars=None):
    """Compute NTK matrix rank and condition number.

    Args:
        agent: The agent with current state
        x_ref_sample: Sample of reference observations (shape: [n_samples, ...])
        scalars: Optional scalar features for each sample (shape: [n_samples, num_features])

    Returns:
        Tuple of (rank, condition_number) as scalars
    """
    state = agent.state

    # Create dummy scalars if not provided
    if scalars is None:
        scalars = jnp.zeros((x_ref_sample.shape[0], 4))

    def values_fn(params, x, s):
        # Create a temporary state with given params
        from dataclasses import replace
        temp_state = replace(state, params=params)
        # Add batch dimension since _values expects batched input
        x_batched = jnp.expand_dims(x, 0)
        s_batched = jnp.expand_dims(s, 0)
        output = agent._values(temp_state, x_batched, s_batched)
        # Convert dict output to array if needed
        if isinstance(output, dict):
            output = output.get("values", output.get("v", list(output.values())[0]))
        # Remove batch dimension from output
        return output[0]

    # Compute Jacobians of output w.r.t. parameters
    jacobian_fn = jax.vmap(jax.jacrev(values_fn, argnums=0), in_axes=(None, 0, 0))
    jacobians = jacobian_fn(state.params, x_ref_sample, scalars)

    # Flatten jacobians: convert pytree to flat array
    def flatten_jacobians(jac_tree):
        leaves = jax.tree_util.tree_leaves(jac_tree)
        flat_leaves = [leaf.reshape(leaf.shape[0], -1) for leaf in leaves]
        return jnp.concatenate(flat_leaves, axis=1)

    jacobians_flat = flatten_jacobians(jacobians)

    # Compute NTK matrix: [n_samples, n_samples]
    ntk_matrix = jacobians_flat @ jacobians_flat.T

    # Compute rank and condition number
    rank = jnp.linalg.matrix_rank(ntk_matrix)
    cond_number = jnp.linalg.cond(ntk_matrix)

    return rank, cond_number


def calculate_ewm_reward(df: pl.DataFrame):
    """Calculate exponentially weighted moving average of rewards.

    Args:
        df: Polars DataFrame with 'rewards' column

    Returns:
        Polars DataFrame with 'ewm_reward' and 'mean_ewm_reward' columns added
    """
    if "rewards" not in df.columns:
        return df

    df = df.with_columns(
        pl.col("rewards").ewm_mean(alpha=1e-3, adjust=True).alias("ewm_reward"),
    )

    for exp in range(1, 20):
        df = df.with_columns(
            pl.col("rewards")
            .ewm_mean(alpha=10**-exp, adjust=True)
            .alias(f"ewm_reward_{exp}"),
        )

    df = df.with_columns(pl.col("ewm_reward").mean().alias("mean_ewm_reward"))

    return df


def calculate_mean_reward(df: pl.DataFrame):
    """Calculate mean of rewards.

    Args:
        df: Polars DataFrame with 'rewards' column

    Returns:
        Polars DataFrame with 'rolling_reward' and 'mean_reward' columns added
    """
    if "rewards" not in df.columns:
        return df

    window_sizes = [10, 100, 1000, 10000, 100000, 1000000]

    for window_size in window_sizes:
        df = df.with_columns(
            pl.col("rewards")
            .rolling_mean(window_size=window_size)
            .alias(f"rolling_reward_{window_size}"),
        )

    df = df.with_columns(pl.col("rewards").mean().alias("mean_reward"))
    return df


def calculate_object_traces(df: pl.DataFrame):
    """Calculate exponentially weighted moving traces for collected objects.

    Args:
        df: Polars DataFrame with 'object_collected_id' column

    Returns:
        Polars DataFrame with object trace columns added
    """
    if "object_collected_id" not in df.columns:
        return df

    max_obj = df.select(pl.col("object_collected_id").max()).item()

    for i in range(1, max_obj + 1):
        df = df.with_columns(
            (pl.col("object_collected_id") == i)
            .cast(pl.Float32)
            .ewm_mean(alpha=1e-2, adjust=True)
            .alias(f"object_trace_{i}_2"),
            (pl.col("object_collected_id") == i)
            .cast(pl.Float32)
            .ewm_mean(alpha=1e-1, adjust=True)
            .alias(f"object_trace_{i}_1"),
        )

    return df


def calculate_biome_occupancy(df: pl.DataFrame):
    """Calculate biome occupancy over time using windowed average.

    Args:
        df: Polars DataFrame with 'biome_id' column

    Returns:
        Polars DataFrame with biome occupancy columns added
    """
    if "biome_id" not in df.columns:
        return df

    window_sizes = [1, 10, 100, 1000, 10000, 100000, 500000, 1000000]

    # Get biome id range from data
    min_id = df.select(pl.col("biome_id").min()).item()
    max_id = df.select(pl.col("biome_id").max()).item()

    # Calculate occupancy for each biome using windowed average
    for i in range(min_id, max_id + 1):
        df = df.with_columns(
            (pl.col("biome_id").filter(pl.col("biome_id") == i).len() / pl.len()).alias(
                f"biome_percent_{i}"
            )
        )
        for window_size in window_sizes:
            df = df.with_columns(
                (pl.col("biome_id") == i)
                .cast(pl.Float32)
                .rolling_mean(window_size=window_size)
                .alias(f"biome_{i}_occupancy_{window_size}")
            )

    return df


def calculate_biome_regret(df: pl.DataFrame):
    """Calculate exponentially weighted moving average of biome regret.

    Args:
        df: Polars DataFrame with 'biome_regret' column

    Returns:
        Polars DataFrame with 'ewm_biome_regret' columns added
    """
    if "biome_regret" not in df.columns:
        return df

    df = df.with_columns(
        pl.col("biome_regret")
        .ewm_mean(alpha=1e-3, adjust=True)
        .alias("ewm_biome_regret"),
    )

    for exp in range(1, 10):
        df = df.with_columns(
            pl.col("biome_regret")
            .ewm_mean(alpha=10**-exp, adjust=True)
            .alias(f"ewm_biome_regret_{exp}"),
        )

    window_sizes = [10, 100, 1000, 10000, 100000, 1000000]
    for window_size in window_sizes:
        df = df.with_columns(
            pl.col("biome_regret")
            .rolling_mean(window_size=window_size)
            .alias(f"rolling_biome_regret_{window_size}"),
        )

    return df


def calculate_biome_rank(df: pl.DataFrame):
    """Calculate exponentially weighted moving average and rolling averages of biome rank.

    Args:
        df: Polars DataFrame with 'biome_rank' column

    Returns:
        Polars DataFrame with 'ewm_biome_rank' and rolling columns added
    """
    if "biome_rank" not in df.columns:
        return df

    df = df.with_columns(
        pl.col("biome_rank").ewm_mean(alpha=1e-3, adjust=True).alias("ewm_biome_rank"),
    )

    for exp in range(1, 10):
        df = df.with_columns(
            pl.col("biome_rank")
            .ewm_mean(alpha=10**-exp, adjust=True)
            .alias(f"ewm_biome_rank_{exp}"),
        )

    window_sizes = [10, 100, 1000, 10000, 100000, 1000000]
    for window_size in window_sizes:
        df = df.with_columns(
            pl.col("biome_rank")
            .rolling_mean(window_size=window_size)
            .alias(f"rolling_biome_rank_{window_size}"),
        )

    return df
