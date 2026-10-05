"""NTK and churn metrics for DQN agents."""

import logging
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from utils.metrics import compute_ntk_metrics

logger = logging.getLogger(__name__)


class DQNMetricsComputer:
    """Manages NTK and churn metric computation for DQN agents.

    Handles reference data collection and maintains churn state across steps.
    Metrics are computed at specified intervals and stored in the provided
    data arrays.
    """

    def __init__(self, glues, datas, ntk_freq, compute_value_head=True):
        """Initialize the metrics computer.

        Args:
            glues: List of RlGlue objects (one per agent/seed).
            datas: Dict of data arrays where metrics will be stored.
            ntk_freq: Frequency (in steps) at which to compute metrics.
                Only metrics are computed if ntk_freq > 0.
            compute_value_head: Whether to compute value head metrics (default True).
        """
        self.glues = glues
        self.datas = datas
        self.ntk_freq = ntk_freq
        self.compute_value_head = compute_value_head
        self.x_ref = None
        # Storage for tracking churn predictions: agent_idx -> predictions
        self._pred_cache = {}

    def collect_x_ref(self, glue_states, v_step, x_ref_steps: int = 500):
        """Collect reference observations from initial random agent steps.

        Args:
            glue_states: Initial glue states.
            v_step: Vmapped step function.
            x_ref_steps: Number of steps to collect observations from.
        """
        if self.ntk_freq == 0:
            return

        logger.info(f"Collecting {x_ref_steps} reference observations for NTK metrics...")
        x_ref_observations = []
        temp_glue_states = glue_states

        for _ in range(x_ref_steps):
            temp_glue_states, interaction = v_step(temp_glue_states)
            # Extract observations from first agent (index 0)
            # v_step is always vmapped, so we always need to extract [0]
            obs = (
                interaction.obs[0]
                if isinstance(interaction.obs, jnp.ndarray)
                else interaction.obs["image"][0]
            )
            x_ref_observations.append(np.asarray(obs))

        self.x_ref = jnp.stack(x_ref_observations)
        del x_ref_observations, temp_glue_states
        logger.info(f"Collected {self.x_ref.shape[0]} reference observations")

    def should_compute(self, step: int) -> bool:
        """Check if metrics should be computed at this step.

        Args:
            step: The current training step.

        Returns:
            True if this step is a metric boundary and metrics are enabled.
        """
        return (
            self.ntk_freq > 0
            and step % self.ntk_freq == 0
            and step > 0
            and self.x_ref is not None
        )

    def compute_and_store(self, agent_idx: int, step_idx: int, glue_state: Any):
        """Compute and store NTK and churn metrics for an agent at a step.

        Temporarily swaps the agent's state to compute metrics, then restores it.
        Churn is computed as the norm of output change from the previous metric
        step. Measurements on the first metric step (when no prior prediction
        exists) are not recorded for churn.

        Args:
            agent_idx: Index of the agent in the glues list.
            step_idx: Index where metrics will be stored in datas arrays.
            glue_state: The glue state at this step.
        """
        if self.ntk_freq == 0 or self.x_ref is None:
            return

        glue = self.glues[agent_idx]
        old_agent_state = glue.agent.state

        try:
            # Temporarily swap in the current state
            if hasattr(glue_state, 'agent_state'):
                glue.agent.state = glue_state.agent_state
            else:
                glue.agent.state = glue_state

            # Prepare reference batch (cap at 100 for memory)
            x_ref_batch, scalars_batch = self._prepare_ref_batch(agent_idx)

            # Compute current predictions for churn
            pred_current = glue.agent._values(glue.agent.state, x_ref_batch, scalars_batch)
            if isinstance(pred_current, dict):
                pred_current = pred_current.get(
                    "values", pred_current.get("v", list(pred_current.values())[0])
                )

            # Compute churn if we have previous predictions
            if agent_idx in self._pred_cache:
                pred_prev = self._pred_cache[agent_idx]
                churn_norm = np.linalg.norm(np.asarray(pred_current) - np.asarray(pred_prev))
                self.datas["churn_norm"][agent_idx, step_idx] = np.float16(churn_norm)

            # Store current predictions for next churn computation
            self._pred_cache[agent_idx] = np.asarray(pred_current)

            # Compute NTK metrics
            rank, cond = compute_ntk_metrics(glue.agent, x_ref_batch, scalars_batch)
            self.datas["ntk_rank"][agent_idx, step_idx] = np.float16(rank)
            self.datas["ntk_cond"][agent_idx, step_idx] = np.float32(cond)

        except Exception as e:
            logger.error(
                f"Failed to compute NTK metrics at step {step_idx} for agent {agent_idx}: {e}",
                exc_info=True,
            )
        finally:
            # Restore original state
            glue.agent.state = old_agent_state

    def _prepare_ref_batch(self, agent_idx: int) -> tuple:
        """Prepare reference batch with subsampling and scalars.

        Args:
            agent_idx: Index of the agent.

        Returns:
            Tuple of (x_ref_batch, scalars_batch).
        """
        n_samples = min(100, len(self.x_ref))

        if n_samples < len(self.x_ref):
            sample_indices = np.linspace(0, len(self.x_ref) - 1, n_samples, dtype=int)
            x_ref_batch = self.x_ref[sample_indices]
        else:
            x_ref_batch = self.x_ref

        # Create dummy scalars (zeros) for reference data
        glue = self.glues[agent_idx]
        if (
            hasattr(glue.agent.state, 'last_timestep')
            and 'scalars' in glue.agent.state.last_timestep
        ):
            scalars_size = glue.agent.state.last_timestep['scalars'].shape[-1]
        else:
            scalars_size = 4  # fallback default

        scalars_batch = jnp.zeros((n_samples, scalars_size))
        return x_ref_batch, scalars_batch
