from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qd_lab.jax_rollout import BatchedRolloutMetrics
from sim_core.genome import TraitGenome
from sim_core.genome_mapper import ACTION_NAMES

# Indices into ACTION_NAMES
_IDX_IDLE = 0
_IDX_RADIATOR = 1
_IDX_LAPTOP = 2
_IDX_BOX = 4
_IDX_STARE = 5
_IDX_RUB = 6
_IDX_CHAOS = 7

MAX_ENTROPY_BITS = float(np.log2(len(ACTION_NAMES)))  # 3.0 bits


@dataclass(frozen=True)
class BehaviorBatch:
    """Computed QD objectives and 3D behavioral descriptors for a batch of genomes."""

    objectives: np.ndarray  # (N,) in [0, 1]
    human_engagement: np.ndarray  # (N,) in [0, 1]
    activity_entropy: np.ndarray  # (N,) normalized Shannon entropy in [0, 1]
    disruption_rate: np.ndarray  # (N,) normalized disruption in [0, 1]

    def measures_2d(self) -> np.ndarray:
        """Primary 2D MAP-Elites coordinates: [human_engagement, activity_entropy]."""
        return np.column_stack([self.human_engagement, self.activity_entropy]).astype(np.float32)

    def measures_3d(self) -> np.ndarray:
        """Full 3D behavioral coordinates: [human_engagement, activity_entropy, disruption_rate]."""
        return np.column_stack(
            [self.human_engagement, self.activity_entropy, self.disruption_rate]
        ).astype(np.float32)


def compute_behavior_batch(metrics: BatchedRolloutMetrics, ticks: int) -> BehaviorBatch:
    """Convert raw batched rollout telemetry into QD descriptors and viability objectives."""
    ratios = np.clip(metrics.action_ratios, 0.0, 1.0)

    # 1. Normalized Shannon action entropy in [0, 1]
    safe_p = np.where(ratios > 1e-9, ratios, 1.0)
    entropy_bits = -np.sum(np.where(ratios > 1e-9, ratios * np.log2(safe_p), 0.0), axis=1)
    activity_entropy = np.clip(entropy_bits / MAX_ENTROPY_BITS, 0.0, 1.0)

    # 2. Human engagement: blend of social/observation policies + physical proximity
    social_action_ratio = ratios[:, _IDX_STARE] + ratios[:, _IDX_RUB] + ratios[:, _IDX_LAPTOP]
    human_engagement = np.clip(
        0.65 * social_action_ratio + 0.35 * metrics.user_proximity_ratio,
        0.0,
        1.0,
    )

    # 3. Environmental disruption rate normalized against ~0.25 disruption/tick ceiling
    disruption_rate = np.clip(metrics.total_disruption / max(1.0, 0.25 * ticks), 0.0, 1.0)

    # Objective: biological viability + generative model coherence + active agency
    homeo_viability = (
        0.30 * metrics.mean_warmth
        + 0.25 * (1.0 - metrics.mean_hunger)
        + 0.25 * metrics.mean_energy
        + 0.20 * metrics.mean_comfort
    )
    vfe_coherence = np.exp(-0.12 * np.maximum(0.0, metrics.mean_vfe - 2.0))
    agency_score = 1.0 - ratios[:, _IDX_IDLE]

    objectives = np.clip(
        0.55 * homeo_viability + 0.30 * vfe_coherence + 0.15 * agency_score,
        0.0,
        1.0,
    )

    return BehaviorBatch(
        objectives=objectives.astype(np.float32),
        human_engagement=human_engagement.astype(np.float32),
        activity_entropy=activity_entropy.astype(np.float32),
        disruption_rate=disruption_rate.astype(np.float32),
    )


def infer_archetype_label(
    genome: TraitGenome,
    action_ratios: np.ndarray,
    disruption_rate: float,
    activity_entropy: float,
) -> str:
    """Assign a human-readable post-hoc label for archive browsing (never used in control)."""
    if genome.max_anomaly >= 0.80:
        return "anomalous_entity"
    if action_ratios[_IDX_CHAOS] >= 0.45 or disruption_rate >= 0.55:
        return "chaotic_gremlin"
    if action_ratios[_IDX_STARE] >= 0.55:
        return "staring_observer"
    if action_ratios[_IDX_RUB] >= 0.45:
        return "velcro_companion"
    if (action_ratios[_IDX_RADIATOR] + action_ratios[_IDX_LAPTOP]) >= 0.55 and genome.fur_density < 0.35:
        return "hairless_heat_seeker"
    if action_ratios[_IDX_BOX] >= 0.45:
        return "shy_hermit"
    if (action_ratios[_IDX_RADIATOR] + action_ratios[_IDX_LAPTOP]) >= 0.55:
        return "warmth_camper"
    if activity_entropy >= 0.50:
        return "restless_explorer"
    return "balanced_hybrid"