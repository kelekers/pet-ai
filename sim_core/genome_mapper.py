from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from sim_core.genome import TraitGenome
from sim_core.homeostasis import HomeostaticState

# 8 behavioral policies mapped directly to Sprint 1 Smart Objects
ACTION_NAMES: tuple[str, ...] = (
    "idle",
    "radiator",
    "laptop",
    "food_bowl",
    "cardboard_box",
    "stare_user",
    "rub_user",
    "chaos_play",
)
NUM_ACTIONS = len(ACTION_NAMES)

# Sparse modality -> hidden-state factor dependencies for pymdp
# Factor 0: activity_state (8 states, 8 controls)
# Factor 1: user_intent (2 states: 0=calm, 1=busy; 1 control)
A_DEPENDENCIES: list[list[int]] = [[0], [0], [0], [0], [0, 1], [0, 1]]
B_DEPENDENCIES: list[list[int]] = [[0], [1]]


@dataclass(frozen=True)
class GenerativeModelSpec:
    A: list[jnp.ndarray]
    B: list[jnp.ndarray]
    C: list[jnp.ndarray]
    D: list[jnp.ndarray]
    gamma: float
    alpha: float


def _shift_preferences(logits: list[float]) -> jnp.ndarray:
    """Shift log-preferences non-negative (invariant under log_softmax, safe under log)."""
    arr = np.asarray(logits, dtype=np.float32)
    shifted = arr - float(np.min(arr)) + 0.05
    return jnp.asarray(shifted, dtype=jnp.float32)


def build_A_matrices(genome: TraitGenome) -> list[jnp.ndarray]:
    """Build likelihood tensors P(o_m | s) with distinct object trade-offs."""
    # Modality 0: thermal (0=cold, 1=neutral, 2=warm) | dep: [0]
    a_thermal = np.array(
        [
            # idle  rad   lap   food  box   stare rub   chaos
            [0.55, 0.02, 0.10, 0.55, 0.08, 0.55, 0.15, 0.55],
            [0.40, 0.04, 0.25, 0.40, 0.44, 0.40, 0.45, 0.40],
            [0.05, 0.94, 0.65, 0.05, 0.48, 0.05, 0.40, 0.05],
        ],
        dtype=np.float32,
    )

    # Modality 1: satiety (0=hungry, 1=fed) | dep: [0]
    a_satiety = np.array(
        [
            [0.85, 0.85, 0.85, 0.04, 0.85, 0.85, 0.85, 0.85],
            [0.15, 0.15, 0.15, 0.96, 0.15, 0.15, 0.15, 0.15],
        ],
        dtype=np.float32,
    )

    # Modality 2: effort (0=restful, 1=moderate, 2=high_exertion) | dep: [0]
    a_effort = np.array(
        [
            [0.92, 0.85, 0.28, 0.25, 0.90, 0.90, 0.15, 0.03],
            [0.06, 0.12, 0.58, 0.65, 0.08, 0.08, 0.65, 0.15],
            [0.02, 0.03, 0.14, 0.10, 0.02, 0.02, 0.20, 0.82],
        ],
        dtype=np.float32,
    )

    # Modality 3: stimulation (0=dull, 1=mild_novelty, 2=chaotic_impact) | dep: [0]
    a_stim = np.array(
        [
            [0.90, 0.78, 0.22, 0.55, 0.72, 0.20, 0.20, 0.02],
            [0.08, 0.19, 0.60, 0.40, 0.23, 0.75, 0.72, 0.06],
            [0.02, 0.03, 0.18, 0.05, 0.05, 0.05, 0.08, 0.92],
        ],
        dtype=np.float32,
    )

    # Modality 4: social_contact (0=solitary, 1=observing, 2=intimate) | dep: [0, 1]
    a_social = np.zeros((3, NUM_ACTIONS, 2), dtype=np.float32)
    for idx in (0, 1, 3, 7):
        a_social[:, idx, :] = np.array([[0.82], [0.13], [0.05]], dtype=np.float32)
    a_social[:, 4, :] = np.array([[0.92], [0.06], [0.02]], dtype=np.float32)  # cardboard_box
    a_social[:, 5, :] = np.array([[0.05], [0.90], [0.05]], dtype=np.float32)  # stare_user
    # Laptop is mild observation; rub_user is true social intimacy
    a_social[:, 2, 0] = [0.25, 0.55, 0.20]
    a_social[:, 2, 1] = [0.35, 0.50, 0.15]
    a_social[:, 6, 0] = [0.04, 0.08, 0.88]
    a_social[:, 6, 1] = [0.12, 0.30, 0.58]

    # Modality 5: user_cue (0=no_signal, 1=cue_calm, 2=cue_busy) | dep: [0, 1]
    acuity = 0.56 + 0.38 * (0.7 * genome.curiosity + 0.3 * genome.sentience_glimmer)
    miss = 0.96 - acuity
    a_cue = np.zeros((3, NUM_ACTIONS, 2), dtype=np.float32)
    for idx in range(NUM_ACTIONS):
        a_cue[:, idx, :] = np.array([[0.90], [0.05], [0.05]], dtype=np.float32)

    a_cue[:, 5, 0] = [0.04, acuity, miss]
    a_cue[:, 5, 1] = [0.04, miss, acuity]
    # Weak incidental cues on laptop/rub so stare_user uniquely dominates epistemic value
    a_cue[:, 2, 0] = [0.55, 0.30, 0.15]
    a_cue[:, 2, 1] = [0.55, 0.15, 0.30]
    a_cue[:, 6, 0] = [0.40, 0.42, 0.18]
    a_cue[:, 6, 1] = [0.40, 0.18, 0.42]

    return [
        jnp.asarray(a_thermal),
        jnp.asarray(a_satiety),
        jnp.asarray(a_effort),
        jnp.asarray(a_stim),
        jnp.asarray(a_social),
        jnp.asarray(a_cue),
    ]

def build_B_matrices() -> list[jnp.ndarray]:
    """Build transition tensors P(s' | s, u) for activity_state and drifting user_intent."""
    # Factor 0: controllable transition into chosen activity state (94% reliable)
    b_activity = np.full((NUM_ACTIONS, NUM_ACTIONS, NUM_ACTIONS), 0.06 / (NUM_ACTIONS - 1), dtype=np.float32)
    for action_idx in range(NUM_ACTIONS):
        b_activity[action_idx, :, action_idx] = 0.94

    # Factor 1: uncontrollable user_intent drifts slowly, regenerating epistemic uncertainty
    b_user = np.array([[[0.82], [0.18]], [[0.18], [0.82]]], dtype=np.float32)

    return [jnp.asarray(b_activity), jnp.asarray(b_user)]


def build_C_preferences(
    genome: TraitGenome,
    homeostasis: HomeostaticState,
) -> list[jnp.ndarray]:
    """Map genetic traits and live homeostatic urgency onto observation prior preferences C."""
    cold_urgency = max(0.0, 0.78 - homeostasis.warmth)
    hunger_urgency = max(0.0, homeostasis.hunger - 0.25)
    fatigue_urgency = max(0.0, 0.55 - homeostasis.energy)
    boredom_urgency = max(0.0, 0.65 - homeostasis.stimulation)

    # 0. Thermal: [cold, neutral, warm]
    c_thermal = _shift_preferences(
        [
            -1.2 * (1.0 - genome.fur_density) - 3.0 * cold_urgency,
            0.0,
            0.6 + 2.8 * (1.0 - genome.fur_density) + 6.5 * cold_urgency,
        ]
    )

    # 1. Satiety: [hungry, fed]
    c_satiety = _shift_preferences(
        [
            -2.5 * hunger_urgency,
            0.2 + 10.5 * hunger_urgency,
        ]
    )

    # 2. Effort: [restful, moderate, high_exertion]
    c_effort = _shift_preferences(
        [
            3.0 * genome.laziness + 5.0 * fatigue_urgency,
            0.2,
            1.8 * (1.0 - genome.laziness)
            + 0.8 * genome.agility
            - 2.4 * genome.laziness
            - 3.5 * fatigue_urgency,
        ]
    )

    # 3. Stimulation / Disruption: [dull, mild_novelty, chaotic_impact]
    c_stim = _shift_preferences(
        [
            -1.5 * genome.curiosity - 2.5 * boredom_urgency,
            2.6 * genome.curiosity + 2.5 * boredom_urgency,
            3.8 * genome.chaos
            + 1.8 * genome.impulsiveness
            + 1.0 * genome.dominance
            - 1.4 * genome.agreeableness
            - 2.2 * genome.laziness,
        ]
    )

    # 4. Social contact: [solitary, observing, intimate]
    c_social = _shift_preferences(
        [
            2.0 * (1.0 - genome.extraversion)
            + 1.4 * genome.neuroticism
            - 1.5 * genome.attachment,
            2.8 * genome.curiosity
            + 1.6 * genome.sentience_glimmer
            - 0.5 * genome.extraversion,
            3.2 * genome.attachment
            + 1.6 * genome.agreeableness
            + 1.2 * genome.extraversion
            - 1.8 * genome.neuroticism,
        ]
    )

    # 5. User cue: strictly flat (pure epistemic value, zero pragmatic bias)
    c_cue = jnp.ones((3,), dtype=jnp.float32)

    return [c_thermal, c_satiety, c_effort, c_stim, c_social, c_cue]


def build_generative_model(
    genome: TraitGenome,
    homeostasis: HomeostaticState | None = None,
) -> GenerativeModelSpec:
    """Compile the full generative model from a TraitGenome."""
    h = homeostasis or HomeostaticState()
    d_activity = np.full((NUM_ACTIONS,), 0.05, dtype=np.float32)
    d_activity[0] = 1.0 - 0.05 * (NUM_ACTIONS - 1)
    d_user = np.array([0.5, 0.5], dtype=np.float32)

    # Impulsiveness lowers policy precision (gamma), creating erratic exploration
    gamma = float(2.2 + 7.8 * (1.0 - genome.impulsiveness))
    alpha = float(2.5 + 9.5 * (1.0 - genome.impulsiveness))

    return GenerativeModelSpec(
        A=build_A_matrices(genome),
        B=build_B_matrices(),
        C=build_C_preferences(genome, h),
        D=[jnp.asarray(d_activity), jnp.asarray(d_user)],
        gamma=gamma,
        alpha=alpha,
    )