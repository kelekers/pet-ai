from __future__ import annotations

import math

import jax.numpy as jnp
from pymdp.agent import Agent

from sim_core.genome import TraitGenome
from sim_core.genome_mapper import (
    A_DEPENDENCIES,
    ACTION_NAMES,
    B_DEPENDENCIES,
    build_A_matrices,
    build_B_matrices,
)
from sim_core.rollout import run_aif_rollout


def test_aif_rollout_is_deterministic_and_finite() -> None:
    genome = TraitGenome.preset_staring_observer()
    run_a = run_aif_rollout(genome=genome, ticks=60, seed=7)
    run_b = run_aif_rollout(genome=genome, ticks=60, seed=7)

    assert run_a.action_counts == run_b.action_counts
    assert math.isfinite(run_a.mean_vfe)
    assert abs(sum(run_a.last_step.policy_probs) - 1.0) < 1e-4


def test_emergent_personality_divergence_without_hardcoding() -> None:
    """Identical rooms must yield distinct dominant behaviors purely from genome priors."""
    sphynx = run_aif_rollout(TraitGenome.preset_hairless_sphynx(), ticks=200, seed=42)
    observer = run_aif_rollout(TraitGenome.preset_staring_observer(), ticks=200, seed=42)
    gremlin = run_aif_rollout(TraitGenome.preset_chaotic_gremlin(), ticks=200, seed=42)

    # Hairless cat seeks heat significantly more than the fluffy observer
    assert sphynx.heat_seeking_ratio > observer.heat_seeking_ratio + 0.15

    # Staring observer watches user significantly more than the gremlin
    assert observer.action_ratios["stare_user"] > gremlin.action_ratios["stare_user"] + 0.25

    # Chaotic gremlin causes significantly more disruption and breaks the water glass
    assert gremlin.total_disruption > observer.total_disruption + 5.0
    assert not gremlin.water_glass_intact


def test_pure_epistemic_information_gain_drives_observation() -> None:
    """With flat C preferences, EFE info-gain alone must prefer stare_user over idle."""
    genome = TraitGenome(curiosity=0.95, sentience_glimmer=0.80)
    a_mats = build_A_matrices(genome)
    b_mats = build_B_matrices()
    flat_c = [jnp.ones((a.shape[0],), dtype=jnp.float32) for a in a_mats]
    d_prior = [
        jnp.full((len(ACTION_NAMES),), 1.0 / len(ACTION_NAMES), dtype=jnp.float32),
        jnp.array([0.5, 0.5], dtype=jnp.float32),
    ]

    agent = Agent(
        A=a_mats,
        B=b_mats,
        C=flat_c,
        D=d_prior,
        A_dependencies=A_DEPENDENCIES,
        B_dependencies=B_DEPENDENCIES,
        policy_len=1,
        use_utility=True,
        use_states_info_gain=True,
        batch_size=1,
    )
    qs = [d[None, None, :] for d in d_prior]
    _, neg_efe = agent.infer_policies(qs)
    scores = jnp.asarray(neg_efe).reshape(-1)

    stare_idx = ACTION_NAMES.index("stare_user")
    idle_idx = ACTION_NAMES.index("idle")
    assert float(scores[stare_idx]) > float(scores[idle_idx]) + 0.15