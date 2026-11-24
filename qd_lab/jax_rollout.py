from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from pymdp.agent import Agent

from sim_core.genome import TraitGenome
from sim_core.genome_mapper import (
    A_DEPENDENCIES,
    B_DEPENDENCIES,
    build_generative_model,
)

THERMONEUTRAL_TEMP_C = 32.0

# Static base likelihood matrices matching sim_core/genome_mapper.py
_TEMPLATE_SPEC = build_generative_model(TraitGenome())
_BASE_A_STATIC: tuple[jnp.ndarray, ...] = tuple(_TEMPLATE_SPEC.A[:5])
_BASE_A_CUE: jnp.ndarray = _TEMPLATE_SPEC.A[5]

_TEMPLATE_AGENT = Agent(
    A=_TEMPLATE_SPEC.A,
    B=_TEMPLATE_SPEC.B,
    C=_TEMPLATE_SPEC.C,
    D=_TEMPLATE_SPEC.D,
    A_dependencies=A_DEPENDENCIES,
    B_dependencies=B_DEPENDENCIES,
    policy_len=1,
    gamma=_TEMPLATE_SPEC.gamma,
    alpha=_TEMPLATE_SPEC.alpha,
    use_utility=True,
    use_states_info_gain=True,
    batch_size=1,
)

# Smart Object geometry & affordance tables indexed by policy 0..7:
# [0:idle, 1:radiator, 2:laptop, 3:food_bowl, 4:cardboard_box, 5:stare_user, 6:rub_user, 7:chaos]
_OBJ_X = jnp.array([3.0, 0.4, 3.0, 1.0, 2.2, 3.0, 3.0, 4.5], dtype=jnp.float32)
_OBJ_Y = jnp.array([3.0, 3.0, 4.8, 0.8, 1.5, 4.3, 4.3, 3.2], dtype=jnp.float32)
_REQ_PROX = jnp.array([False, True, True, True, True, False, True, True])
_ENERGY_COST = jnp.array([0.0, 0.005, 0.01, 0.01, 0.002, 0.001, 0.015, 0.045], dtype=jnp.float32)
_DISRUPTION = jnp.array([0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.15], dtype=jnp.float32)

# Drive deltas per policy: columns = [warmth, hunger, energy, stimulation, comfort]
_DRIVE_DELTAS = jnp.array(
    [
        [0.00, 0.00, 0.00, 0.00, 0.00],  # 0: idle
        [0.14, 0.00, 0.04, 0.00, 0.08],  # 1: radiator
        [0.11, 0.00, 0.00, 0.05, 0.06],  # 2: laptop
        [0.00, -0.35, 0.05, 0.00, 0.04],  # 3: food_bowl
        [0.04, 0.00, 0.12, 0.00, 0.14],  # 4: cardboard_box
        [0.00, 0.00, 0.00, 0.07, 0.00],  # 5: stare_user
        [0.05, 0.00, 0.00, 0.06, 0.12],  # 6: rub_user
        [0.00, 0.00, 0.00, 0.18, 0.05],  # 7: scratching_post fallback
    ],
    dtype=jnp.float32,
)

# Water glass override for policy 7 when intact and agility >= 0.25
_GLASS_DELTAS = jnp.array([0.0, 0.0, 0.0, 0.22, 0.0], dtype=jnp.float32)

# Fixed heat emitters: [radiator, laptop, cardboard_box, user] (sunbeam handled dynamically)
_HEAT_X = jnp.array([0.4, 3.0, 2.2, 3.0], dtype=jnp.float32)
_HEAT_Y = jnp.array([3.0, 4.8, 1.5, 4.3], dtype=jnp.float32)
_HEAT_TEMP = jnp.array([42.0, 38.5, 24.5, 34.0], dtype=jnp.float32)
_HEAT_RADIUS = jnp.array([1.6, 0.8, 0.5, 0.7], dtype=jnp.float32)


@dataclass(frozen=True)
class BatchedRolloutMetrics:
    """NumPy arrays of rollout telemetry for a batch of N genomes."""

    action_ratios: np.ndarray  # (N, 8)
    user_proximity_ratio: np.ndarray  # (N,)
    total_disruption: np.ndarray  # (N,)
    mean_warmth: np.ndarray  # (N,)
    mean_hunger: np.ndarray  # (N,)
    mean_energy: np.ndarray  # (N,)
    mean_comfort: np.ndarray  # (N,)
    mean_vfe: np.ndarray  # (N,)
    water_glass_intact: np.ndarray  # (N,) bool


def _shift_pref_jax(vals: jnp.ndarray) -> jnp.ndarray:
    return vals - jnp.min(vals) + 0.05


def _build_A_jax(genome: jnp.ndarray) -> list[jnp.ndarray]:
    curiosity = genome[5]
    sentience = genome[17]
    acuity = 0.56 + 0.38 * (0.7 * curiosity + 0.3 * sentience)
    miss = 0.96 - acuity

    a_cue = _BASE_A_CUE.at[:, 5, 0].set(jnp.array([0.04, acuity, miss], dtype=jnp.float32))
    a_cue = a_cue.at[:, 5, 1].set(jnp.array([0.04, miss, acuity], dtype=jnp.float32))
    return [jnp.expand_dims(a, axis=0) for a in (*_BASE_A_STATIC, a_cue)]


def _build_C_jax(
    genome: jnp.ndarray,
    warmth: jnp.ndarray,
    hunger: jnp.ndarray,
    energy: jnp.ndarray,
    stimulation: jnp.ndarray,
) -> list[jnp.ndarray]:
    neuroticism = genome[0]
    extraversion = genome[1]
    dominance = genome[2]
    impulsiveness = genome[3]
    agreeableness = genome[4]
    curiosity = genome[5]
    laziness = genome[6]
    attachment = genome[7]
    chaos = genome[8]
    fur_density = genome[9]
    agility = genome[11]
    sentience = genome[17]

    cold_u = jnp.maximum(0.0, 0.78 - warmth)
    hunger_u = jnp.maximum(0.0, hunger - 0.25)
    fatigue_u = jnp.maximum(0.0, 0.55 - energy)
    boredom_u = jnp.maximum(0.0, 0.65 - stimulation)

    c_thermal = _shift_pref_jax(
        jnp.array(
            [
                -1.2 * (1.0 - fur_density) - 3.0 * cold_u,
                0.0,
                0.6 + 2.8 * (1.0 - fur_density) + 6.5 * cold_u,
            ],
            dtype=jnp.float32,
        )
    )
    c_satiety = _shift_pref_jax(
        jnp.array([-2.5 * hunger_u, 0.2 + 10.5 * hunger_u], dtype=jnp.float32)
    )
    c_effort = _shift_pref_jax(
        jnp.array(
            [
                3.0 * laziness + 5.0 * fatigue_u,
                0.2,
                1.8 * (1.0 - laziness) + 0.8 * agility - 2.4 * laziness - 3.5 * fatigue_u,
            ],
            dtype=jnp.float32,
        )
    )
    c_stim = _shift_pref_jax(
        jnp.array(
            [
                -1.5 * curiosity - 2.5 * boredom_u,
                2.6 * curiosity + 2.5 * boredom_u,
                3.8 * chaos
                + 1.8 * impulsiveness
                + 1.0 * dominance
                - 1.4 * agreeableness
                - 2.2 * laziness,
            ],
            dtype=jnp.float32,
        )
    )
    c_social = _shift_pref_jax(
        jnp.array(
            [
                2.0 * (1.0 - extraversion) + 1.4 * neuroticism - 1.5 * attachment,
                2.8 * curiosity + 1.6 * sentience - 0.5 * extraversion,
                3.2 * attachment + 1.6 * agreeableness + 1.2 * extraversion - 1.8 * neuroticism,
            ],
            dtype=jnp.float32,
        )
    )
    c_cue = jnp.ones((3,), dtype=jnp.float32)
    return [jnp.expand_dims(c, axis=0) for c in (c_thermal, c_satiety, c_effort, c_stim, c_social, c_cue)]


def _local_temp_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    tick: jnp.ndarray,
    ambient_temp_c: float,
) -> jnp.ndarray:
    dists = jnp.hypot(_HEAT_X - x, _HEAT_Y - y)
    ratios = jnp.maximum(0.0, 1.0 - dists / _HEAT_RADIUS)
    static_heat = jnp.sum(jnp.maximum(0.0, _HEAT_TEMP - ambient_temp_c) * (ratios**2))

    sun_temp = 29.0 + 4.0 * jnp.sin(tick.astype(jnp.float32) * 0.015)
    sun_dist = jnp.hypot(4.8 - x, 1.2 - y)
    sun_ratio = jnp.maximum(0.0, 1.0 - sun_dist / 1.2)
    sun_heat = jnp.maximum(0.0, sun_temp - ambient_temp_c) * (sun_ratio**2)
    return ambient_temp_c + static_heat + sun_heat


def _rollout_single_jax(
    genome: jnp.ndarray,
    rng_key: jnp.ndarray,
    ticks: int,
    ambient_temp_c: float,
) -> tuple[jnp.ndarray, ...]:
    """Run one T-tick Active Inference rollout completely inside XLA."""
    impulsiveness = genome[3]
    fur_density = genome[9]
    body_mass = genome[10]
    agility = genome[11]
    metabolism = genome[12]
    curiosity = genome[5]
    neuroticism = genome[0]

    conductance = 1.0 - 0.62 * fur_density
    max_step_speed = 0.45 + 0.55 * agility

    a_mats = _build_A_jax(genome)
    gamma_val = jnp.full(
        jnp.shape(_TEMPLATE_AGENT.gamma), 2.2 + 7.8 * (1.0 - impulsiveness), dtype=jnp.float32
    )
    alpha_val = jnp.full(
        jnp.shape(_TEMPLATE_AGENT.alpha), 2.5 + 9.5 * (1.0 - impulsiveness), dtype=jnp.float32
    )
    pet_agent = eqx.tree_at(
        lambda a: (a.A, a.gamma, a.alpha),
        _TEMPLATE_AGENT,
        (a_mats, gamma_val, alpha_val),
    )

    def step_fn(carry: tuple, tick: jnp.ndarray) -> tuple[tuple, tuple]:
        (
            x,
            y,
            warmth,
            hunger,
            energy,
            stimulation,
            comfort,
            glass_intact,
            last_action,
            last_approaching,
            last_disruption,
            empirical_prior,
            key,
        ) = carry

        # 1. Encode observations from live physical state
        local_temp = _local_temp_jax(x, y, tick, ambient_temp_c)
        o_thermal = jnp.where(
            (local_temp >= 28.0) | (warmth >= 0.86),
            2,
            jnp.where((local_temp <= 22.0) & (warmth < 0.72), 0, 1),
        )
        o_satiety = jnp.where(hunger >= 0.40, 0, 1)
        o_effort = jnp.where(
            (~last_approaching) & (last_action == 7),
            2,
            jnp.where(last_approaching | (last_action == 6) | (last_action == 2), 1, 0),
        )
        o_stim = jnp.where(
            last_disruption >= 0.5,
            2,
            jnp.where(
                (stimulation >= 0.60)
                | ((~last_approaching) & ((last_action == 5) | (last_action == 6))),
                1,
                0,
            ),
        )
        dist_user = jnp.hypot(3.0 - x, 4.3 - y)
        o_social = jnp.where(
            ((~last_approaching) & (last_action == 6)) | (dist_user <= 0.6),
            2,
            jnp.where((~last_approaching) & ((last_action == 5) | (last_action == 2)), 1, 0),
        )
        observing_user = (~last_approaching) & ((last_action == 5) | (last_action == 6))
        o_cue = jnp.where(observing_user, jnp.where(((tick // 35) % 2) == 0, 1, 2), 0)

        obs = [
            jnp.array([o_thermal], dtype=jnp.int32),
            jnp.array([o_satiety], dtype=jnp.int32),
            jnp.array([o_effort], dtype=jnp.int32),
            jnp.array([o_stim], dtype=jnp.int32),
            jnp.array([o_social], dtype=jnp.int32),
            jnp.array([o_cue], dtype=jnp.int32),
        ]

        # 2. Allostatic Active Inference step
        new_c = _build_C_jax(genome, warmth, hunger, energy, stimulation)
        agent_t = eqx.tree_at(lambda a: a.C, pet_agent, new_c)
        qs, info = agent_t.infer_states(obs, empirical_prior=empirical_prior, return_info=True)
        q_pi, _ = agent_t.infer_policies(qs)

        key, subkey, noise_key = jr.split(key, 3)
        sample_keys = jr.split(subkey, 2)
        action = agent_t.sample_action(q_pi, rng_key=sample_keys[1:])
        next_prior = agent_t.update_empirical_prior(action, qs)

        action_idx = jnp.asarray(action, dtype=jnp.int32).reshape(-1)[0]
        vfe = jnp.asarray(info["vfe"], dtype=jnp.float32).reshape(-1)[0]

        # 3. Execute Smart Object movement or affordance
        use_glass = (action_idx == 7) & glass_intact & (agility >= 0.25)
        tx = jnp.where(use_glass, 3.4, _OBJ_X[action_idx])
        ty = jnp.where(use_glass, 4.9, _OBJ_Y[action_idx])
        req_prox = _REQ_PROX[action_idx]
        can_execute = (action_idx != 0) & (~((action_idx == 2) & (agility < 0.25)))

        dx = tx - x
        dy = ty - y
        dist = jnp.hypot(dx, dy)
        should_move = can_execute & req_prox & (dist > 0.35)
        should_interact = can_execute & (~should_move)

        step_dist = jnp.minimum(dist, max_step_speed)
        safe_dist = jnp.maximum(dist, 1e-6)
        nx = jnp.where(should_move, jnp.clip(x + (dx / safe_dist) * step_dist, 0.0, 6.0), x)
        ny = jnp.where(should_move, jnp.clip(y + (dy / safe_dist) * step_dist, 0.0, 6.0), y)

        e_cost = jnp.where(use_glass, 0.03, _ENERGY_COST[action_idx])
        deltas = jnp.where(use_glass, _GLASS_DELTAS, _DRIVE_DELTAS[action_idx])
        disruption_delta = jnp.where(
            should_interact, jnp.where(use_glass, 1.0, _DISRUPTION[action_idx]), 0.0
        )
        next_glass_intact = glass_intact & (~(should_interact & use_glass))

        warmth = jnp.where(should_interact, jnp.clip(warmth + deltas[0], 0.0, 1.0), warmth)
        hunger = jnp.where(should_interact, jnp.clip(hunger + deltas[1], 0.0, 1.0), hunger)
        energy = jnp.where(
            should_interact, jnp.clip(energy - e_cost + deltas[2], 0.0, 1.0), energy
        )
        stimulation = jnp.where(
            should_interact, jnp.clip(stimulation + deltas[3], 0.0, 1.0), stimulation
        )
        comfort = jnp.where(should_interact, jnp.clip(comfort + deltas[4], 0.0, 1.0), comfort)

        activity_level = jnp.where(
            should_move,
            jnp.minimum(1.0, 0.25 + step_dist),
            jnp.where(should_interact, jnp.minimum(1.0, 0.1 + e_cost * 12.0), 0.05),
        )

        # 4. Homeostasis tick update
        next_tick = tick + 1
        post_temp = _local_temp_jax(nx, ny, next_tick, ambient_temp_c)
        ambient_noise = jnp.clip(0.1 + 0.02 * jr.normal(noise_key), 0.0, 1.0)

        cold_loss = jnp.maximum(
            0.0, 0.00045 * (THERMONEUTRAL_TEMP_C - post_temp) * conductance - 0.00025 * metabolism
        )
        warm_gain = 0.0022 * (post_temp - THERMONEUTRAL_TEMP_C) * (0.5 + 0.5 * conductance)
        delta_warmth = jnp.where(post_temp < THERMONEUTRAL_TEMP_C, -cold_loss, warm_gain)
        warmth = jnp.clip(warmth + delta_warmth, 0.0, 1.0)

        hunger = jnp.clip(
            hunger + (0.0007 + 0.0011 * metabolism + 0.0008 * activity_level), 0.0, 1.0
        )
        delta_energy = jnp.where(
            activity_level <= 0.1, 0.0015, -(0.0006 + 0.0010 * body_mass) * activity_level
        )
        energy = jnp.clip(energy + delta_energy, 0.0, 1.0)
        stimulation = jnp.clip(stimulation - (0.0006 + 0.0014 * curiosity), 0.0, 1.0)

        stress = (
            0.35 * (1.0 - warmth)
            + 0.30 * hunger
            + 0.35 * ambient_noise * (0.5 + neuroticism)
        )
        target_comfort = jnp.clip(1.0 - stress, 0.0, 1.0)
        comfort = jnp.clip(comfort + 0.08 * (target_comfort - comfort), 0.0, 1.0)

        near_user = (jnp.hypot(3.0 - nx, 4.3 - ny) <= 1.25).astype(jnp.float32)

        next_carry = (
            nx,
            ny,
            warmth,
            hunger,
            energy,
            stimulation,
            comfort,
            next_glass_intact,
            action_idx,
            should_move,
            disruption_delta,
            next_prior,
            key,
        )
        outputs = (action_idx, near_user, disruption_delta, warmth, hunger, energy, comfort, vfe)
        return next_carry, outputs

    init_carry = (
        jnp.array(3.0, dtype=jnp.float32),
        jnp.array(3.0, dtype=jnp.float32),
        jnp.array(0.85, dtype=jnp.float32),
        jnp.array(0.20, dtype=jnp.float32),
        jnp.array(0.85, dtype=jnp.float32),
        jnp.array(0.65, dtype=jnp.float32),
        jnp.array(0.75, dtype=jnp.float32),
        jnp.array(True),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(False),
        jnp.array(0.0, dtype=jnp.float32),
        pet_agent.D,
        rng_key,
    )

    final_carry, trace = jax.lax.scan(step_fn, init_carry, jnp.arange(ticks, dtype=jnp.int32))
    actions, near_user, disruption, warmth_t, hunger_t, energy_t, comfort_t, vfe_t = trace
    final_glass_intact = final_carry[7]

    one_hot = jax.nn.one_hot(actions, num_classes=8, dtype=jnp.float32)
    action_ratios = jnp.mean(one_hot, axis=0)

    return (
        action_ratios,
        jnp.mean(near_user),
        jnp.sum(disruption),
        jnp.mean(warmth_t),
        jnp.mean(hunger_t),
        jnp.mean(energy_t),
        jnp.mean(comfort_t),
        jnp.mean(vfe_t),
        final_glass_intact,
    )


@partial(jax.jit, static_argnames=("ticks", "ambient_temp_c"))
def _batched_rollout_jit(
    genomes: jnp.ndarray,
    keys: jnp.ndarray,
    ticks: int = 160,
    ambient_temp_c: float = 20.0,
) -> tuple[jnp.ndarray, ...]:
    return jax.vmap(
        _rollout_single_jax,
        in_axes=(0, 0, None, None),
    )(genomes, keys, ticks, ambient_temp_c)


def evaluate_genomes_batch(
    genomes: np.ndarray,
    ticks: int = 160,
    seed: int = 42,
    ambient_temp_c: float = 20.0,
) -> BatchedRolloutMetrics:
    """Evaluate a batch of (N, 18) genome vectors in parallel via JAX vmap + scan."""
    clipped = np.clip(np.asarray(genomes, dtype=np.float32), 0.0, 1.0)
    if clipped.ndim == 1:
        clipped = clipped[None, :]
    n = clipped.shape[0]
    keys = jr.split(jr.PRNGKey(seed), n)

    res = _batched_rollout_jit(
        jnp.asarray(clipped, dtype=jnp.float32),
        keys,
        ticks=ticks,
        ambient_temp_c=ambient_temp_c,
    )
    return BatchedRolloutMetrics(
        action_ratios=np.asarray(res[0], dtype=np.float32),
        user_proximity_ratio=np.asarray(res[1], dtype=np.float32),
        total_disruption=np.asarray(res[2], dtype=np.float32),
        mean_warmth=np.asarray(res[3], dtype=np.float32),
        mean_hunger=np.asarray(res[4], dtype=np.float32),
        mean_energy=np.asarray(res[5], dtype=np.float32),
        mean_comfort=np.asarray(res[6], dtype=np.float32),
        mean_vfe=np.asarray(res[7], dtype=np.float32),
        water_glass_intact=np.asarray(res[8], dtype=bool),
    )