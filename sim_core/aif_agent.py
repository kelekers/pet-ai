from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from pymdp.agent import Agent

from sim_core.genome import TraitGenome
from sim_core.genome_mapper import (
    A_DEPENDENCIES,
    ACTION_NAMES,
    B_DEPENDENCIES,
    build_C_preferences,
    build_generative_model,
)
from sim_core.simulation import HeadlessSimulation, TickRecord


@dataclass(frozen=True)
class CognitiveStep:
    action_idx: int
    action_name: str
    target_object_id: str | None
    affordance_name: str | None
    vfe: float
    neg_efe: tuple[float, ...]
    policy_probs: tuple[float, ...]
    user_belief_calm: float


@eqx.filter_jit
def _jit_aif_step(
    agent: Agent,
    new_c: list[jnp.ndarray],
    obs: list[jnp.ndarray],
    empirical_prior: list[jnp.ndarray],
    rng_key: jnp.ndarray,
) -> tuple[
    list[jnp.ndarray],
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    list[jnp.ndarray],
    jnp.ndarray,
]:
    """Single JIT-compiled perception -> planning -> action cycle."""
    updated_agent = eqx.tree_at(lambda a: a.C, agent, new_c)
    qs, info = updated_agent.infer_states(
        obs,
        empirical_prior=empirical_prior,
        return_info=True,
    )
    q_pi, neg_efe = updated_agent.infer_policies(qs)

    rng_key, subkey = jr.split(rng_key)
    sample_keys = jr.split(subkey, updated_agent.batch_size + 1)
    action = updated_agent.sample_action(q_pi, rng_key=sample_keys[1:])
    next_prior = updated_agent.update_empirical_prior(action, qs)
    return qs, info["vfe"], q_pi, neg_efe, action, next_prior, rng_key


class ActiveInferencePet:
    """JAX-backed pymdp 1.0 Active Inference brain for a single TraitGenome."""

    def __init__(self, genome: TraitGenome, seed: int = 42) -> None:
        self.genome = genome
        self.rng_key = jr.PRNGKey(seed)

        spec = build_generative_model(genome)
        self.agent = Agent(
            A=spec.A,
            B=spec.B,
            C=spec.C,
            D=spec.D,
            A_dependencies=A_DEPENDENCIES,
            B_dependencies=B_DEPENDENCIES,
            policy_len=1,
            gamma=spec.gamma,
            alpha=spec.alpha,
            use_utility=True,
            use_states_info_gain=True,
            batch_size=1,
        )
        self.empirical_prior = self.agent.D
        self.last_action_idx = 0

    @staticmethod
    def encode_observation(
        sim: HeadlessSimulation,
        last_record: TickRecord | None,
    ) -> list[jnp.ndarray]:
        """Discretize live simulation state into the 6 POMDP observation modalities."""
        h = sim.pet.homeostasis
        local_temp = sim.world.local_temperature(sim.pet.x, sim.pet.y)

        # 0: thermal (0=cold, 1=neutral, 2=warm)
        if local_temp >= 28.0 or h.warmth >= 0.86:
            o_thermal = 2
        elif local_temp <= 22.0 and h.warmth < 0.72:
            o_thermal = 0
        else:
            o_thermal = 1

        # 1: satiety (0=hungry, 1=fed)
        o_satiety = 0 if h.hunger >= 0.40 else 1

        # 2: effort (0=restful, 1=moderate, 2=high_exertion)
        action_str = last_record.action if last_record is not None else "idle"
        if "knock_over" in action_str or "scratch" in action_str:
            o_effort = 2
        elif "approach:" in action_str or "rub" in action_str or "keyboard" in action_str:
            o_effort = 1
        else:
            o_effort = 0

        # 3: stimulation (0=dull, 1=mild_novelty, 2=chaotic_impact)
        if last_record is not None and last_record.disruption_delta >= 0.5:
            o_stim = 2
        elif h.stimulation >= 0.60 or "stare" in action_str or "rub" in action_str:
            o_stim = 1
        else:
            o_stim = 0

        # 4: social_contact (0=solitary, 1=observing, 2=intimate)
        user_obj = sim.world.objects["user"]
        dist_user = user_obj.distance_to(sim.pet.x, sim.pet.y)
        if "rub_against_user" in action_str or dist_user <= 0.6:
            o_social = 2
        elif "stare_at_user" in action_str or "laptop" in action_str:
            o_social = 1
        else:
            o_social = 0

        # 5: user_cue (0=no_signal, 1=cue_calm, 2=cue_busy)
        if "stare_at_user" in action_str or "rub_against_user" in action_str:
            o_cue = 1 if (sim.world.tick // 35) % 2 == 0 else 2
        else:
            o_cue = 0

        return [
            jnp.array([o_thermal], dtype=jnp.int32),
            jnp.array([o_satiety], dtype=jnp.int32),
            jnp.array([o_effort], dtype=jnp.int32),
            jnp.array([o_stim], dtype=jnp.int32),
            jnp.array([o_social], dtype=jnp.int32),
            jnp.array([o_cue], dtype=jnp.int32),
        ]

    @staticmethod
    def decode_action(
        action_idx: int,
        sim: HeadlessSimulation,
    ) -> tuple[str | None, str | None]:
        """Map discrete policy index to a Smart Object + affordance in RoomWorld."""
        name = ACTION_NAMES[action_idx]
        if name == "idle":
            return (None, None)
        if name == "radiator":
            return ("radiator", "curl_for_warmth")
        if name == "laptop":
            return ("laptop", "sit_on_keyboard")
        if name == "food_bowl":
            return ("food_bowl", "eat")
        if name == "cardboard_box":
            return ("cardboard_box", "hide_and_nap")
        if name == "stare_user":
            return ("user", "stare_at_user")
        if name == "rub_user":
            return ("user", "rub_against_user")
        if name == "chaos_play":
            glass = sim.world.objects.get("water_glass")
            if glass is not None and glass.is_intact and sim.genome.agility >= 0.25:
                return ("water_glass", "knock_over")
            return ("scratching_post", "scratch_vigorously")
        return (None, None)

    def act(
        self,
        sim: HeadlessSimulation,
        last_record: TickRecord | None = None,
        c_offsets: list[jnp.ndarray] | None = None,
    ) -> CognitiveStep:
        """Run one JIT-compiled perception -> planning -> action cycle."""
        base_c = build_C_preferences(self.genome, sim.pet.homeostasis)
        if c_offsets is not None:
            adjusted_c: list[jnp.ndarray] = []
            for c_m, offset_m in zip(base_c, c_offsets, strict=True):
                combined = c_m + jnp.asarray(offset_m, dtype=jnp.float32)
                shifted = combined - jnp.min(combined) + 0.05
                adjusted_c.append(jnp.expand_dims(shifted, axis=0))
            new_c = adjusted_c
        else:
            new_c = [jnp.expand_dims(c_m, axis=0) for c_m in base_c]

        obs = self.encode_observation(sim, last_record)

        qs, vfe_arr, q_pi, neg_efe, action, next_prior, next_key = _jit_aif_step(
            self.agent,
            new_c,
            obs,
            self.empirical_prior,
            self.rng_key,
        )

        self.empirical_prior = next_prior
        self.rng_key = next_key

        action_idx = int(jnp.asarray(action).reshape(-1)[0])
        self.last_action_idx = action_idx
        obj_id, aff_name = self.decode_action(action_idx, sim)

        user_belief = jnp.asarray(qs[1]).reshape(-1)
        return CognitiveStep(
            action_idx=action_idx,
            action_name=ACTION_NAMES[action_idx],
            target_object_id=obj_id,
            affordance_name=aff_name,
            vfe=float(jnp.asarray(vfe_arr).reshape(-1)[0]),
            neg_efe=tuple(float(x) for x in jnp.asarray(neg_efe).reshape(-1)),
            policy_probs=tuple(float(x) for x in jnp.asarray(q_pi).reshape(-1)),
            user_belief_calm=float(user_belief[0]),
        )