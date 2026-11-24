from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from sim_core.aif_agent import ActiveInferencePet, CognitiveStep
from sim_core.genome import TraitGenome
from sim_core.genome_mapper import ACTION_NAMES
from sim_core.simulation import HeadlessSimulation, TickRecord
from sim_core.world import RoomWorld


@dataclass(frozen=True)
class RolloutSummary:
    ticks: int
    action_counts: dict[str, int]
    action_ratios: dict[str, float]
    action_entropy: float
    mean_vfe: float
    total_disruption: float
    user_proximity_ratio: float
    heat_seeking_ratio: float
    mean_warmth: float
    water_glass_intact: bool
    last_step: CognitiveStep
    last_record: TickRecord


def run_aif_rollout(
    genome: TraitGenome,
    ticks: int = 250,
    seed: int = 42,
    ambient_temp_c: float = 20.0,
) -> RolloutSummary:
    """Execute a deterministic Active Inference simulation rollout."""
    world = RoomWorld.create_default(seed=seed, ambient_temp_c=ambient_temp_c)
    sim = HeadlessSimulation(genome=genome, world=world, seed=seed)
    brain = ActiveInferencePet(genome=genome, seed=seed)

    counts: Counter[str] = Counter({name: 0 for name in ACTION_NAMES})
    vfe_sum = 0.0
    warmth_sum = 0.0
    near_user_ticks = 0
    last_record: TickRecord | None = None
    last_step: CognitiveStep | None = None

    for _ in range(ticks):
        cog = brain.act(sim, last_record)
        rec = sim.step(cog.target_object_id, cog.affordance_name)

        counts[cog.action_name] += 1
        vfe_sum += cog.vfe
        warmth_sum += rec.warmth
        if world.objects["user"].distance_to(sim.pet.x, sim.pet.y) <= 1.25:
            near_user_ticks += 1

        last_step = cog
        last_record = rec

    ratios = {name: counts[name] / ticks for name in ACTION_NAMES}
    entropy = -sum(p * math.log2(p) for p in ratios.values() if p > 0.0)

    assert last_step is not None and last_record is not None
    return RolloutSummary(
        ticks=ticks,
        action_counts=dict(counts),
        action_ratios=ratios,
        action_entropy=entropy,
        mean_vfe=vfe_sum / ticks,
        total_disruption=sim.pet.total_disruption,
        user_proximity_ratio=near_user_ticks / ticks,
        heat_seeking_ratio=ratios["radiator"] + ratios["laptop"],
        mean_warmth=warmth_sum / ticks,
        water_glass_intact=world.objects["water_glass"].is_intact,
        last_step=last_step,
        last_record=last_record,
    )