from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from memory.emotion import EmotionEngine, PADState
from memory.memory_store import Episode, EpisodicMemoryStore, TextEmbedder
from sim_core.aif_agent import ActiveInferencePet, CognitiveStep
from sim_core.genome import TraitGenome
from sim_core.simulation import HeadlessSimulation, TickRecord
from sim_core.world import RoomWorld


@dataclass(frozen=True)
class SensoryEvent:
    """External interaction or environmental event experienced by the pet."""

    entity: str
    event_tag: str
    description: str
    delta_p: float
    delta_a: float
    delta_d: float
    importance: float = 1.0
    noise_spike: float = 0.0

    @classmethod
    def calm_petting(cls) -> SensoryEvent:
        return cls(
            entity="user",
            event_tag="calm_petting",
            description="User spoke softly and gently petted the cat",
            delta_p=0.40,
            delta_a=-0.18,
            delta_d=0.12,
            importance=1.2,
        )

    @classmethod
    def gentle_feeding(cls) -> SensoryEvent:
        return cls(
            entity="food_bowl",
            event_tag="gentle_feeding",
            description="User calmly placed warm treats in the food bowl",
            delta_p=0.35,
            delta_a=-0.10,
            delta_d=0.10,
            importance=1.0,
        )

    @classmethod
    def loud_kitchen_clatter(cls) -> SensoryEvent:
        return cls(
            entity="food_bowl",
            event_tag="loud_clatter",
            description="Heavy metal pan crashed loudly near the kitchen food bowl",
            delta_p=-0.48,
            delta_a=0.55,
            delta_d=-0.38,
            importance=1.4,
            noise_spike=0.85,
        )

    @classmethod
    def startling_user_shout(cls) -> SensoryEvent:
        return cls(
            entity="user",
            event_tag="startling_noise",
            description="User slammed desk and shouted unpredictably near the cat",
            delta_p=-0.50,
            delta_a=0.58,
            delta_d=-0.42,
            importance=1.5,
            noise_spike=0.80,
        )


def compute_c_offsets(
    pad: PADState,
    user_trust: float,
    user_fear: float,
    kitchen_fear: float,
    retrieved: list[Episode] | None = None,
) -> list[jnp.ndarray]:
    """Map continuous PAD mood and associative memories onto Active Inference C log-offsets."""
    fear_drive = max(pad.fear_intensity, user_fear)

    # Secondary contextual nudge from top retrieved episodic memories
    recalled_valence = 0.0
    if retrieved:
        recalled_valence = sum(ep.delta_p for ep in retrieved) / len(retrieved)

    # 0: thermal [cold, neutral, warm]
    c_thermal = jnp.array([0.0, 0.0, 0.4 * fear_drive], dtype=jnp.float32)

    # 1: satiety [hungry, fed] -> kitchen trauma temporarily suppresses food-bowl approach
    c_satiety = jnp.array([0.0, -2.6 * kitchen_fear], dtype=jnp.float32)

    # 2: effort [restful, moderate, high_exertion] -> fear favors hiding/freezing
    c_effort = jnp.array(
        [
            2.4 * fear_drive,
            0.0,
            -2.2 * fear_drive + 0.6 * max(0.0, pad.pleasure),
        ],
        dtype=jnp.float32,
    )

    # 3: stimulation [dull, mild_novelty, chaotic_impact]
    c_stim = jnp.array(
        [
            0.5 * fear_drive,
            0.6 * max(0.0, recalled_valence),
            -2.6 * fear_drive + 0.5 * max(0.0, pad.dominance),
        ],
        dtype=jnp.float32,
    )

    # 4: social_contact [solitary, observing, intimate] -> core trust vs avoidance adaptation
    c_social = jnp.array(
        [
            4.2 * user_fear - 2.4 * user_trust,
            -1.4 * user_fear + 0.5 * max(0.0, user_trust),
            4.6 * user_trust - 5.0 * user_fear + 0.8 * recalled_valence,
        ],
        dtype=jnp.float32,
    )

    # 5: user_cue stays purely epistemic
    c_cue = jnp.zeros((3,), dtype=jnp.float32)

    return [c_thermal, c_satiety, c_effort, c_stim, c_social, c_cue]


@dataclass(frozen=True)
class AdaptiveStepResult:
    tick_record: TickRecord
    cognitive_step: CognitiveStep
    pad: PADState
    user_trust: float
    user_fear: float
    kitchen_fear: float
    recalled_count: int


class AdaptivePetAgent:
    """Full Sprint 4 embodied agent: Active Inference + PAD Emotions + sqlite-vec Memory."""

    def __init__(
        self,
        genome: TraitGenome,
        world: RoomWorld | None = None,
        memory_store: EpisodicMemoryStore | None = None,
        embedder: TextEmbedder | None = None,
        seed: int = 42,
    ) -> None:
        self.genome = genome
        self.world = world or RoomWorld.create_default(seed=seed)
        self.sim = HeadlessSimulation(genome=genome, world=self.world, seed=seed)
        self.brain = ActiveInferencePet(genome=genome, seed=seed)
        self.emotion = EmotionEngine(genome=genome)
        self.memory = memory_store or EpisodicMemoryStore(":memory:", embedder=embedder)
        self._last_record: TickRecord | None = None

    def inject_event(self, event: SensoryEvent) -> int:
        """Deliver an external sensory event, update PAD, and persist an episodic memory."""
        if event.noise_spike > 0.0:
            self.world.ambient_noise = min(1.0, max(self.world.ambient_noise, event.noise_spike))

        _, (eff_p, eff_a, eff_d) = self.emotion.apply_stimulus(
            delta_p=event.delta_p,
            delta_a=event.delta_a,
            delta_d=event.delta_d,
        )
        return self.memory.record_episode(
            tick=self.world.tick,
            entity=event.entity,
            action=self.sim.pet.last_action,
            event_tag=event.event_tag,
            summary=event.description,
            delta_p=eff_p,
            delta_a=eff_a,
            delta_d=eff_d,
            importance=event.importance,
            x=self.sim.pet.x,
            y=self.sim.pet.y,
        )

    def step(self) -> AdaptiveStepResult:
        """Retrieve relevant memories, adapt Active Inference priors, act, and update emotions."""
        tick = self.world.tick
        user_trust, user_fear = self.memory.entity_affinity("user", current_tick=tick)
        _, kitchen_fear = self.memory.entity_affinity("food_bowl", current_tick=tick)

        query = (
            f"user food_bowl {self.sim.pet.last_action} mood {self.emotion.state.mood_label()}"
        )
        recalled = self.memory.retrieve_similar(query, k=4)

        c_offsets = compute_c_offsets(
            pad=self.emotion.state,
            user_trust=user_trust,
            user_fear=user_fear,
            kitchen_fear=kitchen_fear,
            retrieved=recalled,
        )

        cog = self.brain.act(self.sim, last_record=self._last_record, c_offsets=c_offsets)
        rec = self.sim.step(cog.target_object_id, cog.affordance_name)
        self._last_record = rec

        pad = self.emotion.step(
            vfe=cog.vfe,
            comfort=rec.comfort,
            ambient_noise=self.world.ambient_noise,
        )

        # Autonomously log high-salience self-experienced episodes (e.g., breaking objects)
        if rec.disruption_delta >= 0.8 and cog.target_object_id is not None:
            self.memory.record_episode(
                tick=rec.tick,
                entity=cog.target_object_id,
                action=rec.action,
                event_tag="high_disruption",
                summary=f"Knocked over {cog.target_object_id} with loud splash",
                delta_p=0.15 * self.genome.chaos - 0.25 * self.genome.neuroticism,
                delta_a=0.35,
                delta_d=0.20 * self.genome.dominance,
                importance=0.9,
                x=rec.x,
                y=rec.y,
            )

        return AdaptiveStepResult(
            tick_record=rec,
            cognitive_step=cog,
            pad=pad,
            user_trust=user_trust,
            user_fear=user_fear,
            kitchen_fear=kitchen_fear,
            recalled_count=len(recalled),
        )