from __future__ import annotations

import numpy as np

from memory.emotion import EmotionEngine
from memory.memory_store import EpisodicMemoryStore, HashEmbedder
from memory.prior_adaptation import AdaptivePetAgent, SensoryEvent
from sim_core.genome import TraitGenome
from sim_core.world import RoomWorld


def test_pad_emotion_reacts_to_stimulus_and_decays_to_baseline() -> None:
    genome = TraitGenome(neuroticism=0.7, agreeableness=0.5, dominance=0.4)
    engine = EmotionEngine(genome)
    baseline_p = engine.baseline.pleasure

    after_shock, _ = engine.apply_stimulus(delta_p=-0.6, delta_a=0.6, delta_d=-0.4)
    assert after_shock.pleasure < baseline_p - 0.3
    assert after_shock.arousal > engine.baseline.arousal + 0.3
    assert after_shock.fear_intensity > 0.35

    for _ in range(120):
        engine.step(vfe=2.0, comfort=0.75, ambient_noise=0.1)

    assert abs(engine.state.pleasure - baseline_p) < abs(after_shock.pleasure - baseline_p)


def test_sqlite_vec_records_retrieves_and_computes_entity_affinity() -> None:
    with EpisodicMemoryStore(":memory:", embedder=HashEmbedder()) as store:
        store.record_episode(
            tick=10,
            entity="user",
            action="user:rub_against_user",
            event_tag="calm_petting",
            summary="User gently petted the cat near the desk",
            delta_p=0.45,
            delta_a=-0.15,
            delta_d=0.15,
            importance=1.2,
            x=3.0,
            y=4.3,
        )
        store.record_episode(
            tick=15,
            entity="food_bowl",
            action="food_bowl:eat",
            event_tag="loud_clatter",
            summary="Metal pan clattered loudly near the kitchen food bowl",
            delta_p=-0.55,
            delta_a=0.60,
            delta_d=-0.40,
            importance=1.4,
            x=1.0,
            y=0.8,
        )

        hits = store.retrieve_similar("loud metal pan clattered near kitchen", k=1)
        assert len(hits) == 1
        assert hits[0].entity == "food_bowl"
        assert hits[0].event_tag == "loud_clatter"

        user_trust, user_fear = store.entity_affinity("user", current_tick=20)
        kitchen_trust, kitchen_fear = store.entity_affinity("food_bowl", current_tick=20)
        assert user_trust > 0.25 and user_fear < 0.05
        assert kitchen_trust < -0.25 and kitchen_fear > 0.35


def test_identical_twins_diverge_in_trust_and_spatial_behavior() -> None:
    """Sprint 4 DoD: Two identical clones diverge under calm vs erratic treatment."""
    twin_genome = TraitGenome(
        neuroticism=0.58,
        extraversion=0.50,
        dominance=0.45,
        impulsiveness=0.35,
        agreeableness=0.55,
        curiosity=0.55,
        laziness=0.55,
        attachment=0.52,
        chaos=0.25,
        fur_density=0.65,
    )

    clone_a = AdaptivePetAgent(
        genome=twin_genome,
        world=RoomWorld.create_default(seed=42, ambient_temp_c=21.0),
        embedder=HashEmbedder(),
        seed=42,
    )
    clone_b = AdaptivePetAgent(
        genome=twin_genome,
        world=RoomWorld.create_default(seed=42, ambient_temp_c=21.0),
        embedder=HashEmbedder(),
        seed=42,
    )

    user_xy = np.array([3.0, 4.3])
    dists_a: list[float] = []
    dists_b: list[float] = []
    rub_a = 0
    rub_b = 0

    for tick in range(1, 201):
        if tick % 20 == 5:
            clone_a.inject_event(SensoryEvent.calm_petting())
            clone_b.inject_event(SensoryEvent.startling_user_shout())
        elif tick % 40 == 25:
            clone_a.inject_event(SensoryEvent.gentle_feeding())
            clone_b.inject_event(SensoryEvent.loud_kitchen_clatter())

        res_a = clone_a.step()
        res_b = clone_b.step()

        dists_a.append(float(np.linalg.norm(np.array([res_a.tick_record.x, res_a.tick_record.y]) - user_xy)))
        dists_b.append(float(np.linalg.norm(np.array([res_b.tick_record.x, res_b.tick_record.y]) - user_xy)))
        if res_a.cognitive_step.action_name == "rub_user":
            rub_a += 1
        if res_b.cognitive_step.action_name == "rub_user":
            rub_b += 1

    # Trust divergence
    assert res_a.user_trust > 0.35
    assert res_b.user_trust < -0.35
    assert res_a.user_trust - res_b.user_trust > 0.90

    # Behavioral and spatial divergence
    assert rub_a > rub_b + 30
    assert float(np.mean(dists_a)) < float(np.mean(dists_b)) - 0.50

    clone_a.memory.close()
    clone_b.memory.close()