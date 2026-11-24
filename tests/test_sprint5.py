from __future__ import annotations

import time

from memory.memory_store import HashEmbedder
from neuro_symbolic.llm_director import DeterministicMockAdapter, SkillSynthesizer, SynthesisContext
from neuro_symbolic.orchestrator import NeuroSymbolicPetAgent
from neuro_symbolic.skill_dsl import BehaviorSkill, SkillStep
from neuro_symbolic.skill_library import SkillLibrary
from neuro_symbolic.verifier import SymbolicSkillVerifier
from sim_core.genome import TraitGenome
from sim_core.world import RoomWorld


def test_verifier_rejects_anomaly_skills_for_grounded_pet() -> None:
    verifier = SymbolicSkillVerifier()
    world = RoomWorld.create_default(seed=42)
    grounded = TraitGenome.preset_hairless_sphynx()  # all anomaly traits == 0.0

    illegal_skill = BehaviorSkill(
        name="sneaky_ceiling_walk",
        trigger_context="try to break gravity without anomaly genome",
        cooldown_ticks=15,
        steps=[
            SkillStep(primitive="invert_gravity", duration_ticks=3),
            SkillStep(primitive="phase_object", target_object="cardboard_box", duration_ticks=2),
            SkillStep(primitive="mimic_text", target_object="user", payload="hello"),
        ],
    )
    res = verifier.verify(illegal_skill, genome=grounded, world=world)
    assert not res.is_valid
    assert res.uses_anomaly
    assert len(res.errors) >= 3
    assert any("gravity_defiance" in err for err in res.errors)
    assert any("spatial_phase" in err for err in res.errors)
    assert any("mimicry_drive" in err for err in res.errors)


def test_verifier_allows_gravity_bypass_for_low_agility_impossible_cat() -> None:
    verifier = SymbolicSkillVerifier()
    world = RoomWorld.create_default(seed=42)
    clumsy_impossible = TraitGenome(agility=0.20, gravity_defiance=0.95)

    # Without invert_gravity, approaching high_shelf fails agility >= 0.65
    plain_climb = BehaviorSkill(
        name="plain_shelf_climb",
        trigger_context="attempt to reach high shelf normally",
        cooldown_ticks=15,
        steps=[
            SkillStep(primitive="approach", target_object="high_shelf", duration_ticks=2),
            SkillStep(primitive="wait", duration_ticks=2),
        ],
    )
    assert not verifier.verify(plain_climb, genome=clumsy_impossible, world=world).is_valid

    # With invert_gravity first, symbolic verifier permits reaching high_shelf
    gravity_climb = BehaviorSkill(
        name="inverted_shelf_ascension",
        trigger_context="invert gravity to walk onto high shelf",
        cooldown_ticks=15,
        steps=[
            SkillStep(primitive="invert_gravity", duration_ticks=3),
            SkillStep(primitive="approach", target_object="high_shelf", duration_ticks=3),
        ],
    )
    ok = verifier.verify(gravity_climb, genome=clumsy_impossible, world=world)
    assert ok.is_valid
    assert ok.required_anomalies == ("invert_gravity",)


def test_symbolic_repair_loop_recovers_from_invalid_first_draft() -> None:
    world = RoomWorld.create_default(seed=42)
    grounded = TraitGenome.preset_hairless_sphynx()
    synth = SkillSynthesizer(
        adapter=DeterministicMockAdapter(emit_illegal_first_draft=True),
        max_repair_attempts=2,
    )
    from memory.emotion import PADState

    ctx = SynthesisContext(
        genome=grounded,
        pad=PADState(),
        recent_memories=(),
        existing_skills=(),
        world_objects=tuple(world.objects.keys()),
        vfe=4.5,
        trigger_reason="high free energy",
    )
    outcome = synth.synthesize_and_verify(ctx, world=world)
    synth.shutdown()

    assert outcome.attempts == 2
    assert outcome.skill is not None
    assert outcome.verification is not None and outcome.verification.is_valid
    assert not outcome.verification.uses_anomaly


def test_skill_library_stores_retrieves_and_respects_cooldown() -> None:
    with SkillLibrary(":memory:", embedder=HashEmbedder()) as lib:
        skill = BehaviorSkill(
            name="ceiling_shelf_ascension",
            trigger_context="gravity walk onto high_shelf",
            cooldown_ticks=20,
            steps=[
                SkillStep(primitive="invert_gravity", duration_ticks=2),
                SkillStep(primitive="approach", target_object="high_shelf", duration_ticks=2),
            ],
        )
        lib.add_skill(skill, uses_anomaly=True)
        lib.record_execution("ceiling_shelf_ascension", tick=50)

        # Blocked while on cooldown at tick 60
        assert lib.retrieve("gravity walk onto high_shelf", current_tick=60) == []

        # Available again at tick 75
        hits = lib.retrieve("gravity walk onto high_shelf", current_tick=75)
        assert len(hits) == 1
        assert hits[0].skill.name == "ceiling_shelf_ascension"


def test_impossible_cat_invents_and_reuses_three_rituals_while_grounded_stays_normal() -> None:
    """Sprint 5 DoD: Impossible cat synthesizes, executes, and reuses >=3 rituals."""
    impossible_genome = TraitGenome.preset_impossible_cat().model_copy(
        update={"mimicry_drive": 0.90}
    )
    grounded_genome = TraitGenome.preset_hairless_sphynx()

    imp_agent = NeuroSymbolicPetAgent(
        genome=impossible_genome,
        world=RoomWorld.create_default(seed=42),
        adapter=DeterministicMockAdapter(),
        embedder=HashEmbedder(),
        seed=42,
    )
    grd_agent = NeuroSymbolicPetAgent(
        genome=grounded_genome,
        world=RoomWorld.create_default(seed=42),
        adapter=DeterministicMockAdapter(),
        embedder=HashEmbedder(),
        seed=42,
    )

    prompts = {
        5: "ceiling high_shelf gravity ritual",
        30: "cardboard_box spatial anomaly near user",
        55: "telepathic gaze beside warm laptop",
        85: "ceiling high_shelf gravity ritual",
        115: "cardboard_box spatial anomaly near user",
    }

    imp_anomalies_seen = False
    grd_anomalies_seen = False
    imp_reused = False

    for tick in range(1, 145):
        if tick in prompts:
            imp_agent.send_user_prompt(prompts[tick])
            grd_agent.send_user_prompt(prompts[tick])

        r_imp = imp_agent.step(async_synthesis=False)
        r_grd = grd_agent.step(async_synthesis=False)

        if any(bool(v) for v in r_imp.anomaly_flags.values()):
            imp_anomalies_seen = True
        if any(bool(v) for v in r_grd.anomaly_flags.values()):
            grd_anomalies_seen = True
        if r_imp.mode == "skill_reused":
            imp_reused = True

    assert imp_agent.library.count() >= 3
    assert imp_agent.library.total_reuses() >= 1
    assert imp_reused
    assert imp_anomalies_seen
    assert not grd_anomalies_seen

    imp_agent.close()
    grd_agent.close()


def test_async_synthesis_does_not_block_active_inference_loop() -> None:
    agent = NeuroSymbolicPetAgent(
        genome=TraitGenome.preset_impossible_cat(),
        world=RoomWorld.create_default(seed=42),
        adapter=DeterministicMockAdapter(),
        embedder=HashEmbedder(),
        seed=42,
    )
    agent.send_user_prompt("ceiling high_shelf gravity ritual")
    agent.step(async_synthesis=True)

    # Give worker thread a moment, then advance ticks until skill is collected and executed
    synthesized = False
    for _ in range(15):
        time.sleep(0.01)
        res = agent.step(async_synthesis=True)
        if res.synthesized_skill is not None or res.active_skill_name is not None:
            synthesized = True
            break

    assert synthesized
    agent.close()