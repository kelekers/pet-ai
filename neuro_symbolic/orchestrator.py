from __future__ import annotations

from dataclasses import dataclass

from memory.emotion import PADState
from memory.memory_store import EpisodicMemoryStore, TextEmbedder
from memory.prior_adaptation import AdaptivePetAgent
from neuro_symbolic.llm_director import (
    DeterministicMockAdapter,
    DirectorAdapter,
    SkillSynthesizer,
    SynthesisContext,
    SynthesisOutcome,
)
from neuro_symbolic.skill_dsl import BehaviorSkill, SkillStep
from neuro_symbolic.skill_library import SkillLibrary
from neuro_symbolic.verifier import SymbolicSkillVerifier
from sim_core.genome import TraitGenome
from sim_core.simulation import TickRecord
from sim_core.world import RoomWorld


@dataclass
class AnomalyEffectState:
    """Active surreal physics/rendering flags exported to telemetry & Godot 4."""

    gravity_inverted_until: int = -1
    phased_object_id: str | None = None
    phased_until: int = -1
    last_mimic_text: str | None = None
    telepathic_target: str | None = None
    telepathic_until: int = -1

    def active_flags(self, tick: int) -> dict[str, object]:
        return {
            "gravity_inverted": tick <= self.gravity_inverted_until,
            "phased_object": self.phased_object_id if tick <= self.phased_until else None,
            "telepathic_target": (
                self.telepathic_target if tick <= self.telepathic_until else None
            ),
            "last_mimic_text": self.last_mimic_text,
        }


@dataclass
class _ActiveSkillRun:
    skill: BehaviorSkill
    uses_anomaly: bool
    reused_from_library: bool
    step_idx: int = 0
    ticks_in_step: int = 0


@dataclass(frozen=True)
class NeuroSymbolicTick:
    tick_record: TickRecord
    pad: PADState
    mode: str  # "active_inference" | "skill_new" | "skill_reused"
    active_skill_name: str | None
    active_primitive: str | None
    vfe: float
    anomaly_flags: dict[str, object]
    synthesized_skill: str | None = None


class NeuroSymbolicPetAgent:
    """Full Sprint 5 agent: Active Inference + PAD Memory + Neuro-Symbolic Skill Library."""

    def __init__(
        self,
        genome: TraitGenome,
        world: RoomWorld | None = None,
        memory_store: EpisodicMemoryStore | None = None,
        skill_library: SkillLibrary | None = None,
        adapter: DirectorAdapter | None = None,
        embedder: TextEmbedder | None = None,
        vfe_trigger_threshold: float = 3.2,
        vfe_sustain_ticks: int = 8,
        seed: int = 42,
    ) -> None:
        self.adaptive = AdaptivePetAgent(
            genome=genome,
            world=world,
            memory_store=memory_store,
            embedder=embedder,
            seed=seed,
        )
        self.verifier = SymbolicSkillVerifier()
        self.library = skill_library or SkillLibrary(":memory:", embedder=embedder)
        self.synthesizer = SkillSynthesizer(
            adapter=adapter or DeterministicMockAdapter(),
            verifier=self.verifier,
        )
        self.anomaly_state = AnomalyEffectState()
        self.vfe_trigger_threshold = vfe_trigger_threshold
        self.vfe_sustain_ticks = vfe_sustain_ticks

        self._active_run: _ActiveSkillRun | None = None
        self._high_vfe_streak: int = 0
        self._last_vfe: float = 0.0
        self._pending_user_prompt: str | None = None

    @property
    def genome(self) -> TraitGenome:
        return self.adaptive.genome

    @property
    def world(self) -> RoomWorld:
        return self.adaptive.world

    def send_user_prompt(self, text: str) -> None:
        """Queue a freeform user utterance/action to trigger recall or skill synthesis."""
        self._pending_user_prompt = text.strip()

    def _build_context(self, trigger_reason: str, user_prompt: str | None) -> SynthesisContext:
        recalled = self.adaptive.memory.retrieve_similar(
            user_prompt or trigger_reason,
            k=4,
        )
        return SynthesisContext(
            genome=self.genome,
            pad=self.adaptive.emotion.state,
            recent_memories=tuple(recalled),
            existing_skills=tuple(self.library.list_names()),
            world_objects=tuple(self.world.objects.keys()),
            vfe=self._last_vfe,
            trigger_reason=trigger_reason,
            user_prompt=user_prompt,
        )

    def _start_skill(self, skill: BehaviorSkill, uses_anomaly: bool, reused: bool) -> None:
        self.library.record_execution(skill.name, tick=self.world.tick, success=True)
        self._active_run = _ActiveSkillRun(
            skill=skill,
            uses_anomaly=uses_anomaly,
            reused_from_library=reused,
        )

    def _execute_primitive_tick(self, step: SkillStep) -> TickRecord:
        sim = self.adaptive.sim
        tick_now = self.world.tick + 1
        prim = step.primitive

        # Update surreal effect flags (guaranteed pre-verified against genome gates)
        if prim == "invert_gravity":
            self.anomaly_state.gravity_inverted_until = tick_now + step.duration_ticks + 4
        elif prim == "phase_object" and step.target_object:
            self.anomaly_state.phased_object_id = step.target_object
            self.anomaly_state.phased_until = tick_now + step.duration_ticks + 2
            obj = self.world.objects[step.target_object]
            sim.pet.x, sim.pet.y = obj.x, obj.y
        elif prim == "mimic_text" and step.payload:
            self.anomaly_state.last_mimic_text = step.payload
        elif prim == "telepathic_gaze" and step.target_object:
            self.anomaly_state.telepathic_target = step.target_object
            self.anomaly_state.telepathic_until = tick_now + step.duration_ticks + 2

        # Map physical movement / interaction onto HeadlessSimulation
        if prim == "approach" and step.target_object in self.world.objects:
            obj = self.world.objects[step.target_object]
            sim._move_toward(obj.x, obj.y)
            rec = sim.step(None, None)
        elif prim == "stare" and step.target_object == "user":
            rec = sim.step("user", "stare_at_user")
        elif prim == "nudge" and step.target_object == "water_glass":
            rec = sim.step("water_glass", "knock_over")
        else:
            rec = sim.step(None, None)

        # Modest stimulation gain from executing a self-invented ritual
        sim.pet.homeostasis.stimulation = min(1.0, sim.pet.homeostasis.stimulation + 0.04)
        self.adaptive._last_record = rec
        return rec

    def _handle_completed_outcome(self, outcome: SynthesisOutcome) -> str | None:
        if outcome.skill is None or outcome.verification is None or not outcome.verification.is_valid:
            return None
        self.library.add_skill(outcome.skill, uses_anomaly=outcome.verification.uses_anomaly)
        self._start_skill(
            outcome.skill,
            uses_anomaly=outcome.verification.uses_anomaly,
            reused=False,
        )
        return outcome.skill.name

    def step(self, async_synthesis: bool = False) -> NeuroSymbolicTick:
        """Advance one simulation tick across Active Inference, Skill Library, and Director."""
        synthesized_name: str | None = None

        # 1. Collect any finished async synthesis job
        if async_synthesis:
            completed = self.synthesizer.poll_async()
            if completed is not None and self._active_run is None:
                synthesized_name = self._handle_completed_outcome(completed)

        # 2. If currently executing a verified skill, step its current primitive
        if self._active_run is not None:
            run = self._active_run
            current_step = run.skill.steps[run.step_idx]
            rec = self._execute_primitive_tick(current_step)
            pad = self.adaptive.emotion.step(
                vfe=max(0.5, self._last_vfe * 0.7),
                comfort=rec.comfort,
                ambient_noise=self.world.ambient_noise,
            )

            mode = "skill_reused" if run.reused_from_library else "skill_new"
            skill_name = run.skill.name
            prim_name = current_step.primitive

            run.ticks_in_step += 1
            if run.ticks_in_step >= current_step.duration_ticks:
                run.step_idx += 1
                run.ticks_in_step = 0
                if run.step_idx >= len(run.skill.steps):
                    # Ritual finished: log episodic memory
                    self.adaptive.memory.record_episode(
                        tick=rec.tick,
                        entity=current_step.target_object or "room",
                        action=f"skill:{skill_name}",
                        event_tag="anomaly_ritual" if run.uses_anomaly else "grounded_ritual",
                        summary=f"Executed ritual {skill_name} ({run.skill.trigger_context})",
                        delta_p=0.18,
                        delta_a=0.12 if run.uses_anomaly else -0.05,
                        delta_d=0.15,
                        importance=1.0,
                        x=rec.x,
                        y=rec.y,
                    )
                    self._active_run = None
                    self._high_vfe_streak = 0

            return NeuroSymbolicTick(
                tick_record=rec,
                pad=pad,
                mode=mode,
                active_skill_name=skill_name,
                active_primitive=prim_name,
                vfe=self._last_vfe,
                anomaly_flags=self.anomaly_state.active_flags(rec.tick),
                synthesized_skill=synthesized_name,
            )

        # 3. Check recall/synthesis triggers (user prompt or sustained high Free Energy)
        trigger_reason: str | None = None
        user_prompt = self._pending_user_prompt
        if user_prompt is not None:
            self._pending_user_prompt = None
            trigger_reason = f"user interaction: {user_prompt}"
        elif self._high_vfe_streak >= self.vfe_sustain_ticks:
            self._high_vfe_streak = 0
            trigger_reason = (
                f"sustained high free energy while {self.adaptive.emotion.state.mood_label()}"
            )

        if trigger_reason is not None:
            # First attempt semantic recall & reuse from SkillLibrary
            matches = self.library.retrieve(
                context_query=user_prompt or trigger_reason,
                k=3,
                max_distance=0.58,
                current_tick=self.world.tick,
            )
            reusable = None
            for cand in matches:
                ver = self.verifier.verify(
                    skill=cand.skill,
                    genome=self.genome,
                    world=self.world,
                    current_tick=self.world.tick,
                    last_executed_tick=cand.last_executed_tick,
                )
                if ver.is_valid:
                    reusable = cand
                    break

            if reusable is not None:
                self._start_skill(reusable.skill, uses_anomaly=reusable.uses_anomaly, reused=True)
                return self.step(async_synthesis=async_synthesis)

            # Otherwise synthesize a novel ritual
            ctx = self._build_context(trigger_reason=trigger_reason, user_prompt=user_prompt)
            if async_synthesis:
                self.synthesizer.request_async(ctx, self.world)
            else:
                outcome = self.synthesizer.synthesize_and_verify(ctx, self.world)
                synthesized_name = self._handle_completed_outcome(outcome)
                if self._active_run is not None:
                    tick_res = self.step(async_synthesis=False)
                    return NeuroSymbolicTick(
                        tick_record=tick_res.tick_record,
                        pad=tick_res.pad,
                        mode=tick_res.mode,
                        active_skill_name=tick_res.active_skill_name,
                        active_primitive=tick_res.active_primitive,
                        vfe=tick_res.vfe,
                        anomaly_flags=tick_res.anomaly_flags,
                        synthesized_skill=synthesized_name,
                    )

        # 4. Default fast path: Sprint 4 Active Inference + PAD + Memory
        base = self.adaptive.step()
        self._last_vfe = base.cognitive_step.vfe
        if self._last_vfe >= self.vfe_trigger_threshold:
            self._high_vfe_streak += 1
        else:
            self._high_vfe_streak = max(0, self._high_vfe_streak - 1)

        return NeuroSymbolicTick(
            tick_record=base.tick_record,
            pad=base.pad,
            mode="active_inference",
            active_skill_name=None,
            active_primitive=None,
            vfe=self._last_vfe,
            anomaly_flags=self.anomaly_state.active_flags(base.tick_record.tick),
            synthesized_skill=synthesized_name,
        )

    def close(self) -> None:
        self.synthesizer.shutdown()
        self.library.close()
        self.adaptive.memory.close()