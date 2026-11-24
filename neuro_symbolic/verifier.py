from __future__ import annotations

from dataclasses import dataclass

from neuro_symbolic.skill_dsl import ANOMALY_GATES, BehaviorSkill
from sim_core.genome import TraitGenome
from sim_core.world import RoomWorld

MAX_SKILL_DURATION_TICKS = 36
_TARGET_REQUIRED_PRIMITIVES: frozenset[str] = frozenset(
    {"approach", "stare", "nudge", "phase_object", "telepathic_gaze"}
)


@dataclass(frozen=True)
class VerificationResult:
    is_valid: bool
    errors: tuple[str, ...]
    uses_anomaly: bool
    required_anomalies: tuple[str, ...]


class SymbolicSkillVerifier:
    """Validates synthesized skills against genome gates, world state, and safety caps."""

    def verify(
        self,
        skill: BehaviorSkill,
        genome: TraitGenome,
        world: RoomWorld,
        current_tick: int = 0,
        last_executed_tick: int | None = None,
    ) -> VerificationResult:
        errors: list[str] = []
        used_anomalies: list[str] = []

        if skill.total_duration > MAX_SKILL_DURATION_TICKS:
            errors.append(
                f"Total duration {skill.total_duration} exceeds cap {MAX_SKILL_DURATION_TICKS}."
            )

        if last_executed_tick is not None:
            elapsed = current_tick - last_executed_tick
            if elapsed < skill.cooldown_ticks:
                errors.append(
                    f"Skill '{skill.name}' on cooldown ({elapsed}/{skill.cooldown_ticks} ticks)."
                )

        gravity_inverted = False

        for idx, step in enumerate(skill.steps):
            prim = step.primitive

            # 1. Enforce strict genome thresholds on all surreal anomaly primitives
            if prim in ANOMALY_GATES:
                trait_name, threshold = ANOMALY_GATES[prim]
                trait_val = float(getattr(genome, trait_name))
                used_anomalies.append(prim)
                if trait_val < threshold:
                    errors.append(
                        f"Step {idx} uses anomaly '{prim}' requiring "
                        f"{trait_name} >= {threshold:.2f} (genome has {trait_val:.2f})."
                    )
                elif prim == "invert_gravity":
                    gravity_inverted = True

            # 2. Enforce target existence & physical preconditions
            if prim in _TARGET_REQUIRED_PRIMITIVES and not step.target_object:
                errors.append(f"Step {idx} ('{prim}') requires a non-null target_object.")

            if step.target_object is not None:
                if step.target_object not in world.objects:
                    errors.append(
                        f"Step {idx} references unknown object '{step.target_object}'."
                    )
                else:
                    obj = world.objects[step.target_object]
                    if prim == "nudge" and not obj.is_intact:
                        errors.append(
                            f"Step {idx} cannot nudge broken object '{step.target_object}'."
                        )
                    if (
                        prim in {"approach", "nudge"}
                        and obj.elevation >= 1.5
                        and genome.agility < 0.65
                        and not gravity_inverted
                    ):
                        errors.append(
                            f"Step {idx} targets elevated '{step.target_object}' requiring "
                            f"agility >= 0.65 or active invert_gravity "
                            f"(genome agility={genome.agility:.2f})."
                        )

            # 3. Enforce payload requirements for speech mimicry
            if prim == "mimic_text":
                text = (step.payload or "").strip()
                if len(text) < 2:
                    errors.append(f"Step {idx} ('mimic_text') requires non-empty payload text.")

        unique_anomalies = tuple(dict.fromkeys(used_anomalies))
        return VerificationResult(
            is_valid=len(errors) == 0,
            errors=tuple(errors),
            uses_anomaly=len(unique_anomalies) > 0,
            required_anomalies=unique_anomalies,
        )