from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

GroundedPrimitive = Literal[
    "approach",
    "stare",
    "nudge",
    "vocalize",
    "groom",
    "crouch",
    "wait",
]

AnomalyPrimitive = Literal[
    "invert_gravity",
    "phase_object",
    "mimic_text",
    "telepathic_gaze",
]

PrimitiveName = Literal[
    "approach",
    "stare",
    "nudge",
    "vocalize",
    "groom",
    "crouch",
    "wait",
    "invert_gravity",
    "phase_object",
    "mimic_text",
    "telepathic_gaze",
]

GROUNDED_PRIMITIVES: frozenset[str] = frozenset(
    {"approach", "stare", "nudge", "vocalize", "groom", "crouch", "wait"}
)

# Maps each anomaly primitive -> (TraitGenome attribute, required minimum threshold)
ANOMALY_GATES: dict[str, tuple[str, float]] = {
    "invert_gravity": ("gravity_defiance", 0.80),
    "phase_object": ("spatial_phase", 0.80),
    "mimic_text": ("mimicry_drive", 0.80),
    "telepathic_gaze": ("sentience_glimmer", 0.80),
}

_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]{2,47}$")


class SkillStep(BaseModel):
    """Single atomic or surreal primitive inside a multi-step ritual."""

    model_config = ConfigDict(extra="forbid")

    primitive: PrimitiveName
    target_object: str | None = Field(
        default=None,
        description="RoomWorld obj_id (e.g. 'user', 'laptop', 'high_shelf') or null.",
    )
    duration_ticks: int = Field(default=2, ge=1, le=12)
    intensity: float = Field(default=0.7, ge=0.0, le=1.0)
    payload: str | None = Field(
        default=None,
        max_length=64,
        description="Short vocalization or mimicked utterance text.",
    )


class BehaviorSkill(BaseModel):
    """Structured neuro-symbolic ritual synthesized by the slow LLM Director."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Unique snake_case skill identifier.")
    trigger_context: str = Field(
        ...,
        min_length=6,
        max_length=180,
        description="Situation or user cue that triggers this ritual.",
    )
    min_vfe: float = Field(default=0.0, ge=0.0, le=20.0)
    cooldown_ticks: int = Field(default=25, ge=10, le=500)
    steps: list[SkillStep] = Field(..., min_length=2, max_length=6)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        if not _SNAKE_CASE_RE.match(normalized):
            raise ValueError(
                "Skill name must be snake_case (3-48 lowercase letters, digits, underscores)."
            )
        return normalized

    @property
    def total_duration(self) -> int:
        return sum(step.duration_ticks for step in self.steps)

    @property
    def anomaly_primitives(self) -> tuple[str, ...]:
        found = [s.primitive for s in self.steps if s.primitive in ANOMALY_GATES]
        return tuple(dict.fromkeys(found))