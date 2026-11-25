from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ObjectSnapshot(BaseModel):
    obj_id: str
    name: str
    x: float
    y: float
    elevation: float
    surface_temp_c: float
    is_intact: bool
    is_user: bool


class MorphologySnapshot(BaseModel):
    fur_density: float
    body_mass: float
    agility: float
    vocal_pitch: float
    max_anomaly: float


class PADSnapshot(BaseModel):
    pleasure: float
    arousal: float
    dominance: float
    mood: str
    fear_intensity: float


class HomeostasisSnapshot(BaseModel):
    warmth: float
    core_temp_c: float
    hunger: float
    energy: float
    stimulation: float
    comfort: float


class FrameState(BaseModel):
    """Compact 5-10 Hz state frame streamed from Python Brain to Godot 4 Body."""

    tick: int
    paused: bool
    preset_label: str
    x: float
    y: float
    elevation: float
    local_temp_c: float
    mode: str
    action: str
    target_object_id: str | None
    active_skill_name: str | None
    active_primitive: str | None
    synthesized_skill: str | None
    vfe: float
    user_trust: float
    user_fear: float
    user_belief_calm: float
    llm_status: str = "idle"
    llm_backend: str = "mock"
    tick_hz: float = 6.0
    policy_probs: dict[str, float] = Field(default_factory=dict)
    skill_use_counts: dict[str, int] = Field(default_factory=dict)
    morphology: MorphologySnapshot
    homeostasis: HomeostasisSnapshot
    pad: PADSnapshot
    anomaly_flags: dict[str, object]
    skill_library: list[str]
    world_objects: list[ObjectSnapshot]


class ClientCommand(BaseModel):
    """Inbound command sent from Godot 4 or Telemetry UI over WebSocket."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "spawn_preset",
        "spawn_archive_cell",
        "user_prompt",
        "sensory_event",
        "move_user",
        "set_paused",
        "step_once",
        "reset_room",
        "refill_water_glass",
        "set_speed",
    ]
    preset: str | None = None
    cell_index: int | None = None
    text: str | None = Field(default=None, max_length=240)
    event: Literal[
        "calm_petting",
        "gentle_feeding",
        "loud_clatter",
        "startling_shout",
    ] | None = None
    x: float | None = Field(default=None, ge=0.0, le=6.0)
    y: float | None = Field(default=None, ge=0.0, le=6.0)
    paused: bool | None = None
    tick_hz: float | None = Field(default=None, ge=0.5, le=30.0)