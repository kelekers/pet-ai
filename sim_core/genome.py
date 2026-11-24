from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field

GENOME_FIELDS: tuple[str, ...] = (
    # Temperament (Feline Five + simulation axes) - 9 dims
    "neuroticism",
    "extraversion",
    "dominance",
    "impulsiveness",
    "agreeableness",
    "curiosity",
    "laziness",
    "attachment",
    "chaos",
    # Physical parameters (coupled to simulation physics) - 5 dims
    "fur_density",
    "body_mass",
    "agility",
    "metabolism",
    "vocal_pitch",
    # Gated anomaly parameters (unlocked in Sprint 5 when > 0.8) - 4 dims
    "gravity_defiance",
    "spatial_phase",
    "mimicry_drive",
    "sentience_glimmer",
)


class TraitGenome(BaseModel):
    """18-dimensional continuous genetic representation of a single creature."""

    DIM: ClassVar[int] = len(GENOME_FIELDS)

    # Temperament
    neuroticism: float = Field(0.5, ge=0.0, le=1.0)
    extraversion: float = Field(0.5, ge=0.0, le=1.0)
    dominance: float = Field(0.5, ge=0.0, le=1.0)
    impulsiveness: float = Field(0.5, ge=0.0, le=1.0)
    agreeableness: float = Field(0.5, ge=0.0, le=1.0)
    curiosity: float = Field(0.5, ge=0.0, le=1.0)
    laziness: float = Field(0.5, ge=0.0, le=1.0)
    attachment: float = Field(0.5, ge=0.0, le=1.0)
    chaos: float = Field(0.5, ge=0.0, le=1.0)

    # Physical
    fur_density: float = Field(0.5, ge=0.0, le=1.0)
    body_mass: float = Field(0.5, ge=0.0, le=1.0)
    agility: float = Field(0.5, ge=0.0, le=1.0)
    metabolism: float = Field(0.5, ge=0.0, le=1.0)
    vocal_pitch: float = Field(0.5, ge=0.0, le=1.0)

    # Anomaly (0.0 = grounded realism; >0.8 = surreal capabilities)
    gravity_defiance: float = Field(0.0, ge=0.0, le=1.0)
    spatial_phase: float = Field(0.0, ge=0.0, le=1.0)
    mimicry_drive: float = Field(0.0, ge=0.0, le=1.0)
    sentience_glimmer: float = Field(0.0, ge=0.0, le=1.0)

    def to_vector(self) -> np.ndarray:
        """Convert genome to a flat float32 vector for pyribs / JAX."""
        return np.array([getattr(self, f) for f in GENOME_FIELDS], dtype=np.float32)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> TraitGenome:
        """Reconstruct a validated TraitGenome from an 18D vector."""
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.shape[0] != cls.DIM:
            raise ValueError(f"Expected {cls.DIM}D genome vector, got {arr.shape[0]}D.")
        clipped = np.clip(arr, 0.0, 1.0)
        return cls(**{field: float(val) for field, val in zip(GENOME_FIELDS, clipped, strict=True)})

    @classmethod
    def random(cls, rng: np.random.Generator, anomaly_chance: float = 0.08) -> TraitGenome:
        """Sample a grounded creature, with a small chance of surreal anomaly spikes."""
        base = rng.uniform(0.0, 1.0, size=14)
        anomalies = np.zeros(4, dtype=np.float32)
        if rng.random() < anomaly_chance:
            # Activate 1 or 2 surreal axes above the 0.8 threshold
            n_active = int(rng.integers(1, 3))
            chosen = rng.choice(4, size=n_active, replace=False)
            anomalies[chosen] = rng.uniform(0.82, 1.0, size=n_active)
        else:
            anomalies = rng.uniform(0.0, 0.15, size=4)
        return cls.from_vector(np.concatenate([base, anomalies]))

    @property
    def max_anomaly(self) -> float:
        return max(
            self.gravity_defiance,
            self.spatial_phase,
            self.mimicry_drive,
            self.sentience_glimmer,
        )

    # Diagnostic presets for testing emergence without hardcoding behavior rules
    @classmethod
    def preset_hairless_sphynx(cls) -> TraitGenome:
        return cls(
            fur_density=0.0,
            metabolism=0.75,
            attachment=0.85,
            agreeableness=0.8,
            curiosity=0.7,
            laziness=0.4,
        )

    @classmethod
    def preset_staring_observer(cls) -> TraitGenome:
        return cls(
            fur_density=0.9,
            curiosity=0.95,
            laziness=0.92,
            extraversion=0.15,
            attachment=0.25,
            sentience_glimmer=0.65,
        )

    @classmethod
    def preset_chaotic_gremlin(cls) -> TraitGenome:
        return cls(
            fur_density=0.45,
            impulsiveness=0.95,
            chaos=0.95,
            agility=0.9,
            laziness=0.1,
            dominance=0.85,
        )

    @classmethod
    def preset_impossible_cat(cls) -> TraitGenome:
        return cls(
            fur_density=0.3,
            curiosity=0.9,
            chaos=0.85,
            gravity_defiance=0.95,
            spatial_phase=0.9,
            sentience_glimmer=0.92,
        )