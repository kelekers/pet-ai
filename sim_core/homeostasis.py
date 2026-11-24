from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim_core.genome import TraitGenome

THERMONEUTRAL_TEMP_C = 32.0


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass
class HomeostaticState:
    """Internal physiological state of the creature (drives normalized to [0, 1])."""

    warmth: float = 0.85  # 1.0 = thermally comfortable, 0.0 = freezing
    hunger: float = 0.20  # 0.0 = full, 1.0 = starving
    energy: float = 0.85  # 1.0 = rested, 0.0 = exhausted
    stimulation: float = 0.65  # 1.0 = engaged, 0.0 = bored
    comfort: float = 0.75  # 1.0 = calm/safe, 0.0 = distressed

    @property
    def core_temp_c(self) -> float:
        """Map warmth reserve onto realistic feline core temp range (36.5C - 38.7C)."""
        return 36.5 + 2.2 * self.warmth

    def as_dict(self) -> dict[str, float]:
        return {
            "warmth": round(self.warmth, 4),
            "core_temp_c": round(self.core_temp_c, 3),
            "hunger": round(self.hunger, 4),
            "energy": round(self.energy, 4),
            "stimulation": round(self.stimulation, 4),
            "comfort": round(self.comfort, 4),
        }


class HomeostasisEngine:
    """Updates biological drives based on genome physics and local environment."""

    @staticmethod
    def thermal_conductance(fur_density: float) -> float:
        """Higher fur density insulates against cold: 0.0 -> 1.00, 0.9 -> 0.442 (2.26x ratio)."""
        return 1.0 - 0.62 * fur_density

    @classmethod
    def step(
        cls,
        state: HomeostaticState,
        genome: TraitGenome,
        local_temp_c: float,
        ambient_noise: float,
        activity_level: float = 0.2,
        dt: float = 1.0,
    ) -> HomeostaticState:
        """Advance homeostatic drives by one simulation tick."""
        conductance = cls.thermal_conductance(genome.fur_density)

        # 1. Thermal dynamics coupled directly to fur_density
        if local_temp_c < THERMONEUTRAL_TEMP_C:
            temp_deficit = THERMONEUTRAL_TEMP_C - local_temp_c
            raw_loss = 0.00045 * temp_deficit * conductance
            metabolic_heat = 0.00025 * genome.metabolism
            delta_warmth = -max(0.0, raw_loss - metabolic_heat) * dt
        else:
            temp_excess = local_temp_c - THERMONEUTRAL_TEMP_C
            # Bare skin absorbs radiant heat slightly faster too
            delta_warmth = 0.0022 * temp_excess * (0.5 + 0.5 * conductance) * dt

        warmth = _clamp01(state.warmth + delta_warmth)

        # 2. Hunger rises faster with higher metabolism and physical exertion
        delta_hunger = (0.0007 + 0.0011 * genome.metabolism + 0.0008 * activity_level) * dt
        hunger = _clamp01(state.hunger + delta_hunger)

        # 3. Energy drains with activity and body mass, recovers slightly when idle
        if activity_level <= 0.1:
            delta_energy = 0.0015 * dt
        else:
            delta_energy = -(0.0006 + 0.0010 * genome.body_mass) * activity_level * dt
        energy = _clamp01(state.energy + delta_energy)

        # 4. Stimulation decays faster in high-curiosity pets
        delta_stim = -(0.0006 + 0.0014 * genome.curiosity) * dt
        stimulation = _clamp01(state.stimulation + delta_stim)

        # 5. Comfort reflects thermal/hunger stress plus noise sensitivity (neuroticism)
        stress = (
            0.35 * (1.0 - warmth)
            + 0.30 * hunger
            + 0.35 * ambient_noise * (0.5 + genome.neuroticism)
        )
        target_comfort = _clamp01(1.0 - stress)
        comfort = _clamp01(state.comfort + 0.08 * (target_comfort - state.comfort) * dt)

        return HomeostaticState(
            warmth=warmth,
            hunger=hunger,
            energy=energy,
            stimulation=stimulation,
            comfort=comfort,
        )

    @staticmethod
    def apply_affordance(
        state: HomeostaticState,
        energy_cost: float,
        drive_deltas: dict[str, float],
    ) -> HomeostaticState:
        """Apply immediate physiological effects from interacting with a Smart Object."""
        return HomeostaticState(
            warmth=_clamp01(state.warmth + drive_deltas.get("warmth", 0.0)),
            hunger=_clamp01(state.hunger + drive_deltas.get("hunger", 0.0)),
            energy=_clamp01(state.energy - energy_cost + drive_deltas.get("energy", 0.0)),
            stimulation=_clamp01(state.stimulation + drive_deltas.get("stimulation", 0.0)),
            comfort=_clamp01(state.comfort + drive_deltas.get("comfort", 0.0)),
        )