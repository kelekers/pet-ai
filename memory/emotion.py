from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim_core.genome import TraitGenome


def _clamp_pad(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


@dataclass(frozen=True)
class PADState:
    """Continuous Pleasure-Arousal-Dominance emotional coordinates in [-1, 1]."""

    pleasure: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    def mood_label(self) -> str:
        """Map PAD octant to a concise descriptive mood label."""
        if self.pleasure >= 0.15 and self.arousal >= 0.15:
            return "playful_excited" if self.dominance >= 0.0 else "eager_curious"
        if self.pleasure >= 0.15 and self.arousal < 0.15:
            return "content_relaxed" if self.dominance >= 0.0 else "docile_calm"
        if self.pleasure <= -0.15 and self.arousal >= 0.15:
            return "agitated_defensive" if self.dominance >= 0.0 else "anxious_fearful"
        if self.pleasure <= -0.15 and self.arousal < 0.15:
            return "aloof_bored" if self.dominance >= 0.0 else "withdrawn_stressed"
        return "neutral_alert"

    @property
    def fear_intensity(self) -> float:
        """Normalized [0, 1] fear/anxiety drive (negative P, high A, low D)."""
        raw = (-0.45 * self.pleasure + 0.30 * self.arousal - 0.25 * self.dominance)
        return float(np.clip(raw, 0.0, 1.0))

    def as_dict(self) -> dict[str, float | str]:
        return {
            "pleasure": round(self.pleasure, 4),
            "arousal": round(self.arousal, 4),
            "dominance": round(self.dominance, 4),
            "mood": self.mood_label(),
        }


class EmotionEngine:
    """Updates PAD state from VFE prediction errors, homeostasis, and sensory events."""

    def __init__(self, genome: TraitGenome) -> None:
        self.genome = genome
        self.baseline = PADState(
            pleasure=_clamp_pad(
                0.35 * (genome.agreeableness - genome.neuroticism)
                + 0.15 * (genome.extraversion - 0.5)
            ),
            arousal=_clamp_pad(
                0.35 * (genome.neuroticism + genome.impulsiveness - genome.laziness - 0.5)
            ),
            dominance=_clamp_pad(0.45 * (genome.dominance - genome.neuroticism)),
        )
        self.state = self.baseline
        self._vfe_ema: float | None = None
        # High-neuroticism pets recover more slowly from emotional perturbations
        self.decay_rate = float(0.055 - 0.032 * genome.neuroticism)

    def step(self, vfe: float, comfort: float, ambient_noise: float) -> PADState:
        """Advance continuous emotional state by one tick."""
        if self._vfe_ema is None:
            self._vfe_ema = vfe
        vfe_surprise = float(np.clip(vfe - self._vfe_ema, -3.0, 3.0))
        self._vfe_ema = 0.90 * self._vfe_ema + 0.10 * vfe

        sensitivity = 0.65 + 0.70 * self.genome.neuroticism
        comfort_signal = (comfort - 0.60) * 0.06
        noise_stress = max(0.0, ambient_noise - 0.15) * 0.25 * sensitivity

        # Positive surprise (unexpected high VFE) spikes arousal and lowers pleasure/dominance
        surprise_p = -0.04 * max(0.0, vfe_surprise) * sensitivity
        surprise_a = 0.06 * abs(vfe_surprise) * sensitivity
        surprise_d = -0.03 * max(0.0, vfe_surprise) * sensitivity

        p = self.state.pleasure + self.decay_rate * (self.baseline.pleasure - self.state.pleasure)
        a = self.state.arousal + self.decay_rate * (self.baseline.arousal - self.state.arousal)
        d = self.state.dominance + self.decay_rate * (self.baseline.dominance - self.state.dominance)

        self.state = PADState(
            pleasure=_clamp_pad(p + comfort_signal + surprise_p - noise_stress),
            arousal=_clamp_pad(a + surprise_a + 1.2 * noise_stress),
            dominance=_clamp_pad(d + surprise_d - 0.7 * noise_stress),
        )
        return self.state

    def apply_stimulus(
        self,
        delta_p: float,
        delta_a: float,
        delta_d: float,
    ) -> tuple[PADState, tuple[float, float, float]]:
        """Apply a discrete external event scaled by temperament; return new state & actual delta."""
        neg_scale = 0.75 + 0.65 * self.genome.neuroticism
        pos_scale = 0.75 + 0.55 * (0.5 * self.genome.attachment + 0.5 * self.genome.agreeableness)

        eff_p = delta_p * (neg_scale if delta_p < 0.0 else pos_scale)
        eff_a = delta_a * (neg_scale if delta_a > 0.0 else 1.0)
        eff_d = delta_d * (neg_scale if delta_d < 0.0 else 1.0)

        new_state = PADState(
            pleasure=_clamp_pad(self.state.pleasure + eff_p),
            arousal=_clamp_pad(self.state.arousal + eff_a),
            dominance=_clamp_pad(self.state.dominance + eff_d),
        )
        actual_delta = (
            new_state.pleasure - self.state.pleasure,
            new_state.arousal - self.state.arousal,
            new_state.dominance - self.state.dominance,
        )
        self.state = new_state
        return self.state, actual_delta