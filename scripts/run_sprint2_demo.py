from __future__ import annotations

from sim_core.genome import TraitGenome
from sim_core.rollout import run_aif_rollout


def main() -> None:
    presets: dict[str, TraitGenome] = {
        "Hairless Sphynx": TraitGenome.preset_hairless_sphynx(),
        "Staring Observer": TraitGenome.preset_staring_observer(),
        "Chaotic Gremlin": TraitGenome.preset_chaotic_gremlin(),
        "Shy Hermit": TraitGenome(
            fur_density=0.80,
            neuroticism=0.95,
            extraversion=0.05,
            attachment=0.05,
            curiosity=0.15,
            laziness=0.85,
            chaos=0.05,
            impulsiveness=0.15,
        ),
        "Impossible Cat": TraitGenome.preset_impossible_cat(),
    }

    print("=== Sprint 2: Active Inference Emergent Personality Profiles (250 ticks) ===\n")
    for label, genome in presets.items():
        res = run_aif_rollout(genome=genome, ticks=250, seed=42)
        top_actions = sorted(res.action_ratios.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_str = ", ".join(f"{name}={pct * 100:4.1f}%" for name, pct in top_actions)

        print(f"[{label}]")
        print(f"  Top policies : {top_str}")
        print(
            f"  Metrics      : heat={res.heat_seeking_ratio * 100:4.1f}% | "
            f"disruption={res.total_disruption:5.1f} | "
            f"entropy={res.action_entropy:.2f} bits | "
            f"mean_VFE={res.mean_vfe:.2f} | "
            f"glass_intact={res.water_glass_intact}\n"
        )


if __name__ == "__main__":
    main()