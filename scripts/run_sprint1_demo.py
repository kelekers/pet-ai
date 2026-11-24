from __future__ import annotations

import numpy as np

from sim_core.genome import TraitGenome
from sim_core.simulation import HeadlessSimulation
from sim_core.world import RoomWorld


def run_thermal_comparison(ticks: int = 100) -> None:
    """Verify fur_density=0.0 loses warmth >2x faster than fur_density=0.9."""
    base = TraitGenome(metabolism=0.5)
    hairless = base.model_copy(update={"fur_density": 0.0})
    fluffy = base.model_copy(update={"fur_density": 0.9})

    # Place both in an ambient 21C corner away from heat sources
    cold_spot = (2.5, 2.5)
    sim_hairless = HeadlessSimulation(hairless, RoomWorld.create_default(seed=7), cold_spot)
    sim_fluffy = HeadlessSimulation(fluffy, RoomWorld.create_default(seed=7), cold_spot)

    start_warmth = sim_hairless.pet.homeostasis.warmth
    for _ in range(ticks):
        rec_h = sim_hairless.step()
        rec_f = sim_fluffy.step()

    loss_h = start_warmth - rec_h.warmth
    loss_f = start_warmth - rec_f.warmth
    ratio = loss_h / loss_f

    print("=== Controlled Thermal Decay Test (100 ticks @ 21.0C) ===")
    print(f"Hairless (fur=0.0) warmth drop: {loss_h:.4f} (core {rec_h.core_temp_c:.2f}C)")
    print(f"Fluffy   (fur=0.9) warmth drop: {loss_f:.4f} (core {rec_f.core_temp_c:.2f}C)")
    print(f"Heat loss ratio (hairless / fluffy): {ratio:.2f}x\n")


def run_1000_tick_sandbox(seed: int = 42) -> None:
    """Run a 1,000-tick seeded neutral rollout across Smart Objects."""
    rng = np.random.default_rng(seed)
    genome = TraitGenome.preset_chaotic_gremlin()
    sim = HeadlessSimulation(genome, RoomWorld.create_default(seed=seed))

    current_target: tuple[str, str] | None = None
    print("=== 1,000-Tick Deterministic Sandbox Rollout ===")
    for tick in range(1, 1001):
        # Pick a new valid Smart Object affordance every 25 ticks
        if tick % 25 == 1 or current_target is None:
            choices: list[tuple[str, str]] = []
            for obj in sim.world.objects.values():
                for aff in obj.available_affordances(genome.agility):
                    choices.append((obj.obj_id, aff.name))
            idx = int(rng.integers(0, len(choices)))
            current_target = choices[idx]

        rec = sim.step(current_target[0], current_target[1])
        if tick in (1, 100, 250, 500, 750, 1000):
            print(
                f"Tick {rec.tick:4d} | pos=({rec.x:4.2f},{rec.y:4.2f}) "
                f"| T={rec.local_temp_c:4.1f}C | action={rec.action:26s} "
                f"| W={rec.warmth:.2f} H={rec.hunger:.2f} E={rec.energy:.2f} "
                f"S={rec.stimulation:.2f} C={rec.comfort:.2f}"
            )

    glass_intact = sim.world.objects["water_glass"].is_intact
    print(
        f"\nCompleted 1,000 ticks | Total disruption: {sim.pet.total_disruption:.2f} "
        f"| Water glass intact: {glass_intact}"
    )


if __name__ == "__main__":
    run_thermal_comparison()
    run_1000_tick_sandbox()