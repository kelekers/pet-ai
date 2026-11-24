import numpy as np

from sim_core.genome import TraitGenome
from sim_core.simulation import HeadlessSimulation
from sim_core.world import RoomWorld


def test_genome_vector_roundtrip() -> None:
    rng = np.random.default_rng(123)
    original = TraitGenome.random(rng, anomaly_chance=1.0)
    vec = original.to_vector()
    assert vec.shape == (18,)
    reconstructed = TraitGenome.from_vector(vec)
    assert np.allclose(original.to_vector(), reconstructed.to_vector())
    assert reconstructed.max_anomaly >= 0.82


def test_hairless_loses_heat_twice_as_fast_as_fluffy() -> None:
    base = TraitGenome(metabolism=0.5)
    hairless = base.model_copy(update={"fur_density": 0.0})
    fluffy = base.model_copy(update={"fur_density": 0.9})

    sim_h = HeadlessSimulation(hairless, RoomWorld.create_default(seed=1), start_pos=(2.5, 2.5))
    sim_f = HeadlessSimulation(fluffy, RoomWorld.create_default(seed=1), start_pos=(2.5, 2.5))

    start_w = sim_h.pet.homeostasis.warmth
    for _ in range(100):
        rec_h = sim_h.step()
        rec_f = sim_f.step()

    loss_h = start_w - rec_h.warmth
    loss_f = start_w - rec_f.warmth
    assert loss_h / loss_f >= 2.0


def test_simulation_is_deterministic_over_1000_ticks() -> None:
    genome = TraitGenome.preset_hairless_sphynx()
    sim_a = HeadlessSimulation(genome, RoomWorld.create_default(seed=99))
    sim_b = HeadlessSimulation(genome, RoomWorld.create_default(seed=99))

    for tick in range(1000):
        target = ("radiator", "curl_for_warmth") if tick % 2 == 0 else ("food_bowl", "eat")
        rec_a = sim_a.step(*target)
        rec_b = sim_b.step(*target)

    assert rec_a == rec_b
    for val in (rec_a.warmth, rec_a.hunger, rec_a.energy, rec_a.stimulation, rec_a.comfort):
        assert 0.0 <= val <= 1.0


def test_water_glass_breaks_on_knock_over() -> None:
    genome = TraitGenome.preset_chaotic_gremlin()
    world = RoomWorld.create_default(seed=5)
    glass = world.objects["water_glass"]
    sim = HeadlessSimulation(genome, world, start_pos=(glass.x, glass.y))

    rec = sim.step("water_glass", "knock_over")
    assert rec.action == "water_glass:knock_over"
    assert rec.disruption_delta == 1.0
    assert not world.objects["water_glass"].is_intact