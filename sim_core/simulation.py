from __future__ import annotations

import math
from dataclasses import dataclass

from sim_core.genome import TraitGenome
from sim_core.homeostasis import HomeostasisEngine, HomeostaticState
from sim_core.world import Affordance, RoomWorld, SmartObject


@dataclass
class PetState:
    x: float = 3.0
    y: float = 3.0
    homeostasis: HomeostaticState = None  # type: ignore[assignment]
    last_action: str = "idle"
    total_disruption: float = 0.0

    def __post_init__(self) -> None:
        if self.homeostasis is None:
            self.homeostasis = HomeostaticState()


@dataclass(frozen=True)
class TickRecord:
    tick: int
    x: float
    y: float
    local_temp_c: float
    action: str
    warmth: float
    core_temp_c: float
    hunger: float
    energy: float
    stimulation: float
    comfort: float
    disruption_delta: float


class HeadlessSimulation:
    """Deterministic single-creature simulation loop with zero graphics or AI coupling."""

    def __init__(
        self,
        genome: TraitGenome,
        world: RoomWorld | None = None,
        start_pos: tuple[float, float] = (3.0, 3.0),
        seed: int = 42,
    ) -> None:
        self.genome = genome
        self.world = world or RoomWorld.create_default(seed=seed)
        self.pet = PetState(x=start_pos[0], y=start_pos[1])
        self.max_step_speed_m = 0.45 + 0.55 * genome.agility

    def _move_toward(self, target_x: float, target_y: float) -> float:
        """Move toward target up to max_step_speed_m; return distance moved."""
        dx = target_x - self.pet.x
        dy = target_y - self.pet.y
        dist = math.hypot(dx, dy)
        if dist <= 1e-6:
            return 0.0
        step_dist = min(dist, self.max_step_speed_m)
        nx, ny = self.world.clamp_position(
            self.pet.x + (dx / dist) * step_dist,
            self.pet.y + (dy / dist) * step_dist,
        )
        self.pet.x, self.pet.y = nx, ny
        return step_dist

    def _resolve_affordance(
        self,
        obj: SmartObject,
        affordance_name: str,
    ) -> Affordance | None:
        for aff in obj.available_affordances(self.genome.agility):
            if aff.name == affordance_name:
                return aff
        return None

    def step(
        self,
        target_object_id: str | None = None,
        affordance_name: str | None = None,
        action_label_override: str | None = None,
        activity_level_override: float | None = None,
    ) -> TickRecord:
        """Execute one deterministic simulation tick."""
        self.world.step()
        activity_level = 0.05
        action_label = "idle"
        disruption_delta = 0.0

        if target_object_id is not None and target_object_id in self.world.objects:
            obj = self.world.objects[target_object_id]
            aff = (
                self._resolve_affordance(obj, affordance_name)
                if affordance_name is not None
                else None
            )

            if aff is not None:
                dist = obj.distance_to(self.pet.x, self.pet.y)
                if aff.requires_proximity and dist > 0.35:
                    moved = self._move_toward(obj.x, obj.y)
                    activity_level = min(1.0, 0.25 + moved)
                    action_label = f"approach:{obj.obj_id}"
                else:
                    # Execute the Smart Object affordance
                    self.pet.homeostasis = HomeostasisEngine.apply_affordance(
                        self.pet.homeostasis,
                        energy_cost=aff.energy_cost,
                        drive_deltas=aff.drive_deltas,
                    )
                    activity_level = min(1.0, 0.1 + aff.energy_cost * 12.0)
                    action_label = f"{obj.obj_id}:{aff.name}"
                    disruption_delta = aff.disruption
                    self.pet.total_disruption += disruption_delta
                    if aff.breaks_object:
                        obj.is_intact = False

        if action_label_override is not None:
            action_label = action_label_override
        if activity_level_override is not None:
            activity_level = float(max(0.0, min(1.0, activity_level_override)))

        local_temp = self.world.local_temperature(self.pet.x, self.pet.y)
        self.pet.homeostasis = HomeostasisEngine.step(
            state=self.pet.homeostasis,
            genome=self.genome,
            local_temp_c=local_temp,
            ambient_noise=self.world.ambient_noise,
            activity_level=activity_level,
        )
        self.pet.last_action = action_label

        h = self.pet.homeostasis
        return TickRecord(
            tick=self.world.tick,
            x=round(self.pet.x, 3),
            y=round(self.pet.y, 3),
            local_temp_c=round(local_temp, 2),
            action=action_label,
            warmth=round(h.warmth, 4),
            core_temp_c=round(h.core_temp_c, 3),
            hunger=round(h.hunger, 4),
            energy=round(h.energy, 4),
            stimulation=round(h.stimulation, 4),
            comfort=round(h.comfort, 4),
            disruption_delta=disruption_delta,
        )