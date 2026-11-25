from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Affordance:
    """An interaction broadcast by a Smart Object."""

    name: str
    energy_cost: float
    drive_deltas: dict[str, float]
    disruption: float = 0.0
    min_agility: float = 0.0
    breaks_object: bool = False
    requires_proximity: bool = True


@dataclass
class SmartObject:
    """Physical object in the room broadcasting state and affordances."""

    obj_id: str
    name: str
    x: float
    y: float
    elevation: float = 0.0
    surface_temp_c: float = 21.0
    heat_radius_m: float = 0.0
    fragility: float = 0.0
    is_intact: bool = True
    is_user: bool = False
    affordances: list[Affordance] = field(default_factory=list)

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)

    def available_affordances(self, agility: float) -> list[Affordance]:
        if not self.is_intact:
            return []
        return [a for a in self.affordances if agility >= a.min_agility]


class RoomWorld:
    """Deterministic 2D continuous room containing Smart Objects and thermal fields."""

    def __init__(
        self,
        width: float = 6.0,
        height: float = 6.0,
        ambient_temp_c: float = 21.0,
        ambient_noise: float = 0.1,
        objects: dict[str, SmartObject] | None = None,
        seed: int = 42,
    ) -> None:
        self.width = width
        self.height = height
        self.ambient_temp_c = ambient_temp_c
        self.ambient_noise = ambient_noise
        self.objects: dict[str, SmartObject] = objects or {}
        self.rng = np.random.default_rng(seed)
        self.tick: int = 0

    @classmethod
    def create_default(cls, seed: int = 42, ambient_temp_c: float = 21.0) -> RoomWorld:
        """Build the standard 6m x 6m research room with 9 Smart Objects."""
        objs = [
            SmartObject(
                obj_id="radiator",
                name="Wall Radiator",
                x=0.4,
                y=3.0,
                surface_temp_c=42.0,
                heat_radius_m=1.6,
                affordances=[
                    Affordance(
                        name="curl_for_warmth",
                        energy_cost=0.005,
                        drive_deltas={"warmth": 0.14, "energy": 0.04, "comfort": 0.08},
                    )
                ],
            ),
            SmartObject(
                obj_id="laptop",
                name="Warm Laptop",
                x=3.0,
                y=4.8,
                elevation=0.75,
                surface_temp_c=38.5,
                heat_radius_m=0.8,
                fragility=0.3,
                affordances=[
                    Affordance(
                        name="sit_on_keyboard",
                        energy_cost=0.01,
                        drive_deltas={"warmth": 0.11, "comfort": 0.06, "stimulation": 0.05},
                        disruption=0.6,
                        min_agility=0.25,
                    )
                ],
            ),
            SmartObject(
                obj_id="sunbeam",
                name="Window Sunbeam",
                x=4.8,
                y=1.2,
                surface_temp_c=31.0,
                heat_radius_m=1.2,
                affordances=[
                    Affordance(
                        name="sunbathe",
                        energy_cost=0.002,
                        drive_deltas={"warmth": 0.08, "energy": 0.06, "comfort": 0.10},
                    )
                ],
            ),
            SmartObject(
                obj_id="water_glass",
                name="Glass of Water",
                x=3.4,
                y=4.9,
                elevation=0.75,
                fragility=0.95,
                affordances=[
                    Affordance(
                        name="knock_over",
                        energy_cost=0.03,
                        drive_deltas={"stimulation": 0.22},
                        disruption=1.0,
                        min_agility=0.25,
                        breaks_object=True,
                    ),
                    Affordance(
                        name="sip_water",
                        energy_cost=0.005,
                        drive_deltas={"hunger": -0.05, "comfort": 0.02},
                        min_agility=0.25,
                    ),
                ],
            ),
            SmartObject(
                obj_id="food_bowl",
                name="Food Bowl",
                x=1.0,
                y=0.8,
                affordances=[
                    Affordance(
                        name="eat",
                        energy_cost=0.01,
                        drive_deltas={"hunger": -0.35, "energy": 0.05, "comfort": 0.04},
                    )
                ],
            ),
            SmartObject(
                obj_id="cardboard_box",
                name="Cardboard Box",
                x=2.2,
                y=1.5,
                surface_temp_c=24.5,
                heat_radius_m=0.5,
                affordances=[
                    Affordance(
                        name="hide_and_nap",
                        energy_cost=0.002,
                        drive_deltas={"energy": 0.12, "comfort": 0.14, "warmth": 0.04},
                    )
                ],
            ),
            SmartObject(
                obj_id="high_shelf",
                name="High Bookshelf",
                x=5.4,
                y=5.2,
                elevation=1.8,
                affordances=[
                    Affordance(
                        name="perch_and_survey",
                        energy_cost=0.04,
                        drive_deltas={"stimulation": 0.12, "comfort": 0.08},
                        min_agility=0.65,
                    )
                ],
            ),
            SmartObject(
                obj_id="scratching_post",
                name="Scratching Post",
                x=4.5,
                y=3.2,
                affordances=[
                    Affordance(
                        name="scratch_vigorously",
                        energy_cost=0.045,
                        drive_deltas={"stimulation": 0.18, "comfort": 0.05},
                        disruption=0.15,
                    )
                ],
            ),
            SmartObject(
                obj_id="user",
                name="Human User",
                x=3.0,
                y=4.3,
                surface_temp_c=34.0,
                heat_radius_m=0.7,
                is_user=True,
                affordances=[
                    Affordance(
                        name="stare_at_user",
                        energy_cost=0.001,
                        drive_deltas={"stimulation": 0.07},
                        requires_proximity=False,
                    ),
                    Affordance(
                        name="rub_against_user",
                        energy_cost=0.015,
                        drive_deltas={"comfort": 0.12, "warmth": 0.05, "stimulation": 0.06},
                    ),
                ],
            ),
        ]
        return cls(
            ambient_temp_c=ambient_temp_c,
            objects={o.obj_id: o for o in objs},
            seed=seed,
        )

    def local_temperature(self, x: float, y: float) -> float:
        """Compute ambient temperature plus radial contributions from warm objects."""
        temp = self.ambient_temp_c
        for obj in self.objects.values():
            if obj.heat_radius_m <= 0.0 or obj.surface_temp_c <= self.ambient_temp_c:
                continue
            dist = obj.distance_to(x, y)
            if dist < obj.heat_radius_m:
                # Smooth quadratic falloff from object surface to ambient edge
                ratio = 1.0 - (dist / obj.heat_radius_m)
                temp += (obj.surface_temp_c - self.ambient_temp_c) * (ratio**2)
        return float(temp)

    def clamp_position(self, x: float, y: float) -> tuple[float, float]:
        return (
            float(np.clip(x, 0.0, self.width)),
            float(np.clip(y, 0.0, self.height)),
        )

    def reset_objects(self) -> None:
        """Restore all Smart Objects to intact state and calm ambient noise."""
        for obj in self.objects.values():
            obj.is_intact = True
        self.ambient_noise = 0.1

    def step(self) -> None:
        """Advance deterministic environment cycles (e.g., drifting sunbeam warmth)."""
        self.tick += 1
        sunbeam = self.objects.get("sunbeam")
        if sunbeam is not None:
            # Slow sinusoidal daylight warmth shift between 25C and 33C
            sunbeam.surface_temp_c = 29.0 + 4.0 * math.sin(self.tick * 0.015)
        # Subtle deterministic ambient noise jitter; smoothly decay loud event spikes
        base_noise = float(np.clip(0.1 + self.rng.normal(0.0, 0.02), 0.0, 1.0))
        if self.ambient_noise > 0.16:
            self.ambient_noise = float(
                np.clip(0.65 * self.ambient_noise + 0.35 * base_noise, 0.0, 1.0)
            )
        else:
            self.ambient_noise = base_noise