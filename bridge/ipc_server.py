from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import msgpack
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import ValidationError

from bridge.protocol import (
    ClientCommand,
    FrameState,
    HomeostasisSnapshot,
    MorphologySnapshot,
    ObjectSnapshot,
    PADSnapshot,
)
from memory.memory_store import HashEmbedder
from memory.prior_adaptation import SensoryEvent
from neuro_symbolic.llm_director import DeterministicMockAdapter, DirectorAdapter, OllamaAdapter
from neuro_symbolic.orchestrator import NeuroSymbolicPetAgent, NeuroSymbolicTick
from sim_core.genome import TraitGenome
from sim_core.genome_mapper import ACTION_NAMES
from sim_core.world import RoomWorld

DEFAULT_ARCHIVE_PATH = Path("data/archives/personality_archive.json")
DEFAULT_INSPECTOR_PATH = Path(__file__).resolve().parent.parent / "telemetry_ui" / "index.html"

PRESET_BUILDERS: dict[str, TraitGenome] = {
    "hairless_sphynx": TraitGenome.preset_hairless_sphynx(),
    "staring_observer": TraitGenome.preset_staring_observer(),
    "chaotic_gremlin": TraitGenome.preset_chaotic_gremlin(),
    "impossible_cat": TraitGenome.preset_impossible_cat().model_copy(
        update={"mimicry_drive": 0.90}
    ),
    "shy_hermit": TraitGenome(
        fur_density=0.80,
        neuroticism=0.95,
        extraversion=0.05,
        attachment=0.05,
        curiosity=0.15,
        laziness=0.85,
        chaos=0.05,
        impulsiveness=0.15,
    ),
}


class SimulationSession:
    """Stateful single-pet session exposed over WebSocket and REST."""

    def __init__(
        self,
        preset: str = "impossible_cat",
        archive_path: Path = DEFAULT_ARCHIVE_PATH,
        use_live_ollama: bool = False,
        ollama_model: str = "qwen3:1.7b",
        async_synthesis: bool = True,
        tick_hz: float = 6.0,
        seed: int = 42,
        adapter_override: DirectorAdapter | None = None,
    ) -> None:
        self.archive_path = archive_path
        self.use_live_ollama = use_live_ollama
        self.ollama_model = ollama_model
        self.async_synthesis = async_synthesis
        self.tick_hz = float(max(0.5, min(30.0, tick_hz)))
        self.seed = seed
        self.paused = False
        self.preset_label = preset
        self._adapter_override = adapter_override

        self._last_tick: NeuroSymbolicTick | None = None
        self._last_target_obj: str | None = None
        self._last_user_belief: float = 0.5
        self._last_policy_probs: dict[str, float] = {}
        self.agent = self._build_agent(PRESET_BUILDERS.get(preset, TraitGenome()))

    def _make_adapter(self) -> DirectorAdapter:
        if self._adapter_override is not None:
            return self._adapter_override
        if self.use_live_ollama:
            return OllamaAdapter(model=self.ollama_model)
        return DeterministicMockAdapter()

    def _build_agent(self, genome: TraitGenome) -> NeuroSymbolicPetAgent:
        return NeuroSymbolicPetAgent(
            genome=genome,
            world=RoomWorld.create_default(seed=self.seed, ambient_temp_c=20.0),
            adapter=self._make_adapter(),
            embedder=HashEmbedder(),
            seed=self.seed,
        )

    def reset_with_genome(self, genome: TraitGenome, label: str) -> None:
        self.agent.close()
        self.preset_label = label
        self._last_tick = None
        self._last_target_obj = None
        self._last_user_belief = 0.5
        self._last_policy_probs = {}
        self.agent = self._build_agent(genome)

    def load_archive(self) -> dict[str, Any]:
        if not self.archive_path.exists():
            return {"metadata": {"num_elites": 0}, "elites": []}
        return dict(json.loads(self.archive_path.read_text(encoding="utf-8")))

    def spawn_from_archive(self, cell_index: int) -> bool:
        archive = self.load_archive()
        for elite in archive.get("elites", []):
            if int(elite["cell_index"]) == cell_index:
                genome = TraitGenome(**elite["genome"])
                label = f"archive:{cell_index}:{elite.get('archetype', 'elite')}"
                self.reset_with_genome(genome, label=label)
                return True
        return False

    def apply_command(self, cmd: ClientCommand) -> FrameState:
        if cmd.type == "spawn_preset" and cmd.preset in PRESET_BUILDERS:
            self.reset_with_genome(PRESET_BUILDERS[cmd.preset], label=cmd.preset)
        elif cmd.type == "spawn_archive_cell" and cmd.cell_index is not None:
            self.spawn_from_archive(cmd.cell_index)
        elif cmd.type == "user_prompt" and cmd.text:
            self.agent.send_user_prompt(cmd.text)
        elif cmd.type == "sensory_event" and cmd.event is not None:
            event_map = {
                "calm_petting": SensoryEvent.calm_petting,
                "gentle_feeding": SensoryEvent.gentle_feeding,
                "loud_clatter": SensoryEvent.loud_kitchen_clatter,
                "startling_shout": SensoryEvent.startling_user_shout,
            }
            self.agent.adaptive.inject_event(event_map[cmd.event]())
        elif cmd.type == "move_user" and cmd.x is not None and cmd.y is not None:
            user = self.agent.world.objects["user"]
            user.x, user.y = self.agent.world.clamp_position(cmd.x, cmd.y)
        elif cmd.type == "set_paused" and cmd.paused is not None:
            self.paused = cmd.paused
        elif cmd.type == "reset_room":
            self.agent.reset_room()
        elif cmd.type == "refill_water_glass":
            glass = self.agent.world.objects.get("water_glass")
            if glass is not None:
                glass.is_intact = True
        elif cmd.type == "set_speed" and cmd.tick_hz is not None:
            self.tick_hz = float(max(0.5, min(30.0, cmd.tick_hz)))
        elif cmd.type == "step_once":
            return self.step()
        return self.snapshot()

    def _resolve_target_object(self, res: NeuroSymbolicTick) -> str | None:
        world_objs = self.agent.world.objects
        if res.target_object_id and res.target_object_id in world_objs:
            return res.target_object_id
        tele_target = res.anomaly_flags.get("telepathic_target")
        if tele_target and str(tele_target) in world_objs:
            return str(tele_target)
        action = res.tick_record.action
        if action.startswith("skill:"):
            parts = action.split(":")
            if len(parts) >= 3 and parts[2] in world_objs:
                return parts[2]
        elif ":" in action:
            prefix, suffix = action.split(":", 1)
            cand = suffix if prefix == "approach" else prefix
            if cand in world_objs:
                return cand
        return "user"

    def step(self) -> FrameState:
        res = self.agent.step(async_synthesis=self.async_synthesis)
        self._last_tick = res
        self._last_user_belief = float(res.user_belief_calm)
        if res.policy_probs and len(res.policy_probs) == len(ACTION_NAMES):
            self._last_policy_probs = {
                name: round(float(prob), 4)
                for name, prob in zip(ACTION_NAMES, res.policy_probs, strict=True)
            }
        self._last_target_obj = self._resolve_target_object(res)
        return self.snapshot()

    def snapshot(self) -> FrameState:
        sim = self.agent.adaptive.sim
        world = self.agent.world
        g = self.agent.genome
        h = sim.pet.homeostasis
        pad = self.agent.adaptive.emotion.state
        tick = world.tick

        flags = self.agent.anomaly_state.active_flags(tick)
        elevation = 2.15 if flags.get("gravity_inverted") else 0.0
        if not elevation and self._last_target_obj in world.objects:
            target = world.objects[self._last_target_obj]
            if target.elevation > 0.0 and target.distance_to(sim.pet.x, sim.pet.y) <= 0.45:
                elevation = target.elevation

        user_trust, user_fear = self.agent.adaptive.memory.entity_affinity("user", tick)
        last = self._last_tick

        return FrameState(
            tick=tick,
            paused=self.paused,
            preset_label=self.preset_label,
            x=round(sim.pet.x, 3),
            y=round(sim.pet.y, 3),
            elevation=round(elevation, 3),
            local_temp_c=round(world.local_temperature(sim.pet.x, sim.pet.y), 2),
            mode=last.mode if last is not None else "active_inference",
            action=sim.pet.last_action,
            target_object_id=self._last_target_obj,
            active_skill_name=last.active_skill_name if last is not None else None,
            active_primitive=last.active_primitive if last is not None else None,
            synthesized_skill=last.synthesized_skill if last is not None else None,
            vfe=round(last.vfe, 4) if last is not None else 0.0,
            user_trust=round(user_trust, 4),
            user_fear=round(user_fear, 4),
            user_belief_calm=round(self._last_user_belief, 4),
            llm_status=self.agent.llm_status,
            llm_backend=(
                "custom"
                if self._adapter_override is not None
                else ("ollama" if self.use_live_ollama else "mock")
            ),
            tick_hz=round(self.tick_hz, 2),
            policy_probs=dict(self._last_policy_probs),
            skill_use_counts=self.agent.library.use_counts(),
            morphology=MorphologySnapshot(
                fur_density=g.fur_density,
                body_mass=g.body_mass,
                agility=g.agility,
                vocal_pitch=g.vocal_pitch,
                max_anomaly=g.max_anomaly,
            ),
            homeostasis=HomeostasisSnapshot(
                warmth=round(h.warmth, 4),
                core_temp_c=round(h.core_temp_c, 3),
                hunger=round(h.hunger, 4),
                energy=round(h.energy, 4),
                stimulation=round(h.stimulation, 4),
                comfort=round(h.comfort, 4),
            ),
            pad=PADSnapshot(
                pleasure=round(pad.pleasure, 4),
                arousal=round(pad.arousal, 4),
                dominance=round(pad.dominance, 4),
                mood=pad.mood_label(),
                fear_intensity=round(pad.fear_intensity, 4),
            ),
            anomaly_flags=flags,
            skill_library=self.agent.library.list_names(),
            world_objects=[
                ObjectSnapshot(
                    obj_id=obj.obj_id,
                    name=obj.name,
                    x=round(obj.x, 3),
                    y=round(obj.y, 3),
                    elevation=round(obj.elevation, 3),
                    surface_temp_c=round(obj.surface_temp_c, 2),
                    is_intact=obj.is_intact,
                    is_user=obj.is_user,
                )
                for obj in world.objects.values()
            ],
        )

    def close(self) -> None:
        self.agent.close()


async def _send_frame(websocket: WebSocket, frame: FrameState, use_msgpack: bool) -> None:
    if use_msgpack:
        await websocket.send_bytes(msgpack.packb(frame.model_dump(), use_bin_type=True))
    else:
        await websocket.send_text(frame.model_dump_json())


def create_app(
    preset: str = "impossible_cat",
    tick_hz: float = 6.0,
    auto_tick: bool = True,
    use_live_ollama: bool = False,
    ollama_model: str = "qwen3:1.7b",
    async_synthesis: bool = True,
    archive_path: Path = DEFAULT_ARCHIVE_PATH,
    adapter_override: DirectorAdapter | None = None,
) -> FastAPI:
    """Create the FastAPI Brain-Body WebSocket bridge application."""
    session = SimulationSession(
        preset=preset,
        archive_path=archive_path,
        use_live_ollama=use_live_ollama,
        ollama_model=ollama_model,
        async_synthesis=async_synthesis,
        tick_hz=tick_hz,
        adapter_override=adapter_override,
    )
    lock = asyncio.Lock()
    broadcast_lock = asyncio.Lock()
    clients: dict[WebSocket, bool] = {}
    speed_changed = asyncio.Event()

    async def broadcast_frame(frame: FrameState) -> None:
        async with broadcast_lock:
            if not clients:
                return
            json_payload: str | None = None
            msgpack_payload: bytes | None = None
            stale: list[WebSocket] = []
            for ws, use_msgpack in list(clients.items()):
                try:
                    if use_msgpack:
                        if msgpack_payload is None:
                            msgpack_payload = msgpack.packb(
                                frame.model_dump(),
                                use_bin_type=True,
                            )
                        await ws.send_bytes(msgpack_payload)
                    else:
                        if json_payload is None:
                            json_payload = frame.model_dump_json()
                        await ws.send_text(json_payload)
                except (WebSocketDisconnect, RuntimeError, OSError):
                    stale.append(ws)
            for ws in stale:
                clients.pop(ws, None)

    async def single_ticker_loop() -> None:
        while True:
            interval = 1.0 / max(0.5, session.tick_hz)
            try:
                await asyncio.wait_for(speed_changed.wait(), timeout=interval)
                speed_changed.clear()
                continue
            except TimeoutError:
                pass

            if not clients:
                continue

            frame: FrameState | None = None
            async with lock:
                if clients and not session.paused:
                    try:
                        frame = await asyncio.to_thread(session.step)
                    except Exception as exc:  # noqa: BLE001 - isolate ticker loop from step errors
                        session.agent.llm_status = f"error: tick failed ({exc})"
                        frame = session.snapshot()
            if frame is not None:
                await broadcast_frame(frame)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        ticker_task: asyncio.Task[None] | None = None
        if auto_tick:
            ticker_task = asyncio.create_task(single_ticker_loop())
        try:
            yield
        finally:
            if ticker_task is not None:
                ticker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ticker_task
            clients.clear()
            session.close()

    app = FastAPI(title="PET-AI Cognitive Bridge", version="0.7.0", lifespan=lifespan)
    app.state.session = session

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/inspector")

    @app.get("/inspector", response_model=None)
    async def get_inspector() -> FileResponse:
        if not DEFAULT_INSPECTOR_PATH.exists():
            raise HTTPException(status_code=404, detail="Telemetry inspector HTML not found.")
        return FileResponse(DEFAULT_INSPECTOR_PATH, media_type="text/html")

    @app.get("/health")
    async def health() -> dict[str, object]:
        async with lock:
            snap = session.snapshot()
        return {
            "status": "ok",
            "tick": snap.tick,
            "preset": snap.preset_label,
            "tick_hz": snap.tick_hz,
            "llm_status": snap.llm_status,
            "llm_backend": snap.llm_backend,
            "presets_available": sorted(PRESET_BUILDERS.keys()),
        }

    @app.get("/archive")
    async def get_archive() -> dict[str, Any]:
        return session.load_archive()

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        use_msgpack = websocket.query_params.get("format") == "msgpack"

        try:
            async with lock:
                initial_snap = session.snapshot()
            async with broadcast_lock:
                await _send_frame(websocket, initial_snap, use_msgpack)
                clients[websocket] = use_msgpack

            while True:
                raw_text = await websocket.receive_text()
                try:
                    cmd = ClientCommand.model_validate_json(raw_text)
                except ValidationError as exc:
                    async with broadcast_lock:
                        await websocket.send_text(json.dumps({"error": str(exc)}))
                    continue
                async with lock:
                    frame = await asyncio.to_thread(session.apply_command, cmd)
                if cmd.type == "set_speed":
                    speed_changed.set()
                await broadcast_frame(frame)
        except WebSocketDisconnect:
            return
        finally:
            clients.pop(websocket, None)

    return app