from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import msgpack
from fastapi.testclient import TestClient

from bridge.ipc_server import create_app
from bridge.protocol import FrameState
from neuro_symbolic.llm_director import SynthesisContext


class _FailingAdapter:
    """Simulated unreachable/failing LLM backend for resilience tests."""

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, object],
        context: SynthesisContext,
    ) -> str:
        del system_prompt, user_prompt, json_schema, context
        raise RuntimeError("Ollama server offline: connection refused")


def test_bridge_health_and_websocket_commands(tmp_path: Path) -> None:
    archive_file = tmp_path / "personality_archive.json"
    archive_file.write_text(
        json.dumps(
            {
                "metadata": {"num_elites": 1},
                "elites": [
                    {
                        "cell_index": 7,
                        "archetype": "hairless_heat_seeker",
                        "genome": {"fur_density": 0.0, "metabolism": 0.8},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    app = create_app(
        preset="hairless_sphynx",
        auto_tick=False,
        async_synthesis=False,
        archive_path=archive_file,
    )
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["preset"] == "hairless_sphynx"

        arch = client.get("/archive")
        assert arch.status_code == 200
        assert arch.json()["metadata"]["num_elites"] == 1

        with client.websocket_connect("/ws") as ws:
            initial = FrameState.model_validate_json(ws.receive_text())
            assert initial.morphology.fur_density == 0.0
            assert len(initial.world_objects) == 9

            # Switch to impossible_cat and trigger a gravity ritual via user_prompt
            ws.send_text(json.dumps({"type": "spawn_preset", "preset": "impossible_cat"}))
            spawned = FrameState.model_validate_json(ws.receive_text())
            assert spawned.preset_label == "impossible_cat"
            assert spawned.morphology.max_anomaly >= 0.90

            ws.send_text(
                json.dumps(
                    {
                        "type": "user_prompt",
                        "text": "ceiling high_shelf gravity ritual",
                    }
                )
            )
            ws.receive_text()

            ws.send_text(json.dumps({"type": "step_once"}))
            stepped = FrameState.model_validate_json(ws.receive_text())
            assert stepped.tick == 1
            assert stepped.active_skill_name == "ceiling_shelf_ascension"
            assert stepped.action == "skill:invert_gravity"
            assert stepped.anomaly_flags["gravity_inverted"] is True
            assert stepped.elevation > 1.5

            # Advance into the approach:high_shelf step and verify action & target preservation
            approach_frame = stepped
            for _ in range(4):
                ws.send_text(json.dumps({"type": "step_once"}))
                approach_frame = FrameState.model_validate_json(ws.receive_text())
                if approach_frame.active_primitive == "approach":
                    break
            assert approach_frame.active_primitive == "approach"
            assert approach_frame.action == "skill:approach:high_shelf"
            assert approach_frame.target_object_id == "high_shelf"

            # Move user and inject calm petting sensory event
            ws.send_text(json.dumps({"type": "move_user", "x": 1.5, "y": 2.0}))
            moved = FrameState.model_validate_json(ws.receive_text())
            user_obj = next(o for o in moved.world_objects if o.obj_id == "user")
            assert user_obj.x == 1.5 and user_obj.y == 2.0

            ws.send_text(json.dumps({"type": "sensory_event", "event": "calm_petting"}))
            petted = FrameState.model_validate_json(ws.receive_text())
            assert petted.user_trust > 0.0

            # Spawn directly from Sprint 3 MAP-Elites archive cell
            ws.send_text(json.dumps({"type": "spawn_archive_cell", "cell_index": 7}))
            from_arch = FrameState.model_validate_json(ws.receive_text())
            assert from_arch.preset_label.startswith("archive:7:")
            assert from_arch.morphology.fur_density == 0.0


def test_bridge_supports_binary_msgpack_frames() -> None:
    app = create_app(preset="staring_observer", auto_tick=False, async_synthesis=False)
    with TestClient(app) as client, client.websocket_connect("/ws?format=msgpack") as ws:
        raw_bytes = ws.receive_bytes()
        unpacked = msgpack.unpackb(raw_bytes, raw=False)
        frame = FrameState.model_validate(unpacked)
        assert frame.preset_label == "staring_observer"
        assert len(frame.world_objects) == 9


def test_multi_client_websocket_does_not_double_step() -> None:
    app = create_app(
        preset="staring_observer",
        tick_hz=12.0,
        auto_tick=True,
        async_synthesis=False,
    )
    with (
        TestClient(app) as client,
        client.websocket_connect("/ws") as ws1,
        client.websocket_connect("/ws") as ws2,
    ):
        init1 = FrameState.model_validate_json(ws1.receive_text())
        init2 = FrameState.model_validate_json(ws2.receive_text())
        assert init2.tick >= init1.tick

        ticks1: list[int] = []
        ticks2: list[int] = []
        for _ in range(3):
            f1 = FrameState.model_validate_json(ws1.receive_text())
            f2 = FrameState.model_validate_json(ws2.receive_text())
            ticks1.append(f1.tick)
            ticks2.append(f2.tick)

        # Each client observes strictly consecutive +1 ticks (single shared ticker)
        assert [b - a for a, b in itertools.pairwise(ticks1)] == [1, 1]
        assert [b - a for a, b in itertools.pairwise(ticks2)] == [1, 1]
        # The two clients stay synchronized on the same broadcast stream
        assert abs(ticks1[-1] - ticks2[-1]) <= 1


def test_failing_llm_adapter_is_isolated_and_surfaces_status() -> None:
    app = create_app(
        preset="impossible_cat",
        auto_tick=False,
        async_synthesis=True,
        adapter_override=_FailingAdapter(),
    )
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        initial = FrameState.model_validate_json(ws.receive_text())
        assert initial.llm_status == "idle"

        ws.send_text(
            json.dumps(
                {
                    "type": "user_prompt",
                    "text": "ceiling high_shelf gravity ritual",
                }
            )
        )
        ws.receive_text()

        # First step dispatches async synthesis; subsequent steps must keep ticking & surface error
        saw_error = False
        last_frame = initial
        for _ in range(15):
            ws.send_text(json.dumps({"type": "step_once"}))
            last_frame = FrameState.model_validate_json(ws.receive_text())
            if last_frame.llm_status.startswith("error:"):
                saw_error = True
                break
            time.sleep(0.01)

        assert saw_error
        assert "offline" in last_frame.llm_status.lower()
        assert last_frame.tick >= 1

        # Verify subsequent ticks continue uninterrupted after LLM failure
        prev_tick = last_frame.tick
        ws.send_text(json.dumps({"type": "step_once"}))
        next_frame = FrameState.model_validate_json(ws.receive_text())
        assert next_frame.tick == prev_tick + 1


def test_user_belief_calm_updates_from_active_inference() -> None:
    app = create_app(preset="staring_observer", auto_tick=False, async_synthesis=False)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        initial = FrameState.model_validate_json(ws.receive_text())
        assert initial.user_belief_calm == 0.5

        beliefs: list[float] = []
        last_frame = initial
        for _ in range(15):
            ws.send_text(json.dumps({"type": "step_once"}))
            last_frame = FrameState.model_validate_json(ws.receive_text())
            beliefs.append(last_frame.user_belief_calm)

        assert all(0.0 <= b <= 1.0 for b in beliefs)
        assert any(abs(b - 0.5) > 1e-3 for b in beliefs)
        assert len(last_frame.policy_probs) == 8
        assert abs(sum(last_frame.policy_probs.values()) - 1.0) < 0.01


def test_reset_room_and_set_speed_commands() -> None:
    app = create_app(preset="chaotic_gremlin", tick_hz=6.0, auto_tick=False, async_synthesis=False)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        initial = FrameState.model_validate_json(ws.receive_text())
        assert initial.tick_hz == 6.0

        # Live speed update
        ws.send_text(json.dumps({"type": "set_speed", "tick_hz": 12.0}))
        faster = FrameState.model_validate_json(ws.receive_text())
        assert faster.tick_hz == 12.0
        assert client.get("/health").json()["tick_hz"] == 12.0

        # Break the water glass directly and verify reset_room restores it
        app.state.session.agent.world.objects["water_glass"].is_intact = False
        app.state.session.agent.world.ambient_noise = 0.9
        ws.send_text(json.dumps({"type": "reset_room"}))
        restored = FrameState.model_validate_json(ws.receive_text())
        glass = next(o for o in restored.world_objects if o.obj_id == "water_glass")
        assert glass.is_intact is True
        assert app.state.session.agent.world.ambient_noise == 0.1

        # Also verify refill_water_glass restores water_glass specifically
        app.state.session.agent.world.objects["water_glass"].is_intact = False
        ws.send_text(json.dumps({"type": "refill_water_glass"}))
        refilled = FrameState.model_validate_json(ws.receive_text())
        glass_after = next(o for o in refilled.world_objects if o.obj_id == "water_glass")
        assert glass_after.is_intact is True


def test_godot_client_files_present() -> None:
    required = [
        Path("godot_client/project.godot"),
        Path("godot_client/scenes/MainRoom.tscn"),
        Path("godot_client/scripts/BrainBridge.gd"),
        Path("godot_client/scripts/RoomBuilder.gd"),
        Path("godot_client/scripts/MicroExpressionController.gd"),
        Path("godot_client/scripts/ProceduralPet.gd"),
        Path("godot_client/scripts/OrbitCamera.gd"),
        Path("godot_client/scripts/VFESparkline.gd"),
        Path("godot_client/scripts/TelemetryHUD.gd"),
        Path("godot_client/scripts/MainController.gd"),
    ]
    for path in required:
        assert path.exists() and path.stat().st_size > 80, f"Missing or empty: {path}"