from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from memory.emotion import PADState
from memory.memory_store import Episode
from neuro_symbolic.skill_dsl import ANOMALY_GATES, GROUNDED_PRIMITIVES, BehaviorSkill
from neuro_symbolic.verifier import SymbolicSkillVerifier, VerificationResult
from sim_core.genome import TraitGenome
from sim_core.world import RoomWorld

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass(frozen=True)
class SynthesisContext:
    genome: TraitGenome
    pad: PADState
    recent_memories: tuple[Episode, ...]
    existing_skills: tuple[str, ...]
    world_objects: tuple[str, ...]
    vfe: float
    trigger_reason: str
    user_prompt: str | None = None


@dataclass(frozen=True)
class SynthesisOutcome:
    skill: BehaviorSkill | None
    verification: VerificationResult | None
    attempts: int
    raw_errors: tuple[str, ...] = ()


class DirectorAdapter(Protocol):
    """Pluggable LLM backend producing raw JSON strings matching BehaviorSkill."""

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, object],
        context: SynthesisContext,
    ) -> str:
        """Return a JSON string candidate for BehaviorSkill."""


class OllamaAdapter:
    """Local Ollama adapter tuned for 6 GB VRAM (qwen3:1.7b, num_ctx=2048)."""

    def __init__(
        self,
        model: str = "qwen3:1.7b",
        host: str | None = None,
        num_ctx: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.host = host
        self.num_ctx = num_ctx
        self.temperature = temperature

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, object],
        context: SynthesisContext,
    ) -> str:
        del context
        import ollama

        client = ollama.Client(host=self.host) if self.host else ollama.Client()
        # Append /no_think for Qwen3 models so structured JSON returns rapidly
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": f"{system_prompt}\n/no_think"},
                {"role": "user", "content": user_prompt},
            ],
            format=json_schema,
            options={
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
            },
        )
        return str(response["message"]["content"])


class CloudAPIAdapter:
    """Zero-dependency adapter for OpenAI-compatible cloud endpoints (Groq, Gemini, OpenRouter)."""

    def __init__(
        self,
        endpoint_url: str,
        model: str,
        api_key_env: str = "PET_AI_API_KEY",
        timeout_sec: float = 20.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_sec = timeout_sec

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, object],
        context: SynthesisContext,
    ) -> str:
        del json_schema, context
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Environment variable '{self.api_key_env}' is not set.")

        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0.7,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            self.endpoint_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return str(payload["choices"][0]["message"]["content"])


class DeterministicMockAdapter:
    """Offline, reproducible synthesizer for CI tests and zero-VRAM local prototyping."""

    def __init__(self, emit_illegal_first_draft: bool = False) -> None:
        self.emit_illegal_first_draft = emit_illegal_first_draft
        self._emitted_illegal_once = False

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, object],
        context: SynthesisContext,
    ) -> str:
        del system_prompt, json_schema
        existing = set(context.existing_skills)
        g = context.genome
        prompt_text = (context.user_prompt or context.trigger_reason).lower()

        # Optional test mode: emit a broken draft once so the symbolic repair loop is exercised
        if self.emit_illegal_first_draft and not self._emitted_illegal_once:
            self._emitted_illegal_once = True
            return json.dumps(
                {
                    "name": "illegal_draft_ritual",
                    "trigger_context": "testing symbolic verifier repair feedback loop",
                    "min_vfe": 0.0,
                    "cooldown_ticks": 15,
                    "steps": [
                        {"primitive": "invert_gravity", "target_object": None, "duration_ticks": 2},
                        {"primitive": "approach", "target_object": "nonexistent_orb", "duration_ticks": 2},
                    ],
                }
            )

        # If this is a repair retry (feedback present in user_prompt), emit a clean grounded ritual
        if "verifier rejected" in user_prompt.lower() and g.max_anomaly < 0.80:
            return json.dumps(
                {
                    "name": "repaired_box_vigil",
                    "trigger_context": f"grounded recovery ritual for {prompt_text[:60]}",
                    "min_vfe": 0.0,
                    "cooldown_ticks": 15,
                    "steps": [
                        {"primitive": "approach", "target_object": "cardboard_box", "duration_ticks": 2},
                        {"primitive": "stare", "target_object": "user", "duration_ticks": 3},
                    ],
                }
            )

        # Candidate pool for high-anomaly "impossible cats"
        anomaly_catalog: list[dict[str, object]] = []
        if g.gravity_defiance >= 0.80:
            anomaly_catalog.append(
                {
                    "name": "ceiling_shelf_ascension",
                    "trigger_context": "high free energy ceiling walk onto high_shelf gravity ritual",
                    "min_vfe": 0.0,
                    "cooldown_ticks": 14,
                    "steps": [
                        {"primitive": "invert_gravity", "target_object": None, "duration_ticks": 3},
                        {"primitive": "approach", "target_object": "high_shelf", "duration_ticks": 3},
                        {"primitive": "stare", "target_object": "user", "duration_ticks": 2},
                    ],
                }
            )
        if g.spatial_phase >= 0.80:
            anomaly_catalog.append(
                {
                    "name": "ghost_box_phase_ambush",
                    "trigger_context": "curious spatial anomaly phasing through cardboard_box near user",
                    "min_vfe": 0.0,
                    "cooldown_ticks": 14,
                    "steps": [
                        {"primitive": "phase_object", "target_object": "cardboard_box", "duration_ticks": 2},
                        {"primitive": "approach", "target_object": "user", "duration_ticks": 2},
                        {"primitive": "stare", "target_object": "user", "duration_ticks": 2},
                    ],
                }
            )
        if g.sentience_glimmer >= 0.80:
            anomaly_catalog.append(
                {
                    "name": "telepathic_laptop_vigil",
                    "trigger_context": "intense user scrutiny telepathic gaze beside warm laptop",
                    "min_vfe": 0.0,
                    "cooldown_ticks": 14,
                    "steps": [
                        {"primitive": "approach", "target_object": "laptop", "duration_ticks": 2},
                        {
                            "primitive": "telepathic_gaze",
                            "target_object": "user",
                            "duration_ticks": 3,
                            "payload": "i_see_your_cursor",
                        },
                        {"primitive": "wait", "target_object": None, "duration_ticks": 1},
                    ],
                }
            )
        if g.mimicry_drive >= 0.80:
            anomaly_catalog.append(
                {
                    "name": "uncanny_echo_greeting",
                    "trigger_context": "user spoke aloud triggering uncanny voice mimicry",
                    "min_vfe": 0.0,
                    "cooldown_ticks": 14,
                    "steps": [
                        {"primitive": "stare", "target_object": "user", "duration_ticks": 2},
                        {
                            "primitive": "mimic_text",
                            "target_object": "user",
                            "duration_ticks": 2,
                            "payload": "feed the cat now",
                        },
                    ],
                }
            )

        # Grounded feline ritual catalog
        grounded_catalog: list[dict[str, object]] = [
            {
                "name": "radiator_groom_routine",
                "trigger_context": "chilly room seeking warmth near radiator and grooming",
                "min_vfe": 0.0,
                "cooldown_ticks": 15,
                "steps": [
                    {"primitive": "approach", "target_object": "radiator", "duration_ticks": 2},
                    {"primitive": "groom", "target_object": None, "duration_ticks": 3},
                ],
            },
            {
                "name": "cautious_user_chirp",
                "trigger_context": "curious greeting vocalization toward human user",
                "min_vfe": 0.0,
                "cooldown_ticks": 15,
                "steps": [
                    {"primitive": "crouch", "target_object": None, "duration_ticks": 2},
                    {"primitive": "stare", "target_object": "user", "duration_ticks": 2},
                    {
                        "primitive": "vocalize",
                        "target_object": "user",
                        "duration_ticks": 1,
                        "payload": "mrrp?",
                    },
                ],
            },
            {
                "name": "box_ambush_patrol",
                "trigger_context": "restless patrol inspecting cardboard_box and sunbeam",
                "min_vfe": 0.0,
                "cooldown_ticks": 15,
                "steps": [
                    {"primitive": "approach", "target_object": "cardboard_box", "duration_ticks": 2},
                    {"primitive": "crouch", "target_object": None, "duration_ticks": 2},
                    {"primitive": "approach", "target_object": "sunbeam", "duration_ticks": 2},
                ],
            },
        ]

        pool = anomaly_catalog + grounded_catalog
        for candidate in pool:
            if str(candidate["name"]) not in existing:
                return json.dumps(candidate)

        # Fallback uniquely named grounded ritual if catalog is exhausted
        idx = len(existing) + 1
        return json.dumps(
            {
                "name": f"improvised_vigil_{idx}",
                "trigger_context": f"improvised habit for {prompt_text[:80]}",
                "min_vfe": 0.0,
                "cooldown_ticks": 15,
                "steps": [
                    {"primitive": "stare", "target_object": "user", "duration_ticks": 2},
                    {"primitive": "groom", "target_object": None, "duration_ticks": 2},
                ],
            }
        )


def _extract_json_blob(raw_text: str) -> str:
    cleaned = _THINK_TAG_RE.sub("", raw_text).strip()
    fence_match = _CODE_FENCE_RE.search(cleaned)
    if fence_match:
        return fence_match.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


class SkillSynthesizer:
    """Async neuro-symbolic director combining LLM synthesis with symbolic verification."""

    def __init__(
        self,
        adapter: DirectorAdapter,
        verifier: SymbolicSkillVerifier | None = None,
        max_repair_attempts: int = 2,
    ) -> None:
        self.adapter = adapter
        self.verifier = verifier or SymbolicSkillVerifier()
        self.max_repair_attempts = max(1, max_repair_attempts)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pet-llm")
        self._pending: Future[SynthesisOutcome] | None = None

    @staticmethod
    def build_prompts(
        ctx: SynthesisContext,
        repair_errors: tuple[str, ...] = (),
    ) -> tuple[str, str]:
        unlocked_anomalies = [
            prim
            for prim, (field, thresh) in ANOMALY_GATES.items()
            if float(getattr(ctx.genome, field)) >= thresh
        ]
        allowed_primitives = sorted(GROUNDED_PRIMITIVES) + sorted(unlocked_anomalies)
        memories_str = (
            "; ".join(f"[{m.entity}:{m.event_tag}] {m.summary}" for m in ctx.recent_memories[:4])
            or "none"
        )
        existing_str = ", ".join(ctx.existing_skills[-8:]) or "none"

        system_prompt = (
            "You are the prefrontal cortex of a simulated feline creature. "
            "Output ONLY valid JSON matching the BehaviorSkill schema (2 to 6 steps, total "
            "duration <= 36 ticks, cooldown >= 10). Never invent primitives or objects outside "
            "the allowed lists. Never reuse an existing skill name."
        )
        user_lines = [
            f"Allowed primitives: {', '.join(allowed_primitives)}",
            f"Room objects: {', '.join(ctx.world_objects)}",
            f"Existing skill names (do not repeat): {existing_str}",
            (
                f"Traits: curiosity={ctx.genome.curiosity:.2f}, chaos={ctx.genome.chaos:.2f}, "
                f"laziness={ctx.genome.laziness:.2f}, fur={ctx.genome.fur_density:.2f}, "
                f"agility={ctx.genome.agility:.2f}, max_anomaly={ctx.genome.max_anomaly:.2f}"
            ),
            f"Mood: {ctx.pad.mood_label()} (P={ctx.pad.pleasure:+.2f}, A={ctx.pad.arousal:+.2f})",
            f"Recent memories: {memories_str}",
            f"Trigger: {ctx.trigger_reason} (VFE={ctx.vfe:.2f})",
        ]
        if ctx.user_prompt:
            user_lines.append(f"User said/did: {ctx.user_prompt}")
        if repair_errors:
            user_lines.append(
                "Previous draft was verifier rejected with errors: " + " | ".join(repair_errors)
            )
        return system_prompt, "\n".join(user_lines)

    def synthesize_and_verify(
        self,
        ctx: SynthesisContext,
        world: RoomWorld,
    ) -> SynthesisOutcome:
        """Synthesize a BehaviorSkill and verify it symbolically, with bounded repair retries."""
        schema = BehaviorSkill.model_json_schema()
        repair_errors: tuple[str, ...] = ()
        last_ver: VerificationResult | None = None

        for attempt in range(1, self.max_repair_attempts + 1):
            sys_p, usr_p = self.build_prompts(ctx, repair_errors=repair_errors)
            try:
                raw = self.adapter.generate_json(sys_p, usr_p, schema, ctx)
                blob = _extract_json_blob(raw)
                skill = BehaviorSkill.model_validate_json(blob)
            except (
                ValidationError,
                json.JSONDecodeError,
                urllib.error.URLError,
                OSError,
                RuntimeError,
                ValueError,
                KeyError,
            ) as exc:
                repair_errors = (f"Schema/transport error: {exc}",)
                continue

            ver = self.verifier.verify(
                skill=skill,
                genome=ctx.genome,
                world=world,
                current_tick=world.tick,
            )
            last_ver = ver
            if ver.is_valid:
                return SynthesisOutcome(skill=skill, verification=ver, attempts=attempt)
            repair_errors = ver.errors

        return SynthesisOutcome(
            skill=None,
            verification=last_ver,
            attempts=self.max_repair_attempts,
            raw_errors=repair_errors,
        )

    def request_async(self, ctx: SynthesisContext, world: RoomWorld) -> bool:
        """Dispatch synthesis to background thread without blocking real-time ticks."""
        if self._pending is not None and not self._pending.done():
            return False
        self._pending = self._executor.submit(self.synthesize_and_verify, ctx, world)
        return True

    def poll_async(self) -> SynthesisOutcome | None:
        """Collect completed async synthesis result if ready; returns None while running."""
        if self._pending is None or not self._pending.done():
            return None
        future = self._pending
        self._pending = None
        return future.result()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)