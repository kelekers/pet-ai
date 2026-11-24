from __future__ import annotations

import argparse

from memory.memory_store import HashEmbedder
from neuro_symbolic.llm_director import DeterministicMockAdapter, DirectorAdapter, OllamaAdapter
from neuro_symbolic.orchestrator import NeuroSymbolicPetAgent
from sim_core.genome import TraitGenome
from sim_core.world import RoomWorld


def run_agent_demo(label: str, genome: TraitGenome, adapter: DirectorAdapter) -> None:
    print(f"=== {label} (max_anomaly={genome.max_anomaly:.2f}) ===")
    agent = NeuroSymbolicPetAgent(
        genome=genome,
        world=RoomWorld.create_default(seed=42, ambient_temp_c=20.0),
        adapter=adapter,
        embedder=HashEmbedder(),
        vfe_trigger_threshold=2.8,
        vfe_sustain_ticks=10,
        seed=42,
    )

    prompts = {
        12: "Look up at the ceiling high_shelf and show me a gravity trick",
        42: "Check out the cardboard_box spatial anomaly near me",
        72: "Stare at me from beside the warm laptop with telepathic intensity",
        105: "Do that ceiling high_shelf gravity ritual again",
        135: "Do the cardboard_box spatial anomaly ambush again",
    }

    synthesized: list[str] = []
    reused_events: list[str] = []
    seen_anomalies: set[str] = set()

    for tick in range(1, 161):
        if tick in prompts:
            agent.send_user_prompt(prompts[tick])

        res = agent.step(async_synthesis=False)
        if res.synthesized_skill:
            synthesized.append(res.synthesized_skill)
            print(f"  [Tick {tick:3d}] SYNTHESIZED & VERIFIED: {res.synthesized_skill}")
        if res.mode == "skill_reused" and res.active_skill_name:
            tag = f"{res.active_skill_name}@{tick}"
            if not reused_events or not reused_events[-1].startswith(res.active_skill_name):
                reused_events.append(tag)
                print(f"  [Tick {tick:3d}] RECALLED & REUSED   : {res.active_skill_name}")

        for flag, val in res.anomaly_flags.items():
            if val:
                seen_anomalies.add(f"{flag}={val}")

    print(f"  Library size    : {agent.library.count()} skills -> {agent.library.list_names()}")
    print(f"  Total reuses    : {agent.library.total_reuses()}")
    print(f"  Anomaly effects : {sorted(seen_anomalies) or ['NONE (strictly grounded)']}\n")
    agent.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 5 Neuro-Symbolic Skill Synthesizer Demo")
    parser.add_argument(
        "--live-ollama",
        action="store_true",
        help="Use local Ollama (qwen3:1.7b) instead of DeterministicMockAdapter.",
    )
    parser.add_argument("--model", default="qwen3:1.7b", help="Ollama model tag.")
    args = parser.parse_args()

    if args.live_ollama:
        print(f"Running with LIVE local Ollama model: {args.model}\n")
        adapter_impossible: DirectorAdapter = OllamaAdapter(model=args.model)
        adapter_grounded: DirectorAdapter = OllamaAdapter(model=args.model)
    else:
        print("Running with DeterministicMockAdapter (pass --live-ollama to use qwen3:1.7b)\n")
        adapter_impossible = DeterministicMockAdapter()
        adapter_grounded = DeterministicMockAdapter()

    impossible = TraitGenome.preset_impossible_cat().model_copy(update={"mimicry_drive": 0.90})
    grounded = TraitGenome.preset_hairless_sphynx()

    run_agent_demo("Impossible Cat", impossible, adapter_impossible)
    run_agent_demo("Grounded Hairless Sphynx", grounded, adapter_grounded)


if __name__ == "__main__":
    main()