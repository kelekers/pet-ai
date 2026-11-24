from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from memory.memory_store import EpisodicMemoryStore, FastEmbedder
from memory.prior_adaptation import AdaptivePetAgent, SensoryEvent
from sim_core.genome import TraitGenome
from sim_core.world import RoomWorld


def run_twin_experiment(ticks: int = 240, seed: int = 42) -> None:
    print("=== Sprint 4: Twin Divergence Experiment (PAD + sqlite-vec ONNX) ===\n")
    print("Loading CPU ONNX embedder (sentence-transformers/all-MiniLM-L6-v2)...")
    t0 = time.perf_counter()
    embedder = FastEmbedder()
    print(f"Embedder ready in {time.perf_counter() - t0:.2f}s (dim={embedder.dim})\n")

    # Identical balanced genome for both twins
    twin_genome = TraitGenome(
        neuroticism=0.58,
        extraversion=0.50,
        dominance=0.45,
        impulsiveness=0.35,
        agreeableness=0.55,
        curiosity=0.55,
        laziness=0.55,
        attachment=0.52,
        chaos=0.25,
        fur_density=0.65,
    )

    store_a = EpisodicMemoryStore(":memory:", embedder=embedder)
    store_b = EpisodicMemoryStore(":memory:", embedder=embedder)

    clone_a = AdaptivePetAgent(
        genome=twin_genome,
        world=RoomWorld.create_default(seed=seed, ambient_temp_c=21.0),
        memory_store=store_a,
        seed=seed,
    )
    clone_b = AdaptivePetAgent(
        genome=twin_genome,
        world=RoomWorld.create_default(seed=seed, ambient_temp_c=21.0),
        memory_store=store_b,
        seed=seed,
    )

    trust_a: list[float] = []
    trust_b: list[float] = []
    pleasure_a: list[float] = []
    pleasure_b: list[float] = []
    pos_a: list[tuple[float, float]] = []
    pos_b: list[tuple[float, float]] = []
    actions_a: Counter[str] = Counter()
    actions_b: Counter[str] = Counter()

    for tick in range(1, ticks + 1):
        # Expose Clone A to calm care and Clone B to startling noises every 20 ticks
        if tick % 20 == 5:
            clone_a.inject_event(SensoryEvent.calm_petting())
            clone_b.inject_event(SensoryEvent.startling_user_shout())
        elif tick % 40 == 25:
            clone_a.inject_event(SensoryEvent.gentle_feeding())
            clone_b.inject_event(SensoryEvent.loud_kitchen_clatter())

        res_a = clone_a.step()
        res_b = clone_b.step()

        trust_a.append(res_a.user_trust)
        trust_b.append(res_b.user_trust)
        pleasure_a.append(res_a.pad.pleasure)
        pleasure_b.append(res_b.pad.pleasure)
        pos_a.append((res_a.tick_record.x, res_a.tick_record.y))
        pos_b.append((res_b.tick_record.x, res_b.tick_record.y))
        actions_a[res_a.cognitive_step.action_name] += 1
        actions_b[res_b.cognitive_step.action_name] += 1

    # Measure vector retrieval latency (average over 20 queries)
    q_start = time.perf_counter()
    for _ in range(20):
        store_b.retrieve_similar("loud noise near kitchen and user", k=5)
    avg_retrieval_ms = ((time.perf_counter() - q_start) / 20.0) * 1000.0

    user_xy = np.array([3.0, 4.3])
    mean_dist_a = float(np.mean([np.linalg.norm(np.array(p) - user_xy) for p in pos_a]))
    mean_dist_b = float(np.mean([np.linalg.norm(np.array(p) - user_xy) for p in pos_b]))

    print(f"[Clone A - Calm Environment] (memories={store_a.count()})")
    print(f"  Final user trust : {trust_a[-1]:+.3f} | Final mood: {clone_a.emotion.state.mood_label()}")
    print(f"  Mean dist to user: {mean_dist_a:.2f}m")
    print(f"  Top actions      : {actions_a.most_common(3)}\n")

    print(f"[Clone B - Erratic/Loud Environment] (memories={store_b.count()})")
    print(f"  Final user trust : {trust_b[-1]:+.3f} | Final mood: {clone_b.emotion.state.mood_label()}")
    print(f"  Mean dist to user: {mean_dist_b:.2f}m")
    print(f"  Top actions      : {actions_b.most_common(3)}\n")

    print(f"Avg sqlite-vec + ONNX retrieval latency: {avg_retrieval_ms:.2f} ms")

    # Save 3-panel research figure
    out_path = Path("data/runs/twin_divergence.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)

    t_axis = np.arange(1, ticks + 1)
    axes[0].plot(t_axis, trust_a, label="Clone A (Calm)", color="#2a9d8f", linewidth=2.0)
    axes[0].plot(t_axis, trust_b, label="Clone B (Erratic)", color="#e76f51", linewidth=2.0)
    axes[0].axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    axes[0].set_title("User Trust Divergence")
    axes[0].set_xlabel("Simulation Tick")
    axes[0].set_ylabel("Associative Trust [-1, 1]")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].legend(frameon=False)

    for ax, positions, title, cmap in (
        (axes[1], pos_a, "Clone A Occupancy (Calm)", "Greens"),
        (axes[2], pos_b, "Clone B Occupancy (Erratic)", "OrRd"),
    ):
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        ax.hist2d(xs, ys, bins=12, range=[[0.0, 6.0], [0.0, 6.0]], cmap=cmap)
        for obj in clone_a.world.objects.values():
            ax.scatter(obj.x, obj.y, s=30, c="#1d3557", zorder=4)
            ax.annotate(obj.obj_id, (obj.x + 0.08, obj.y + 0.08), fontsize=7, color="#1d3557")
        ax.set_title(title)
        ax.set_xlabel("Room X (m)")
        ax.set_ylabel("Room Y (m)")
        ax.set_xlim(0.0, 6.0)
        ax.set_ylim(0.0, 6.0)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved divergence plot: {out_path}")

    store_a.close()
    store_b.close()


if __name__ == "__main__":
    run_twin_experiment()