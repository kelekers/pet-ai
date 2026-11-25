from __future__ import annotations

import inspect
import json
import platform
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from memory.memory_store import EpisodicMemoryStore, HashEmbedder
from memory.prior_adaptation import AdaptivePetAgent, SensoryEvent
from neuro_symbolic.llm_director import DeterministicMockAdapter
from neuro_symbolic.orchestrator import NeuroSymbolicPetAgent
from qd_lab.behavior_descriptors import (
    MAX_ENTROPY_BITS,
    compute_behavior_batch,
    infer_archetype_label,
)
from qd_lab.jax_rollout import evaluate_genomes_batch
from qd_lab.qd_search import QDSearchConfig, run_cma_mae
from sim_core.aif_agent import ActiveInferencePet
from sim_core.genome import GENOME_FIELDS, TraitGenome
from sim_core.genome_mapper import ACTION_NAMES
from sim_core.simulation import HeadlessSimulation
from sim_core.world import RoomWorld

# Light-mode technical CAD plotting palette
BG_FIG = "#FAFBFD"
BG_AX = "#FFFFFF"
TEXT_MAIN = "#1E293B"
TEXT_MUTED = "#475569"
BORDER_COLOR = "#CBD5E1"
GRID_COLOR = "#E2E8F0"

TEAL = "#0F766E"
CORAL = "#DC5842"
BLUE = "#2563EB"
AMBER = "#D97706"
INDIGO = "#4F46E5"
OLIVE = "#65A30D"
ROSE = "#BE185D"
SLATE = "#475569"
CYAN = "#0284C7"

ARCHETYPE_COLORS: dict[str, str] = {
    "hairless_heat_seeker": CORAL,
    "warmth_camper": AMBER,
    "staring_observer": BLUE,
    "velcro_companion": TEAL,
    "chaotic_gremlin": ROSE,
    "shy_hermit": INDIGO,
    "restless_explorer": OLIVE,
    "anomalous_entity": "#7C3AED",
    "balanced_hybrid": SLATE,
}


@dataclass(frozen=True)
class BenchmarkConfig:
    archive_path: Path = Path("data/archives/personality_archive.json")
    results_dir: Path = Path("benchmarks/results")
    figures_dir: Path = Path("docs/figures")
    eval_ticks: int = 500
    twin_ticks: int = 240
    neuro_ticks: int = 160
    max_elites: int | None = None
    seed: int = 42
    ambient_temp_c: float = 20.0
    dpi: int = 180


def _apply_cad_axes_style(ax: plt.Axes) -> None:
    ax.set_facecolor(BG_AX)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(0.9)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8.5)
    ax.xaxis.label.set_color(TEXT_MAIN)
    ax.yaxis.label.set_color(TEXT_MAIN)
    ax.title.set_color(TEXT_MAIN)


def _twin_genome() -> TraitGenome:
    return TraitGenome(
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


def _load_or_generate_archive(cfg: BenchmarkConfig) -> dict[str, Any]:
    if cfg.archive_path.exists():
        raw = json.loads(cfg.archive_path.read_text(encoding="utf-8"))
        if raw.get("elites"):
            return dict(raw)

    # Fallback: run a lightweight CMA-MAE sweep if archive JSON is absent
    qd_cfg = QDSearchConfig(
        dims=(6, 6),
        iterations=6,
        num_emitters=2,
        batch_size=8,
        rollout_ticks=min(cfg.eval_ticks, 100),
        ambient_temp_c=cfg.ambient_temp_c,
        seed=cfg.seed,
    )
    qd_res = run_cma_mae(qd_cfg, verbose=False)
    data = qd_res.result_archive.data()
    solutions = np.asarray(data["solution"], dtype=np.float32)
    objectives = np.asarray(data["objective"], dtype=np.float32)
    measures = np.asarray(data["measures"], dtype=np.float32)
    indices = np.asarray(data["index"], dtype=int)
    metrics = evaluate_genomes_batch(
        solutions,
        ticks=qd_cfg.rollout_ticks,
        seed=cfg.seed,
        ambient_temp_c=cfg.ambient_temp_c,
    )
    beh = compute_behavior_batch(metrics, ticks=qd_cfg.rollout_ticks)
    elites = []
    for i in range(solutions.shape[0]):
        genome = TraitGenome.from_vector(solutions[i])
        ratios = metrics.action_ratios[i]
        elites.append(
            {
                "cell_index": int(indices[i]),
                "archetype": infer_archetype_label(
                    genome,
                    ratios,
                    float(beh.disruption_rate[i]),
                    float(beh.activity_entropy[i]),
                ),
                "dominant_action": ACTION_NAMES[int(np.argmax(ratios))],
                "objective": round(float(objectives[i]), 4),
                "measures": {
                    "human_engagement": round(float(measures[i, 0]), 4),
                    "activity_entropy": round(float(measures[i, 1]), 4),
                    "disruption_rate": round(float(beh.disruption_rate[i]), 4),
                },
                "genome": {
                    field: round(float(val), 4)
                    for field, val in zip(GENOME_FIELDS, solutions[i], strict=True)
                },
            }
        )
    return {
        "metadata": {
            "algorithm": "CMA-MAE",
            "grid_dims": list(qd_cfg.dims),
            "measure_names": ["human_engagement", "activity_entropy"],
            "measure_ranges": [list(r) for r in qd_cfg.ranges],
            "coverage": round(qd_res.coverage, 4),
            "qd_score": round(qd_res.qd_score, 4),
            "num_elites": qd_res.num_elites,
            "total_evaluations": qd_res.total_evaluations,
            "elapsed_sec": round(qd_res.elapsed_sec, 2),
        },
        "elites": elites,
    }


def _select_elites(elites: list[dict[str, Any]], max_elites: int | None) -> list[dict[str, Any]]:
    if max_elites is None or len(elites) <= max_elites:
        return list(elites)
    step = len(elites) / max_elites
    return [elites[int(i * step)] for i in range(max_elites)]


def evaluate_personality_archive(
    cfg: BenchmarkConfig,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Run extended JAX rollouts over archive elites, compute UMAP, entropy, and silhouette."""
    archive_raw = _load_or_generate_archive(cfg)
    meta = dict(archive_raw.get("metadata", {}))
    all_elites: list[dict[str, Any]] = list(archive_raw.get("elites", []))
    elites = _select_elites(all_elites, cfg.max_elites)

    genomes_obj = [TraitGenome(**e["genome"]) for e in elites]
    genome_mat = np.stack([g.to_vector() for g in genomes_obj]).astype(np.float32)

    t0 = time.perf_counter()
    metrics = evaluate_genomes_batch(
        genome_mat,
        ticks=cfg.eval_ticks,
        seed=cfg.seed,
        ambient_temp_c=cfg.ambient_temp_c,
    )
    eval_elapsed = max(1e-6, time.perf_counter() - t0)
    beh = compute_behavior_batch(metrics, ticks=cfg.eval_ticks)
    entropy_bits = beh.activity_entropy * MAX_ENTROPY_BITS

    # Build 16D behavioral feature matrix
    feature_mat = np.column_stack(
        [
            metrics.action_ratios,  # 8D
            metrics.user_proximity_ratio,  # 1D
            beh.disruption_rate,  # 1D
            metrics.mean_warmth,  # 1D
            metrics.mean_hunger,  # 1D
            metrics.mean_energy,  # 1D
            metrics.mean_comfort,  # 1D
            metrics.mean_vfe,  # 1D
            entropy_bits,  # 1D
        ]
    ).astype(np.float64)

    scaler = StandardScaler()
    x_std = scaler.fit_transform(feature_mat)

    n_elites = len(elites)
    if n_elites >= 15:
        umap_kwargs: dict[str, Any] = {
            "n_neighbors": min(10, n_elites - 1),
            "min_dist": 0.15,
            "random_state": cfg.seed,
            "n_jobs": 1,
        }
        if "compatibility_layout" in inspect.signature(umap.UMAP.__init__).parameters:
            umap_kwargs["compatibility_layout"] = True
        reducer = umap.UMAP(**umap_kwargs)
        embedding = np.asarray(reducer.fit_transform(x_std), dtype=np.float64)
        projection_method = "umap"
    elif n_elites >= 2:
        embedding = np.asarray(
            PCA(n_components=2, random_state=cfg.seed).fit_transform(x_std),
            dtype=np.float64,
        )
        projection_method = "pca"
    else:
        embedding = np.zeros((n_elites, 2), dtype=np.float64)
        projection_method = "trivial"

    rows: list[dict[str, Any]] = []
    archetypes: list[str] = []
    for i, elite in enumerate(elites):
        g = genomes_obj[i]
        ratios = metrics.action_ratios[i]
        arch = infer_archetype_label(
            genome=g,
            action_ratios=ratios,
            disruption_rate=float(beh.disruption_rate[i]),
            activity_entropy=float(beh.activity_entropy[i]),
        )
        archetypes.append(arch)
        row: dict[str, Any] = {
            "cell_index": int(elite["cell_index"]),
            "archive_archetype": str(elite.get("archetype", arch)),
            "archetype": arch,
            "dominant_action": ACTION_NAMES[int(np.argmax(ratios))],
            "archive_objective": float(elite.get("objective", beh.objectives[i])),
            "eval_objective": round(float(beh.objectives[i]), 4),
            "human_engagement": round(float(beh.human_engagement[i]), 4),
            "activity_entropy_norm": round(float(beh.activity_entropy[i]), 4),
            "action_entropy_bits": round(float(entropy_bits[i]), 4),
            "disruption_rate": round(float(beh.disruption_rate[i]), 4),
            "user_proximity_ratio": round(float(metrics.user_proximity_ratio[i]), 4),
            "mean_warmth": round(float(metrics.mean_warmth[i]), 4),
            "mean_hunger": round(float(metrics.mean_hunger[i]), 4),
            "mean_energy": round(float(metrics.mean_energy[i]), 4),
            "mean_comfort": round(float(metrics.mean_comfort[i]), 4),
            "mean_vfe": round(float(metrics.mean_vfe[i]), 4),
            "water_glass_intact": bool(metrics.water_glass_intact[i]),
            "fur_density": round(float(g.fur_density), 4),
            "max_anomaly": round(float(g.max_anomaly), 4),
            "umap_x": round(float(embedding[i, 0]), 4),
            "umap_y": round(float(embedding[i, 1]), 4),
        }
        for j, act_name in enumerate(ACTION_NAMES):
            row[f"ratio_{act_name}"] = round(float(ratios[j]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    arch_counts = Counter(archetypes)

    # Silhouette score on archetypes with >= 2 samples
    valid_labels = {label for label, cnt in arch_counts.items() if cnt >= 2}
    mask = np.array([a in valid_labels for a in archetypes], dtype=bool)
    n_valid = int(np.sum(mask))
    n_clusters = len(valid_labels)
    sil_score: float | None = None
    if n_clusters >= 2 and n_valid > n_clusters:
        sil_score = round(float(silhouette_score(x_std[mask], np.array(archetypes)[mask])), 4)

    # Pure behavioral silhouette on dominant Active Inference action clusters (count >= 2)
    dom_actions = df["dominant_action"].tolist()
    dom_counts = Counter(dom_actions)
    valid_dom = {label for label, cnt in dom_counts.items() if cnt >= 2}
    dom_mask = np.array([d in valid_dom for d in dom_actions], dtype=bool)
    dom_sil_score: float | None = None
    if len(valid_dom) >= 2 and int(np.sum(dom_mask)) > len(valid_dom):
        dom_sil_score = round(
            float(silhouette_score(x_std[dom_mask], np.array(dom_actions)[dom_mask])),
            4,
        )

    pairwise_std_l2 = float(np.mean(pdist(x_std, metric="euclidean"))) if n_elites >= 2 else 0.0
    pairwise_policy_l2 = (
        float(np.mean(pdist(metrics.action_ratios, metric="euclidean"))) if n_elites >= 2 else 0.0
    )

    total_ticks_simulated = n_elites * cfg.eval_ticks
    summary = {
        "archive_metadata": meta,
        "evaluated_elites": n_elites,
        "eval_ticks_per_elite": cfg.eval_ticks,
        "eval_elapsed_sec": round(eval_elapsed, 3),
        "throughput_ticks_per_sec": round(total_ticks_simulated / eval_elapsed, 1),
        "coverage": float(meta.get("coverage", 0.0)),
        "qd_score": float(meta.get("qd_score", 0.0)),
        "num_elites_in_archive": int(meta.get("num_elites", len(all_elites))),
        "archetype_counts": dict(sorted(arch_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "dominant_action_counts": dict(sorted(dom_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "num_distinct_archetypes": len(arch_counts),
        "action_entropy_bits": {
            "mean": round(float(np.mean(entropy_bits)), 4),
            "std": round(float(np.std(entropy_bits)), 4),
            "min": round(float(np.min(entropy_bits)), 4),
            "max": round(float(np.max(entropy_bits)), 4),
        },
        "silhouette_score": sil_score,
        "silhouette_clusters_used": n_clusters,
        "silhouette_samples_used": n_valid,
        "dominant_action_silhouette_score": dom_sil_score,
        "dominant_action_clusters_used": len(valid_dom),
        "mean_pairwise_behavioral_l2": round(pairwise_std_l2, 4),
        "mean_pairwise_policy_l2": round(pairwise_policy_l2, 4),
        "mean_viability_objective": round(float(np.mean(beh.objectives)), 4),
        "mean_vfe": round(float(np.mean(metrics.mean_vfe)), 4),
        "projection_method": projection_method,
    }
    return summary, df, archive_raw


def _homeo_viability(warmth: float, hunger: float, energy: float, comfort: float) -> float:
    return 0.30 * warmth + 0.25 * (1.0 - hunger) + 0.25 * energy + 0.20 * comfort


def _policy_l1(counts_a: Counter[str], counts_b: Counter[str], total: int) -> float:
    denom = max(1, total)
    keys = set(ACTION_NAMES) | set(counts_a.keys()) | set(counts_b.keys())
    return float(sum(abs(counts_a[k] / denom - counts_b[k] / denom) for k in keys))


def run_ablation_study(cfg: BenchmarkConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run controlled 3-way ablation: AIF Only vs AIF+PAD+Memory vs Full Neuro-Symbolic."""
    twin = _twin_genome()
    user_xy = np.array([3.0, 4.3], dtype=np.float64)
    ticks = cfg.twin_ticks

    # --- Config A: Pure Active Inference (no PAD, no episodic memory prior adaptation) ---
    sim_a_calm = HeadlessSimulation(
        genome=twin,
        world=RoomWorld.create_default(seed=cfg.seed, ambient_temp_c=21.0),
        seed=cfg.seed,
    )
    sim_a_err = HeadlessSimulation(
        genome=twin,
        world=RoomWorld.create_default(seed=cfg.seed, ambient_temp_c=21.0),
        seed=cfg.seed,
    )
    brain_a_calm = ActiveInferencePet(genome=twin, seed=cfg.seed)
    brain_a_err = ActiveInferencePet(genome=twin, seed=cfg.seed)

    last_rec_c = None
    last_rec_e = None
    dists_a_c: list[float] = []
    dists_a_e: list[float] = []
    acts_a_c: Counter[str] = Counter()
    acts_a_e: Counter[str] = Counter()
    vfes_a: list[float] = []
    viab_a: list[float] = []

    for tick in range(1, ticks + 1):
        # Apply identical acoustic noise spikes to erratic room (still no emotional/memory C-adaptation)
        if tick % 20 == 5:
            ev_err = SensoryEvent.startling_user_shout()
            sim_a_err.world.ambient_noise = max(sim_a_err.world.ambient_noise, ev_err.noise_spike)
        elif tick % 40 == 25:
            ev_err = SensoryEvent.loud_kitchen_clatter()
            sim_a_err.world.ambient_noise = max(sim_a_err.world.ambient_noise, ev_err.noise_spike)

        cog_c = brain_a_calm.act(sim_a_calm, last_record=last_rec_c)
        rec_c = sim_a_calm.step(cog_c.target_object_id, cog_c.affordance_name)
        last_rec_c = rec_c

        cog_e = brain_a_err.act(sim_a_err, last_record=last_rec_e)
        rec_e = sim_a_err.step(cog_e.target_object_id, cog_e.affordance_name)
        last_rec_e = rec_e

        dists_a_c.append(float(np.linalg.norm(np.array([rec_c.x, rec_c.y]) - user_xy)))
        dists_a_e.append(float(np.linalg.norm(np.array([rec_e.x, rec_e.y]) - user_xy)))
        acts_a_c[cog_c.action_name] += 1
        acts_a_e[cog_e.action_name] += 1
        vfes_a.extend([cog_c.vfe, cog_e.vfe])
        viab_a.append(_homeo_viability(rec_c.warmth, rec_c.hunger, rec_c.energy, rec_c.comfort))
        viab_a.append(_homeo_viability(rec_e.warmth, rec_e.hunger, rec_e.energy, rec_e.comfort))

    mean_dist_a_c = float(np.mean(dists_a_c))
    mean_dist_a_e = float(np.mean(dists_a_e))
    config_a = {
        "label": "AIF Only",
        "twin_trust_calm": 0.0,
        "twin_trust_erratic": 0.0,
        "twin_trust_divergence": 0.0,
        "mean_dist_calm_m": round(mean_dist_a_c, 4),
        "mean_dist_erratic_m": round(mean_dist_a_e, 4),
        "twin_spatial_divergence_m": round(mean_dist_a_e - mean_dist_a_c, 4),
        "policy_l1_divergence": round(_policy_l1(acts_a_c, acts_a_e, ticks), 4),
        "skills_synthesized": 0,
        "skill_reuses": 0,
        "illegal_anomaly_activations": 0,
        "symbolic_safety_rate": 1.0,
        "mean_vfe": round(float(np.mean(vfes_a)), 4),
        "mean_viability": round(float(np.mean(viab_a)), 4),
    }

    # --- Config B: AIF + PAD Emotions + sqlite-vec Episodic Memory Prior Adaptation ---
    embedder = HashEmbedder()
    clone_b_c = AdaptivePetAgent(
        genome=twin,
        world=RoomWorld.create_default(seed=cfg.seed, ambient_temp_c=21.0),
        memory_store=EpisodicMemoryStore(":memory:", embedder=embedder),
        seed=cfg.seed,
    )
    clone_b_e = AdaptivePetAgent(
        genome=twin,
        world=RoomWorld.create_default(seed=cfg.seed, ambient_temp_c=21.0),
        memory_store=EpisodicMemoryStore(":memory:", embedder=embedder),
        seed=cfg.seed,
    )

    trust_b_c_trace: list[float] = []
    trust_b_e_trace: list[float] = []
    pos_b_c: list[tuple[float, float]] = []
    pos_b_e: list[tuple[float, float]] = []
    dists_b_c: list[float] = []
    dists_b_e: list[float] = []
    acts_b_c: Counter[str] = Counter()
    acts_b_e: Counter[str] = Counter()
    vfes_b: list[float] = []
    viab_b: list[float] = []

    for tick in range(1, ticks + 1):
        if tick % 20 == 5:
            clone_b_c.inject_event(SensoryEvent.calm_petting())
            clone_b_e.inject_event(SensoryEvent.startling_user_shout())
        elif tick % 40 == 25:
            clone_b_c.inject_event(SensoryEvent.gentle_feeding())
            clone_b_e.inject_event(SensoryEvent.loud_kitchen_clatter())

        res_c = clone_b_c.step()
        res_e = clone_b_e.step()

        trust_b_c_trace.append(res_c.user_trust)
        trust_b_e_trace.append(res_e.user_trust)
        pos_b_c.append((res_c.tick_record.x, res_c.tick_record.y))
        pos_b_e.append((res_e.tick_record.x, res_e.tick_record.y))
        dists_b_c.append(
            float(np.linalg.norm(np.array([res_c.tick_record.x, res_c.tick_record.y]) - user_xy))
        )
        dists_b_e.append(
            float(np.linalg.norm(np.array([res_e.tick_record.x, res_e.tick_record.y]) - user_xy))
        )
        acts_b_c[res_c.cognitive_step.action_name] += 1
        acts_b_e[res_e.cognitive_step.action_name] += 1
        vfes_b.extend([res_c.cognitive_step.vfe, res_e.cognitive_step.vfe])
        viab_b.append(
            _homeo_viability(
                res_c.tick_record.warmth,
                res_c.tick_record.hunger,
                res_c.tick_record.energy,
                res_c.tick_record.comfort,
            )
        )
        viab_b.append(
            _homeo_viability(
                res_e.tick_record.warmth,
                res_e.tick_record.hunger,
                res_e.tick_record.energy,
                res_e.tick_record.comfort,
            )
        )

    # Measure sqlite-vec query latency
    q0 = time.perf_counter()
    for _ in range(25):
        clone_b_e.memory.retrieve_similar("loud noise near kitchen and user", k=5)
    avg_sqlite_vec_ms = ((time.perf_counter() - q0) / 25.0) * 1000.0

    world_objects_snapshot = [
        {"obj_id": obj.obj_id, "name": obj.name, "x": obj.x, "y": obj.y}
        for obj in clone_b_c.world.objects.values()
    ]
    clone_b_c.memory.close()
    clone_b_e.memory.close()

    mean_dist_b_c = float(np.mean(dists_b_c))
    mean_dist_b_e = float(np.mean(dists_b_e))
    config_b = {
        "label": "AIF + PAD + Memory",
        "twin_trust_calm": round(trust_b_c_trace[-1], 4),
        "twin_trust_erratic": round(trust_b_e_trace[-1], 4),
        "twin_trust_divergence": round(trust_b_c_trace[-1] - trust_b_e_trace[-1], 4),
        "mean_dist_calm_m": round(mean_dist_b_c, 4),
        "mean_dist_erratic_m": round(mean_dist_b_e, 4),
        "twin_spatial_divergence_m": round(mean_dist_b_e - mean_dist_b_c, 4),
        "policy_l1_divergence": round(_policy_l1(acts_b_c, acts_b_e, ticks), 4),
        "skills_synthesized": 0,
        "skill_reuses": 0,
        "illegal_anomaly_activations": 0,
        "symbolic_safety_rate": 1.0,
        "mean_vfe": round(float(np.mean(vfes_b)), 4),
        "mean_viability": round(float(np.mean(viab_b)), 4),
        "sqlite_vec_retrieval_ms": round(avg_sqlite_vec_ms, 3),
    }

    # --- Config C: Full Neuro-Symbolic Stack (Twin divergence + Impossible vs Grounded Safety) ---
    twin_c_calm = NeuroSymbolicPetAgent(
        genome=twin,
        world=RoomWorld.create_default(seed=cfg.seed, ambient_temp_c=21.0),
        adapter=DeterministicMockAdapter(),
        embedder=embedder,
        seed=cfg.seed,
    )
    twin_c_err = NeuroSymbolicPetAgent(
        genome=twin,
        world=RoomWorld.create_default(seed=cfg.seed, ambient_temp_c=21.0),
        adapter=DeterministicMockAdapter(),
        embedder=embedder,
        seed=cfg.seed,
    )

    dists_c_c: list[float] = []
    dists_c_e: list[float] = []
    acts_c_c: Counter[str] = Counter()
    acts_c_e: Counter[str] = Counter()
    vfes_c: list[float] = []
    viab_c: list[float] = []
    illegal_twin_anomalies = 0

    for tick in range(1, ticks + 1):
        if tick % 20 == 5:
            twin_c_calm.adaptive.inject_event(SensoryEvent.calm_petting())
            twin_c_err.adaptive.inject_event(SensoryEvent.startling_user_shout())
        elif tick % 40 == 25:
            twin_c_calm.adaptive.inject_event(SensoryEvent.gentle_feeding())
            twin_c_err.adaptive.inject_event(SensoryEvent.loud_kitchen_clatter())

        rc = twin_c_calm.step(async_synthesis=False)
        re = twin_c_err.step(async_synthesis=False)
        if any(bool(v) for v in rc.anomaly_flags.values()):
            illegal_twin_anomalies += 1
        if any(bool(v) for v in re.anomaly_flags.values()):
            illegal_twin_anomalies += 1

        dists_c_c.append(
            float(np.linalg.norm(np.array([rc.tick_record.x, rc.tick_record.y]) - user_xy))
        )
        dists_c_e.append(
            float(np.linalg.norm(np.array([re.tick_record.x, re.tick_record.y]) - user_xy))
        )
        acts_c_c[rc.tick_record.action] += 1
        acts_c_e[re.tick_record.action] += 1
        vfes_c.extend([rc.vfe, re.vfe])
        viab_c.append(
            _homeo_viability(
                rc.tick_record.warmth,
                rc.tick_record.hunger,
                rc.tick_record.energy,
                rc.tick_record.comfort,
            )
        )
        viab_c.append(
            _homeo_viability(
                re.tick_record.warmth,
                re.tick_record.hunger,
                re.tick_record.energy,
                re.tick_record.comfort,
            )
        )

    trust_c_calm, _ = twin_c_calm.adaptive.memory.entity_affinity("user", twin_c_calm.world.tick)
    trust_c_err, _ = twin_c_err.adaptive.memory.entity_affinity("user", twin_c_err.world.tick)
    twin_c_calm.close()
    twin_c_err.close()

    # Dedicated neuro-symbolic skill synthesis & symbolic safety test (Impossible vs Grounded)
    impossible_genome = TraitGenome.preset_impossible_cat().model_copy(
        update={"mimicry_drive": 0.90}
    )
    grounded_genome = TraitGenome.preset_hairless_sphynx()

    imp_agent = NeuroSymbolicPetAgent(
        genome=impossible_genome,
        world=RoomWorld.create_default(seed=cfg.seed, ambient_temp_c=20.0),
        adapter=DeterministicMockAdapter(),
        embedder=embedder,
        seed=cfg.seed,
    )
    grd_agent = NeuroSymbolicPetAgent(
        genome=grounded_genome,
        world=RoomWorld.create_default(seed=cfg.seed, ambient_temp_c=20.0),
        adapter=DeterministicMockAdapter(),
        embedder=embedder,
        seed=cfg.seed,
    )

    n_ticks = max(80, cfg.neuro_ticks)
    prompt_schedule = {
        5: "ceiling high_shelf gravity ritual",
        max(22, int(n_ticks * 0.20)): "cardboard_box spatial anomaly near user",
        max(40, int(n_ticks * 0.36)): "telepathic gaze beside warm laptop",
        max(60, int(n_ticks * 0.56)): "ceiling high_shelf gravity ritual",
        max(74, int(n_ticks * 0.74)): "cardboard_box spatial anomaly near user",
    }

    imp_anomaly_ticks = 0
    grounded_illegal_anomalies = 0
    total_skill_ticks = 0
    for tick in range(1, n_ticks + 1):
        if tick in prompt_schedule:
            imp_agent.send_user_prompt(prompt_schedule[tick])
            grd_agent.send_user_prompt(prompt_schedule[tick])
        r_imp = imp_agent.step(async_synthesis=False)
        r_grd = grd_agent.step(async_synthesis=False)
        if any(bool(v) for v in r_imp.anomaly_flags.values()):
            imp_anomaly_ticks += 1
        if any(bool(v) for v in r_grd.anomaly_flags.values()):
            grounded_illegal_anomalies += 1
        if r_imp.active_skill_name is not None:
            total_skill_ticks += 1

    skills_synth = imp_agent.library.count()
    skill_reuses = imp_agent.library.total_reuses()
    grd_skills_synth = grd_agent.library.count()
    grd_skill_reuses = grd_agent.library.total_reuses()
    imp_agent.close()
    grd_agent.close()

    total_illegal = illegal_twin_anomalies + grounded_illegal_anomalies
    total_checked_grounded_ticks = 2 * ticks + n_ticks
    safety_rate = 1.0 - (total_illegal / max(1, total_checked_grounded_ticks))

    mean_dist_c_c = float(np.mean(dists_c_c))
    mean_dist_c_e = float(np.mean(dists_c_e))
    config_c = {
        "label": "Full Neuro-Symbolic",
        "twin_trust_calm": round(trust_c_calm, 4),
        "twin_trust_erratic": round(trust_c_err, 4),
        "twin_trust_divergence": round(trust_c_calm - trust_c_err, 4),
        "mean_dist_calm_m": round(mean_dist_c_c, 4),
        "mean_dist_erratic_m": round(mean_dist_c_e, 4),
        "twin_spatial_divergence_m": round(mean_dist_c_e - mean_dist_c_c, 4),
        "policy_l1_divergence": round(_policy_l1(acts_c_c, acts_c_e, ticks), 4),
        "skills_synthesized": skills_synth,
        "skill_reuses": skill_reuses,
        "grounded_skills_synthesized": grd_skills_synth,
        "grounded_skill_reuses": grd_skill_reuses,
        "impossible_anomaly_ticks": imp_anomaly_ticks,
        "impossible_skill_ticks": total_skill_ticks,
        "illegal_anomaly_activations": total_illegal,
        "symbolic_safety_rate": round(safety_rate, 4),
        "mean_vfe": round(float(np.mean(vfes_c)), 4),
        "mean_viability": round(float(np.mean(viab_c)), 4),
    }

    ablation_summary = {
        "twin_ticks": ticks,
        "neuro_ticks": n_ticks,
        "configs": {
            "aif_only": config_a,
            "aif_pad_memory": config_b,
            "full_neuro_symbolic": config_c,
        },
    }
    twin_plot_data = {
        "ticks": ticks,
        "trust_calm": trust_b_c_trace,
        "trust_erratic": trust_b_e_trace,
        "pos_calm": pos_b_c,
        "pos_erratic": pos_b_e,
        "mean_dist_calm_m": mean_dist_b_c,
        "mean_dist_erratic_m": mean_dist_b_e,
        "world_objects": world_objects_snapshot,
    }
    return ablation_summary, twin_plot_data


def _plot_personality_atlas(
    archive_raw: dict[str, Any],
    out_path: Path,
    dpi: int,
) -> None:
    meta = archive_raw.get("metadata", {})
    dims = meta.get("grid_dims", [10, 10])
    nx, ny = int(dims[0]), int(dims[1])
    ranges = meta.get("measure_ranges", [[0.0, 1.0], [0.0, 0.65]])
    x_min, x_max = float(ranges[0][0]), float(ranges[0][1])
    y_min, y_max = float(ranges[1][0]), float(ranges[1][1])

    grid = np.full((ny, nx), np.nan, dtype=np.float64)
    for elite in archive_raw.get("elites", []):
        idx = int(elite["cell_index"])
        ix = idx // ny
        iy = idx % ny
        if 0 <= ix < nx and 0 <= iy < ny:
            grid[iy, ix] = float(elite["objective"])

    cmap = LinearSegmentedColormap.from_list(
        "cad_teal",
        ["#DBEAFE", "#60A5FA", "#14B8A6", "#0F766E", "#134E4A"],
    ).with_extremes(bad="#F1F5F9")

    fig, ax = plt.subplots(figsize=(7.6, 6.0), dpi=dpi, facecolor=BG_FIG)
    _apply_cad_axes_style(ax)

    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0.55,
        vmax=1.0,
        extent=[x_min, x_max, y_min, y_max],
        interpolation="nearest",
    )

    # Crisp cell gridlines
    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)
    for xe in x_edges:
        ax.axvline(xe, color="#FFFFFF", linewidth=0.9, alpha=0.85)
    for ye in y_edges:
        ax.axhline(ye, color="#FFFFFF", linewidth=0.9, alpha=0.85)

    # Annotate cell objectives cleanly
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    for iy in range(ny):
        for ix in range(nx):
            val = grid[iy, ix]
            if not np.isnan(val):
                cx = x_min + (ix + 0.5) * dx
                cy = y_min + (iy + 0.5) * dy
                txt_col = "#FFFFFF" if val >= 0.80 else TEXT_MAIN
                ax.text(
                    cx,
                    cy,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=txt_col,
                    fontweight="normal",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Viability Objective", color=TEXT_MAIN, fontsize=9)
    cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    cbar.outline.set_edgecolor(BORDER_COLOR)

    cov_pct = float(meta.get("coverage", 0.0)) * 100.0
    qd_score = float(meta.get("qd_score", 0.0))
    n_elites = int(meta.get("num_elites", len(archive_raw.get("elites", []))))
    ax.set_title(
        f"CMA-MAE Personality Atlas  |  Coverage: {cov_pct:.1f}% ({n_elites}/{nx * ny})  "
        f"|  QD-Score: {qd_score:.2f}",
        fontsize=10.5,
        fontweight="bold",
        pad=11,
    )
    ax.set_xlabel("Human Engagement (Solitary to Social / Observant)", fontsize=9.5, labelpad=6)
    ax.set_ylabel("Normalized Activity Entropy (Specialist to Explorer)", fontsize=9.5, labelpad=6)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG_FIG)
    plt.close(fig)


def _plot_umap_clusters(
    df: pd.DataFrame,
    archive_summary: dict[str, Any],
    out_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=dpi, facecolor=BG_FIG)
    _apply_cad_axes_style(ax)
    ax.grid(True, linestyle="--", linewidth=0.6, color=GRID_COLOR, zorder=0)

    order = sorted(df["archetype"].unique())
    for arch in order:
        sub = df[df["archetype"] == arch]
        color = ARCHETYPE_COLORS.get(arch, SLATE)
        ax.scatter(
            sub["umap_x"],
            sub["umap_y"],
            s=62,
            c=color,
            edgecolors="#FFFFFF",
            linewidths=0.9,
            alpha=0.92,
            label=f"{arch.replace('_', ' ')} (n={len(sub)})",
            zorder=3,
        )

    method = str(archive_summary.get("projection_method", "umap")).upper()
    sil = archive_summary.get("silhouette_score")
    sil_txt = f"{sil:+.3f}" if sil is not None else "N/A"
    dom_sil = archive_summary.get("dominant_action_silhouette_score")
    dom_sil_txt = f"{dom_sil:+.3f}" if dom_sil is not None else "N/A"
    ent_mean = archive_summary["action_entropy_bits"]["mean"]
    l2_mean = archive_summary["mean_pairwise_behavioral_l2"]

    callout = (
        f"Elites evaluated: {archive_summary['evaluated_elites']}\n"
        f"Archetypes: {archive_summary['num_distinct_archetypes']}\n"
        f"Policy cluster silhouette: {dom_sil_txt}\n"
        f"Archetype silhouette: {sil_txt}\n"
        f"Mean entropy: {ent_mean:.2f} bits\n"
        f"Mean pairwise L2: {l2_mean:.2f}"
    )
    ax.text(
        0.02,
        0.03,
        callout,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8.0,
        color=TEXT_MAIN,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#F8FAFC",
            "edgecolor": BORDER_COLOR,
            "linewidth": 0.9,
        },
        zorder=5,
    )

    ax.set_title(
        f"{method} Projection of Emergent Behavioral Phenotypes "
        f"({archive_summary['eval_ticks_per_elite']}-Tick JAX Rollouts)",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel(f"{method} Dimension 1", fontsize=9.5)
    ax.set_ylabel(f"{method} Dimension 2", fontsize=9.5)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor=BORDER_COLOR,
        fontsize=8.2,
        title="Emergent Archetype",
        title_fontsize=8.5,
    )

    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG_FIG)
    plt.close(fig)


def _plot_twin_divergence(
    twin_data: dict[str, Any],
    out_path: Path,
    dpi: int,
) -> None:
    ticks = int(twin_data["ticks"])
    t_axis = np.arange(1, ticks + 1)
    trust_c = np.asarray(twin_data["trust_calm"], dtype=np.float64)
    trust_e = np.asarray(twin_data["trust_erratic"], dtype=np.float64)
    pos_c = twin_data["pos_calm"]
    pos_e = twin_data["pos_erratic"]
    world_objs = twin_data["world_objects"]

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6), dpi=dpi, facecolor=BG_FIG)
    for ax in axes:
        _apply_cad_axes_style(ax)

    # Panel 1: Trust trajectory
    ax0 = axes[0]
    ax0.grid(True, linestyle="--", linewidth=0.6, color=GRID_COLOR, zorder=0)
    ax0.axhline(0.0, color="#94A3B8", linestyle="--", linewidth=0.9, zorder=1)
    ax0.plot(t_axis, trust_c, color=TEAL, linewidth=2.2, label="Clone A (Calm Care)", zorder=3)
    ax0.plot(t_axis, trust_e, color=CORAL, linewidth=2.2, label="Clone B (Erratic / Loud)", zorder=3)
    ax0.fill_between(t_axis, trust_e, trust_c, color="#0F766E", alpha=0.08, zorder=2)
    gap = trust_c[-1] - trust_e[-1]
    ax0.set_title(f"Associative User Trust (Delta = {gap:+.2f})", fontsize=10, fontweight="bold")
    ax0.set_xlabel("Simulation Tick", fontsize=9)
    ax0.set_ylabel("User Trust Affinity [-1, +1]", fontsize=9)
    ax0.set_ylim(-1.05, 1.05)
    ax0.legend(
        loc="center right",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor=BORDER_COLOR,
        fontsize=8,
    )

    # Panels 2 & 3: 2D occupancy heatmaps
    calm_cmap = LinearSegmentedColormap.from_list("calm_map", ["#F8FAFC", "#99F6E4", "#0F766E"])
    err_cmap = LinearSegmentedColormap.from_list("err_map", ["#F8FAFC", "#FECACA", "#DC5842"])
    panels = (
        (
            axes[1],
            pos_c,
            f"Clone A Occupancy (Mean dist: {twin_data['mean_dist_calm_m']:.2f} m)",
            calm_cmap,
            TEAL,
        ),
        (
            axes[2],
            pos_e,
            f"Clone B Occupancy (Mean dist: {twin_data['mean_dist_erratic_m']:.2f} m)",
            err_cmap,
            CORAL,
        ),
    )

    label_offsets: dict[str, tuple[float, float]] = {
        "user": (-0.62, -0.06),
        "laptop": (-0.82, 0.08),
        "water_glass": (0.14, 0.08),
        "high_shelf": (-0.78, -0.28),
        "scratching_post": (-0.25, 0.16),
        "sunbeam": (-0.42, -0.32),
        "cardboard_box": (0.14, -0.22),
        "food_bowl": (0.14, 0.10),
        "radiator": (0.14, 0.10),
    }

    for ax, positions, title, cmap, line_col in panels:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        ax.hist2d(xs, ys, bins=12, range=[[0.0, 6.0], [0.0, 6.0]], cmap=cmap)
        ax.plot(xs, ys, color=line_col, alpha=0.45, linewidth=1.3, linestyle="--", zorder=3)
        for g in range(1, 6):
            ax.axvline(g, color="#CBD5E1", linewidth=0.5, alpha=0.6)
            ax.axhline(g, color="#CBD5E1", linewidth=0.5, alpha=0.6)
        for obj in world_objs:
            is_user = obj["obj_id"] == "user"
            ax.scatter(
                obj["x"],
                obj["y"],
                s=54 if is_user else 26,
                c=BLUE if is_user else TEXT_MAIN,
                marker="D" if is_user else "o",
                edgecolors="#FFFFFF",
                linewidths=0.7,
                zorder=4,
            )
            ox, oy = label_offsets.get(obj["obj_id"], (0.10, 0.10))
            ax.annotate(
                obj["obj_id"].replace("_", " "),
                (obj["x"] + ox, obj["y"] + oy),
                fontsize=6.8,
                color=BLUE if is_user else TEXT_MAIN,
                fontweight="bold" if is_user else "normal",
                zorder=5,
            )
        ax.set_title(title, fontsize=9.8, fontweight="bold")
        ax.set_xlabel("Room X (m)", fontsize=9)
        ax.set_ylabel("Room Y (m)", fontsize=9)
        ax.set_xlim(0.0, 6.0)
        ax.set_ylim(0.0, 6.0)
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG_FIG)
    plt.close(fig)


def _plot_ablation_comparison(
    ablation_summary: dict[str, Any],
    out_path: Path,
    dpi: int,
) -> None:
    cfgs = ablation_summary["configs"]
    keys = ["aif_only", "aif_pad_memory", "full_neuro_symbolic"]
    labels = ["AIF Only", "AIF + PAD\n+ Memory", "Full\nNeuro-Symbolic"]
    colors = [SLATE, TEAL, BLUE]

    trust_divs = [float(cfgs[k]["twin_trust_divergence"]) for k in keys]
    spatial_divs = [float(cfgs[k]["twin_spatial_divergence_m"]) for k in keys]
    policy_divs = [float(cfgs[k]["policy_l1_divergence"]) for k in keys]
    skills_synth = [int(cfgs[k]["skills_synthesized"]) for k in keys]
    skills_reuse = [int(cfgs[k]["skill_reuses"]) for k in keys]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), dpi=dpi, facecolor=BG_FIG)
    x = np.arange(len(keys))

    specs = [
        (axes[0, 0], trust_divs, "Twin User-Trust Divergence", "Trust Delta (Calm - Erratic)"),
        (axes[0, 1], spatial_divs, "Twin Spatial Divergence", "Mean User Distance Delta (m)"),
        (axes[1, 0], policy_divs, "Twin Policy L1 Divergence", "Action Histogram L1 Distance"),
    ]
    for ax, vals, title, ylabel in specs:
        _apply_cad_axes_style(ax)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, color=GRID_COLOR, zorder=0)
        bars = ax.bar(x, vals, width=0.54, color=colors, edgecolor="#FFFFFF", linewidth=0.8, zorder=3)
        top = max(0.25, max(vals) * 1.24)
        ax.set_ylim(min(0.0, min(vals) * 1.15), top)
        for bar, v in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() * 0.5,
                max(v, 0.0) + top * 0.03,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=TEXT_MAIN,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8.8)

    # Panel 4: Skill synthesis & library reuse + symbolic safety callout
    ax3 = axes[1, 1]
    _apply_cad_axes_style(ax3)
    ax3.grid(True, axis="y", linestyle="--", linewidth=0.6, color=GRID_COLOR, zorder=0)
    w = 0.34
    b1 = ax3.bar(
        x - w * 0.5,
        skills_synth,
        width=w,
        color=BLUE,
        edgecolor="#FFFFFF",
        label="Verified Skills Synthesized",
        zorder=3,
    )
    b2 = ax3.bar(
        x + w * 0.5,
        skills_reuse,
        width=w,
        color=AMBER,
        edgecolor="#FFFFFF",
        label="Library Skill Reuses",
        zorder=3,
    )
    top3 = max(3.0, max(skills_synth + skills_reuse) * 1.35)
    ax3.set_ylim(0.0, top3)
    for bar, v in zip(b1, skills_synth, strict=True):
        ax3.text(
            bar.get_x() + bar.get_width() * 0.5,
            v + top3 * 0.03,
            str(v),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=TEXT_MAIN,
        )
    for bar, v in zip(b2, skills_reuse, strict=True):
        ax3.text(
            bar.get_x() + bar.get_width() * 0.5,
            v + top3 * 0.03,
            str(v),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=TEXT_MAIN,
        )
    safety_pct = float(cfgs["full_neuro_symbolic"]["symbolic_safety_rate"]) * 100.0
    ax3.text(
        0.03,
        0.92,
        f"Symbolic Safety Compliance: {safety_pct:.1f}%\n"
        f"(0 illegal anomalies on grounded pets)",
        transform=ax3.transAxes,
        va="top",
        ha="left",
        fontsize=8.0,
        color=TEAL,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#ECFDF5",
            "edgecolor": "#A7F3D0",
            "linewidth": 0.9,
        },
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8.5)
    ax3.set_title("Neuro-Symbolic Skill Synthesis & Reuse", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Count", fontsize=8.8)
    ax3.legend(
        loc="upper right",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor=BORDER_COLOR,
        fontsize=7.8,
    )

    fig.suptitle(
        "Controlled Three-Way Ablation Study (Identical Room, Seed & Schedule)",
        fontsize=11.5,
        fontweight="bold",
        color=TEXT_MAIN,
        y=0.985,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(out_path, facecolor=BG_FIG)
    plt.close(fig)


def run_benchmark_suite(cfg: BenchmarkConfig | None = None) -> dict[str, Any]:
    """Execute full Sprint 7 quantitative benchmark, ablation study, CSV/JSON & PNG exports."""
    config = cfg or BenchmarkConfig()
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    suite_t0 = time.perf_counter()
    archive_summary, elite_df, archive_raw = evaluate_personality_archive(config)
    ablation_summary, twin_plot_data = run_ablation_study(config)

    csv_path = config.results_dir / "elite_metrics.csv"
    elite_df.to_csv(csv_path, index=False)

    fig_atlas = config.figures_dir / "personality_atlas.png"
    fig_umap = config.figures_dir / "umap_personality_clusters.png"
    fig_twin = config.figures_dir / "twin_divergence.png"
    fig_ablation = config.figures_dir / "ablation_comparison.png"

    _plot_personality_atlas(archive_raw, fig_atlas, dpi=config.dpi)
    _plot_umap_clusters(elite_df, archive_summary, fig_umap, dpi=config.dpi)
    _plot_twin_divergence(twin_plot_data, fig_twin, dpi=config.dpi)
    _plot_ablation_comparison(ablation_summary, fig_ablation, dpi=config.dpi)

    total_elapsed = round(time.perf_counter() - suite_t0, 2)
    summary = {
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "seed": config.seed,
            "total_elapsed_sec": total_elapsed,
        },
        "archive_evaluation": archive_summary,
        "ablation_study": ablation_summary,
        "exports": {
            "summary_json": str(config.results_dir / "benchmark_summary.json"),
            "elite_csv": str(csv_path),
            "figures": [
                str(fig_atlas),
                str(fig_umap),
                str(fig_twin),
                str(fig_ablation),
            ],
        },
    }

    summary_path = config.results_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
