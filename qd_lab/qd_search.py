from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import grid_archive_heatmap

from qd_lab.behavior_descriptors import compute_behavior_batch, infer_archetype_label
from qd_lab.jax_rollout import evaluate_genomes_batch
from sim_core.genome import GENOME_FIELDS, TraitGenome
from sim_core.genome_mapper import ACTION_NAMES


@dataclass(frozen=True)
class QDSearchConfig:
    dims: tuple[int, int] = (10, 10)
    ranges: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 0.65))
    iterations: int = 35
    num_emitters: int = 4
    batch_size: int = 16
    sigma0: float = 0.28
    learning_rate: float = 0.05
    rollout_ticks: int = 160
    ambient_temp_c: float = 20.0
    seed: int = 42


@dataclass(frozen=True)
class QDSearchResult:
    result_archive: GridArchive
    coverage: float
    qd_score: float
    num_elites: int
    total_evaluations: int
    elapsed_sec: float


def _build_emitters(archive: GridArchive, cfg: QDSearchConfig) -> list[EvolutionStrategyEmitter]:
    dim = TraitGenome.DIM
    sig = inspect.signature(EvolutionStrategyEmitter.__init__).parameters
    bound_kwargs: dict[str, object] = (
        {
            "lower_bounds": np.zeros(dim, dtype=np.float32),
            "upper_bounds": np.ones(dim, dtype=np.float32),
        }
        if "lower_bounds" in sig
        else {"bounds": [(0.0, 1.0)] * dim}
    )

    rng = np.random.default_rng(cfg.seed)
    emitters: list[EvolutionStrategyEmitter] = []
    for idx in range(cfg.num_emitters):
        x0 = rng.uniform(0.2, 0.8, size=dim).astype(np.float32)
        emitters.append(
            EvolutionStrategyEmitter(
                archive=archive,
                x0=x0,
                sigma0=cfg.sigma0,
                ranker="imp",
                selection_rule="mu",
                restart_rule="basic",
                batch_size=cfg.batch_size,
                seed=cfg.seed + idx * 17,
                **bound_kwargs,
            )
        )
    return emitters


def run_cma_mae(cfg: QDSearchConfig | None = None, verbose: bool = True) -> QDSearchResult:
    """Illuminate the 2D personality space using pyribs CMA-MAE over batched JAX rollouts."""
    config = cfg or QDSearchConfig()
    start_time = time.perf_counter()

    # CMA-MAE uses a discounted-threshold archive for emitters and a standard result archive
    archive = GridArchive(
        solution_dim=TraitGenome.DIM,
        dims=list(config.dims),
        ranges=list(config.ranges),
        learning_rate=config.learning_rate,
        threshold_min=0.0,
        seed=config.seed,
    )
    result_archive = GridArchive(
        solution_dim=TraitGenome.DIM,
        dims=list(config.dims),
        ranges=list(config.ranges),
        seed=config.seed,
    )

    emitters = _build_emitters(archive, config)
    scheduler = Scheduler(archive, emitters, result_archive=result_archive)

    # Seed iteration 0 with diverse presets + uniform Latin-style random sweep
    rng = np.random.default_rng(config.seed)
    presets = [
        TraitGenome.preset_hairless_sphynx().to_vector(),
        TraitGenome.preset_staring_observer().to_vector(),
        TraitGenome.preset_chaotic_gremlin().to_vector(),
        TraitGenome.preset_impossible_cat().to_vector(),
    ]
    n_seed = max(32, config.num_emitters * config.batch_size)
    seed_random = np.stack(
        [TraitGenome.random(rng, anomaly_chance=0.15).to_vector() for _ in range(n_seed - len(presets))]
    )
    seed_batch = np.vstack([np.stack(presets), seed_random]).astype(np.float32)

    seed_metrics = evaluate_genomes_batch(
        seed_batch,
        ticks=config.rollout_ticks,
        seed=config.seed,
        ambient_temp_c=config.ambient_temp_c,
    )
    seed_beh = compute_behavior_batch(seed_metrics, ticks=config.rollout_ticks)
    result_archive.add(seed_batch, seed_beh.objectives, seed_beh.measures_2d())
    archive.add(seed_batch, seed_beh.objectives, seed_beh.measures_2d())
    total_evals = int(seed_batch.shape[0])

    # Main CMA-MAE loop
    for itr in range(1, config.iterations + 1):
        solutions = scheduler.ask()
        metrics = evaluate_genomes_batch(
            solutions,
            ticks=config.rollout_ticks,
            seed=config.seed + itr * 101,
            ambient_temp_c=config.ambient_temp_c,
        )
        beh = compute_behavior_batch(metrics, ticks=config.rollout_ticks)
        scheduler.tell(beh.objectives, beh.measures_2d())
        total_evals += int(solutions.shape[0])

        if verbose and (itr == 1 or itr % 5 == 0 or itr == config.iterations):
            stats = result_archive.stats
            print(
                f"Iter {itr:3d}/{config.iterations} | "
                f"evals={total_evals:5d} | "
                f"elites={stats.num_elites:3d}/{result_archive.cells} | "
                f"coverage={stats.coverage * 100:5.1f}% | "
                f"QD-score={stats.qd_score:6.2f}"
            )

    elapsed = time.perf_counter() - start_time
    stats = result_archive.stats
    return QDSearchResult(
        result_archive=result_archive,
        coverage=float(stats.coverage),
        qd_score=float(stats.qd_score),
        num_elites=int(stats.num_elites),
        total_evaluations=total_evals,
        elapsed_sec=elapsed,
    )


def export_personality_archive(
    result: QDSearchResult,
    cfg: QDSearchConfig,
    json_path: Path,
    heatmap_path: Path | None = None,
) -> Path:
    """Export discovered elites to personality_archive.json and optionally plot a heatmap."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    data = result.result_archive.data()
    solutions = np.asarray(data["solution"], dtype=np.float32)
    objectives = np.asarray(data["objective"], dtype=np.float32)
    measures = np.asarray(data["measures"], dtype=np.float32)
    indices = np.asarray(data["index"], dtype=int)

    # Re-evaluate all elites deterministically to store rich 3D descriptors & action histograms
    metrics = evaluate_genomes_batch(
        solutions,
        ticks=cfg.rollout_ticks,
        seed=cfg.seed,
        ambient_temp_c=cfg.ambient_temp_c,
    )
    beh = compute_behavior_batch(metrics, ticks=cfg.rollout_ticks)

    elites: list[dict[str, object]] = []
    for i in range(solutions.shape[0]):
        genome = TraitGenome.from_vector(solutions[i])
        ratios = metrics.action_ratios[i]
        dominant_idx = int(np.argmax(ratios))
        label = infer_archetype_label(
            genome=genome,
            action_ratios=ratios,
            disruption_rate=float(beh.disruption_rate[i]),
            activity_entropy=float(beh.activity_entropy[i]),
        )
        elites.append(
            {
                "cell_index": int(indices[i]),
                "archetype": label,
                "dominant_action": ACTION_NAMES[dominant_idx],
                "objective": round(float(objectives[i]), 4),
                "measures": {
                    "human_engagement": round(float(measures[i, 0]), 4),
                    "activity_entropy": round(float(measures[i, 1]), 4),
                    "disruption_rate": round(float(beh.disruption_rate[i]), 4),
                },
                "metrics": {
                    "mean_warmth": round(float(metrics.mean_warmth[i]), 4),
                    "mean_vfe": round(float(metrics.mean_vfe[i]), 4),
                    "water_glass_intact": bool(metrics.water_glass_intact[i]),
                    "action_ratios": {
                        name: round(float(ratios[j]), 4) for j, name in enumerate(ACTION_NAMES)
                    },
                },
                "genome": {
                    field: round(float(val), 4)
                    for field, val in zip(GENOME_FIELDS, solutions[i], strict=True)
                },
            }
        )

    payload = {
        "metadata": {
            "algorithm": "CMA-MAE",
            "grid_dims": list(cfg.dims),
            "measure_names": ["human_engagement", "activity_entropy"],
            "measure_ranges": [list(r) for r in cfg.ranges],
            "coverage": round(result.coverage, 4),
            "qd_score": round(result.qd_score, 4),
            "num_elites": result.num_elites,
            "total_evaluations": result.total_evaluations,
            "elapsed_sec": round(result.elapsed_sec, 2),
        },
        "elites": elites,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if heatmap_path is not None:
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=150)
        grid_archive_heatmap(result.result_archive, ax=ax, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(
            f"PET-AI Personality Atlas (CMA-MAE | Coverage: {result.coverage * 100:.1f}% | "
            f"Elites: {result.num_elites}/{result.result_archive.cells})",
            fontsize=11,
            pad=10,
        )
        ax.set_xlabel("Human Engagement (Solitary -> Social / Observant)")
        ax.set_ylabel("Normalized Activity Entropy (Specialist -> Explorer)")
        fig.tight_layout()
        fig.savefig(heatmap_path)
        plt.close(fig)

    return json_path