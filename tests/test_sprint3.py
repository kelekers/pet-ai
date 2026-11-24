from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qd_lab.behavior_descriptors import compute_behavior_batch
from qd_lab.jax_rollout import evaluate_genomes_batch
from qd_lab.qd_search import QDSearchConfig, export_personality_archive, run_cma_mae
from sim_core.genome import TraitGenome


def test_jax_batched_rollout_matches_preset_divergence() -> None:
    presets = np.stack(
        [
            TraitGenome.preset_hairless_sphynx().to_vector(),
            TraitGenome.preset_staring_observer().to_vector(),
            TraitGenome.preset_chaotic_gremlin().to_vector(),
        ]
    )
    metrics = evaluate_genomes_batch(presets, ticks=160, seed=42)
    beh = compute_behavior_batch(metrics, ticks=160)

    assert metrics.action_ratios.shape == (3, 8)
    assert np.all(np.isfinite(beh.objectives))
    assert np.all((beh.objectives >= 0.0) & (beh.objectives <= 1.0))

    # Chaotic gremlin must break the glass and exceed staring observer disruption
    assert not metrics.water_glass_intact[2]
    assert beh.disruption_rate[2] > beh.disruption_rate[1] + 0.15


def test_cma_mae_achieves_high_coverage_and_exports_valid_archive(tmp_path: Path) -> None:
    cfg = QDSearchConfig(
        dims=(8, 8),
        ranges=((0.0, 1.0), (0.0, 0.65)),
        iterations=18,
        num_emitters=4,
        batch_size=12,
        rollout_ticks=120,
        seed=42,
    )
    result = run_cma_mae(cfg, verbose=False)

    assert result.coverage >= 0.60
    assert result.qd_score > 20.0

    json_out = tmp_path / "personality_archive.json"
    png_out = tmp_path / "personality_heatmap.png"
    export_personality_archive(result, cfg, json_path=json_out, heatmap_path=png_out)

    assert json_out.exists()
    assert png_out.exists() and png_out.stat().st_size > 1000

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["metadata"]["num_elites"] == len(payload["elites"])
    assert len(payload["elites"]) >= 39

    # Verify exported genome dict round-trips cleanly into TraitGenome
    first_elite = payload["elites"][0]
    reconstructed = TraitGenome(**first_elite["genome"])
    assert reconstructed.to_vector().shape == (18,)