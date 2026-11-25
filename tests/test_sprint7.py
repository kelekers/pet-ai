from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from benchmarks.eval_suite import BenchmarkConfig, run_benchmark_suite
from bridge.ipc_server import create_app
from scripts.launch_demo import build_arg_parser, find_godot_executable


def test_benchmark_suite_exports_metrics_figures_and_ablation_invariants(tmp_path: Path) -> None:
    cfg = BenchmarkConfig(
        archive_path=Path("data/archives/personality_archive.json"),
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
        eval_ticks=70,
        twin_ticks=140,
        neuro_ticks=100,
        max_elites=14,
        seed=42,
        dpi=100,
    )
    summary = run_benchmark_suite(cfg)

    summary_path = tmp_path / "results" / "benchmark_summary.json"
    csv_path = tmp_path / "results" / "elite_metrics.csv"
    assert summary_path.exists()
    assert csv_path.exists()

    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["archive_evaluation"]["evaluated_elites"] == 14
    assert loaded["archive_evaluation"]["mean_pairwise_behavioral_l2"] > 0.0

    df = pd.read_csv(csv_path)
    assert len(df) == 14
    assert {"cell_index", "archetype", "action_entropy_bits", "umap_x", "umap_y"}.issubset(
        set(df.columns)
    )

    for fig_name in (
        "personality_atlas.png",
        "umap_personality_clusters.png",
        "twin_divergence.png",
        "ablation_comparison.png",
    ):
        fig_path = tmp_path / "figures" / fig_name
        assert fig_path.exists() and fig_path.stat().st_size > 2_000, f"Invalid figure: {fig_path}"

    # Verify core ablation invariants mathematically
    abl = summary["ablation_study"]["configs"]
    aif_only = abl["aif_only"]
    aif_pad = abl["aif_pad_memory"]
    full_ns = abl["full_neuro_symbolic"]

    assert aif_pad["twin_trust_divergence"] > aif_only["twin_trust_divergence"] + 0.70
    assert aif_pad["twin_spatial_divergence_m"] > aif_only["twin_spatial_divergence_m"] + 0.40
    assert full_ns["skills_synthesized"] >= 3
    assert full_ns["skill_reuses"] >= 1
    assert full_ns["illegal_anomaly_activations"] == 0
    assert full_ns["symbolic_safety_rate"] == 1.0


def test_inspector_route_serves_html_dashboard() -> None:
    app = create_app(preset="staring_observer", auto_tick=False, async_synthesis=False)
    with TestClient(app) as client:
        resp = client.get("/inspector")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "PET-AI Cognitive Telemetry Inspector" in resp.text
        assert "atlasGrid" in resp.text
        assert "spawn_archive_cell" in resp.text

        root_resp = client.get("/", follow_redirects=False)
        assert root_resp.status_code in (302, 307)
        assert root_resp.headers["location"] == "/inspector"


def test_launch_demo_cli_and_godot_discovery(tmp_path: Path, monkeypatch) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        ["--preset", "hairless_sphynx", "--hz", "10", "--port", "9001", "--no-godot", "--open-ui"]
    )
    assert args.preset == "hairless_sphynx"
    assert args.hz == 10.0
    assert args.port == 9001
    assert args.no_godot is True
    assert args.open_ui is True

    fake_godot = tmp_path / "Godot_v4.exe"
    fake_godot.write_text("stub", encoding="utf-8")
    assert find_godot_executable(fake_godot) == fake_godot
    assert find_godot_executable(tmp_path / "missing_godot.exe") is None

    monkeypatch.setenv("GODOT_PATH", str(fake_godot))
    assert find_godot_executable(None) == fake_godot
