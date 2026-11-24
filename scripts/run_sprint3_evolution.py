from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from qd_lab.qd_search import QDSearchConfig, export_personality_archive, run_cma_mae


def main() -> None:
    cfg = QDSearchConfig(
        dims=(10, 10),
        ranges=((0.0, 1.0), (0.0, 0.65)),
        iterations=35,
        num_emitters=4,
        batch_size=16,
        rollout_ticks=160,
        seed=42,
    )

    print("=== Sprint 3: Offline Personality Illumination (CMA-MAE + JAX vmap) ===\n")
    result = run_cma_mae(cfg, verbose=True)

    json_path = Path("data/archives/personality_archive.json")
    heatmap_path = Path("data/archives/personality_heatmap.png")
    export_personality_archive(result, cfg, json_path=json_path, heatmap_path=heatmap_path)

    archive_data = json.loads(json_path.read_text(encoding="utf-8"))
    archetype_counts = Counter(e["archetype"] for e in archive_data["elites"])

    print(f"\nFinished in {result.elapsed_sec:.2f}s ({result.total_evaluations} rollouts)")
    print(f"Coverage : {result.coverage * 100:.1f}% ({result.num_elites}/{result.result_archive.cells} cells)")
    print(f"QD-Score : {result.qd_score:.2f}")
    print(f"Saved    : {json_path} and {heatmap_path}\n")
    print("Discovered Archetype Distribution:")
    for name, count in archetype_counts.most_common():
        print(f"  - {name:22s}: {count:3d} elites")


if __name__ == "__main__":
    main()