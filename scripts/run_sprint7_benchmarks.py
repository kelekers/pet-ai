from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.eval_suite import BenchmarkConfig, run_benchmark_suite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PET-AI Sprint 7 Quantitative Research Benchmarks and Ablation Suite"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/archives/personality_archive.json"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--eval-ticks", type=int, default=500)
    parser.add_argument("--twin-ticks", type=int, default=240)
    parser.add_argument("--neuro-ticks", type=int, default=160)
    parser.add_argument("--max-elites", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        archive_path=args.archive,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        eval_ticks=args.eval_ticks,
        twin_ticks=args.twin_ticks,
        neuro_ticks=args.neuro_ticks,
        max_elites=args.max_elites,
        seed=args.seed,
    )

    print("=== Running PET-AI Sprint 7 Quantitative Benchmark Suite ===")
    summary = run_benchmark_suite(cfg)
    arch = summary["archive_evaluation"]
    abl = summary["ablation_study"]["configs"]

    print(
        f"\n[Archive Evaluation] elites={arch['evaluated_elites']} "
        f"(coverage={arch['coverage'] * 100:.1f}%, QD={arch['qd_score']:.2f})"
    )
    ent = arch["action_entropy_bits"]
    print(
        f"  Entropy (bits) : mean={ent['mean']:.3f}, std={ent['std']:.3f}, "
        f"range=[{ent['min']:.3f}, {ent['max']:.3f}]"
    )
    print(
        f"  Silhouette     : policy_clusters={arch['dominant_action_silhouette_score']} "
        f"| archetypes={arch['silhouette_score']} "
        f"({arch['silhouette_clusters_used']} clusters, {arch['silhouette_samples_used']} elites)"
    )
    print(
        f"  Pairwise L2    : behavioral={arch['mean_pairwise_behavioral_l2']:.3f}, "
        f"policy={arch['mean_pairwise_policy_l2']:.3f}"
    )
    print(f"  JAX Throughput : {arch['throughput_ticks_per_sec']:.0f} ticks/sec")

    print("\n[Three-Way Ablation]")
    for key in ("aif_only", "aif_pad_memory", "full_neuro_symbolic"):
        c = abl[key]
        print(
            f"  {c['label']:<20} | trust_div={c['twin_trust_divergence']:+.3f} | "
            f"spatial_div={c['twin_spatial_divergence_m']:+.3f}m | "
            f"policy_L1={c['policy_l1_divergence']:.3f} | "
            f"skills={c['skills_synthesized']} (reuses={c['skill_reuses']}) | "
            f"safety={c['symbolic_safety_rate'] * 100:.1f}%"
        )

    print(f"\nExported summary : {summary['exports']['summary_json']}")
    print(f"Exported CSV     : {summary['exports']['elite_csv']}")
    for fig in summary["exports"]["figures"]:
        print(f"Exported figure  : {fig}")
    print(f"Completed in {summary['runtime']['total_elapsed_sec']:.2f}s.")


if __name__ == "__main__":
    main()
