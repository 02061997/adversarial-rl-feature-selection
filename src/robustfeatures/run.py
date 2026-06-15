import argparse

import matplotlib.pyplot as plt
import pandas as pd

from .artifacts import environment, output_dir, publish_latest, save
from .core import ENVIRONMENTS, PUBLISHED_RESULTS, benchmark_diagnostics, evaluate


def diagnostics_markdown(diagnostics: pd.DataFrame) -> str:
    rows = [
        "| Environment | Local best | Published best | Agreement | Local accuracy range |",
        "|---|---|---|---:|---:|",
    ]
    for _, row in diagnostics.iterrows():
        rows.append(
            "| {environment} | {local_best_model} + {local_best_metric} ({local_best_accuracy:.3f}) "
            "| {published_best_model} + {published_best_metric} ({published_best_accuracy:.3f}) "
            "| {ranking_agrees_with_published} | {local_accuracy_range:.3f} |".format(**row)
        )
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = output_dir(args.smoke)
    records, predictions = evaluate(samples=900 if args.smoke else 3600)
    frame = pd.DataFrame(records)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    summary = (
        frame.groupby(["environment", "model", "metric"]).accuracy.mean().reset_index()
    )
    diagnostics = benchmark_diagnostics(records)
    summary.to_csv(out / "summary.csv", index=False)
    diagnostics.to_csv(out / "benchmark_diagnostics.csv", index=False)
    pivot = summary.pivot_table(index=["environment", "model"], columns="metric", values="accuracy")
    ax = pivot.plot.bar(figsize=(10, 5), ylim=(0, 1), ylabel="Detection accuracy")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "imposter_detection.png", dpi=180)
    plt.close(ax.figure)
    save(out / "metrics.json", summary.to_dict(orient="records"))
    save(
        out / "statistical_tests.json",
        {
            "published_results": PUBLISHED_RESULTS,
            "published_results_reproduced": False,
            "local_reference_results": summary.to_dict(orient="records"),
            "benchmark_diagnostics": diagnostics.to_dict(orient="records"),
            "not_run": [
                {
                    "experiment": "PPO Lunar Lander 200000-step trajectory extraction",
                    "reason": "Original trained policy and trajectories are unavailable.",
                },
                {
                    "experiment": "ARS Bipedal Walker 1000-iteration trajectory extraction",
                    "reason": "Original trained policy and trajectories are unavailable.",
                },
            ],
        },
    )
    save(out / "environment.json", environment())
    save(
        out / "data_manifest.json",
        {
            "source": "deterministic trajectory fixtures",
            "environments": {
                name: {
                    "gymnasium_name": spec.name,
                    "dimensions": spec.dimensions,
                    "imposter_counts": spec.imposter_counts,
                    "paper_training_protocol": spec.training_protocol,
                }
                for name, spec in ENVIRONMENTS.items()
            },
            "original_training_trajectories_included": False,
        },
    )
    save(out / "config.yaml", {"noise": ["gaussian", "uniform"], "classifiers": 4})
    (out / "BENCHMARK_NOTE.md").write_text(
        "# Public Benchmark Diagnostics\n\n"
        "This deterministic fixture is a reconstruction stress test, not the original PPO/ARS "
        "experiment. `ranking_agrees_with_published=false` means the synthetic trajectory "
        "distribution does not reproduce the published best model/metric ordering.\n\n"
        + diagnostics_markdown(diagnostics)
        + "\n"
    )
    (out / "run.log").write_text("completed\n")
    if not args.smoke:
        publish_latest(out)
    print(out)


if __name__ == "__main__":
    main()
