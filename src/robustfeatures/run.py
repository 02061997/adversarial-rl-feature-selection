import argparse

import matplotlib.pyplot as plt
import pandas as pd

from .artifacts import environment, output_dir, save
from .core import ENVIRONMENTS, PUBLISHED_RESULTS, evaluate


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
    summary.to_csv(out / "summary.csv", index=False)
    pivot = summary.pivot_table(index=["environment", "model"], columns="metric", values="accuracy")
    ax = pivot.plot.bar(figsize=(10, 5), ylim=(0, 1), ylabel="Detection accuracy")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "imposter_detection.png", dpi=180)
    plt.close(ax.figure)
    save(out / "metrics.json", summary.to_dict(orient="records"))
    save(
        out / "statistical_tests.json",
        {"published_results": PUBLISHED_RESULTS, "published_results_reproduced": False},
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
    (out / "run.log").write_text("completed\n")
    print(out)


if __name__ == "__main__":
    main()
