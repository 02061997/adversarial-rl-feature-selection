import argparse

import matplotlib.pyplot as plt
import pandas as pd

from .artifacts import environment, output_dir, save
from .core import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = output_dir(args.smoke)
    records = evaluate(
        samples=800 if args.smoke else 6000, seeds=(7,) if args.smoke else (1, 7, 19)
    )
    frame = pd.DataFrame(records)
    frame.to_parquet(out / "predictions.parquet", index=False)
    summary = frame.groupby(["method", "scale"]).accuracy.agg(["mean", "std"]).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    pivot = summary.pivot(index="scale", columns="method", values="mean")
    ax = pivot.plot(marker="o", figsize=(8, 4), ylabel="Action accuracy")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "robustness.png", dpi=180)
    plt.close(ax.figure)
    save(out / "metrics.json", summary.to_dict(orient="records"))
    save(out / "statistical_tests.json", {"seeds": sorted(frame.seed.unique().tolist())})
    save(out / "environment.json", environment())
    save(
        out / "data_manifest.json",
        {"environment": "Gymnasium CartPole-v1", "samples": 800 if args.smoke else 6000},
    )
    save(out / "config.yaml", {"noise_scales": [0.1, 0.25, 0.5]})
    (out / "run.log").write_text("completed\n")
    print(out)


if __name__ == "__main__":
    main()
