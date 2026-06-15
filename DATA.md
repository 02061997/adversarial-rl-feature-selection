# Data Documentation

## Original Paper Data

The KSE 2024 paper used trajectories from trained PPO and ARS agents in Lunar
Lander and Bipedal Walker. Those trained policies, rollouts, and training
artifacts are not available in this public reconstruction.

## Included Fixture

Executable experiments use deterministic synthetic trajectory fixtures that
match the paper's feature-dimensional profiles and imposter-injection setup.
They are suitable for validating feature descriptors, return-deviation gates,
detector code, artifact generation, and tests.

## Redistribution Boundary

No private trajectories, trained policies, model checkpoints, simulator logs,
or unpublished experiment artifacts are redistributed. Published paper results
are stored separately from local reconstruction results.

## Generated Artifacts

`make reproduce-results` writes `reports/latest/`, including:

- `metrics.json`
- `predictions.parquet`
- `statistical_tests.json`
- `summary.csv`
- `benchmark_diagnostics.csv`
- `BENCHMARK_NOTE.md`
- `imposter_detection.png`

## NOT_RUN Limitations

The original PPO 200,000-step and ARS 1,000-iteration trajectory experiments
remain `NOT_RUN`. The public fixture does not claim to reproduce the original
paper's trained-policy trajectory distribution.
