# Imposter Injection Feature Detection

Paper-faithful public reconstruction of *Imposter Injection: Learning to Select
Features in Reinforcement Learning* (KSE 2024).

The repository implements appended Gaussian/uniform imposters, Lunar Lander
and Bipedal Walker dimensional profiles, entropy/joint-entropy/KL feature
descriptors, the paper's return-deviation gates, and Naive/RF/KNN/SVM
detectors. Original trained PPO/ARS policies and trajectories were unavailable,
so local experiments use deterministic trajectory fixtures.

Full runs also emit `benchmark_diagnostics.csv` and `BENCHMARK_NOTE.md`, which
compare the local fixture's best model/metric ordering with the published best
rows. This makes the reconstruction gap inspectable instead of implying that
synthetic rankings are paper results.

```bash
uv sync
make test
make reproduce-smoke
make reproduce-results
```
