# Imposter Injection Feature Detection

Paper-faithful public reconstruction of *Imposter Injection: Learning to Select
Features in Reinforcement Learning* (KSE 2024).

The repository implements appended Gaussian/uniform imposters, Lunar Lander
and Bipedal Walker dimensional profiles, entropy/joint-entropy/KL feature
descriptors, the paper's return-deviation gates, and Naive/RF/KNN/SVM
detectors. Original trained PPO/ARS policies and trajectories were unavailable,
so local experiments use deterministic trajectory fixtures.

```bash
uv sync
make test
make reproduce-smoke
make reproduce-results
```
