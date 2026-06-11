# Adversarial RL Feature Selection

Reference reconstruction of feature-selection ideas described in
*Imposter Injection: Learning to Select Features in Reinforcement Learning*.
This is not the original paper code. It provides a reproducible CartPole
observation-noise benchmark comparing entropy, KL divergence, mutual
information, and all-feature baselines.

```bash
uv sync && make test && make reproduce-results
```
