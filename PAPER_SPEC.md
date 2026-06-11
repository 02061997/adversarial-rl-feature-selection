# Paper Specification

Source: *Imposter Injection: Learning to Select Features in Reinforcement
Learning*, KSE 2024, DOI `10.1109/KSE63888.2024.11063527`.

## Implemented specification

- Lunar Lander profile: 8 original features and 1, 2, 3, or 4 imposters.
- Bipedal Walker profile: 24 original features and 1, 2, 4, or 8 imposters.
- Appended Gaussian and uniform imposter features.
- Shannon entropy, pairwise joint entropy, and KL-divergence descriptors.
- Algorithm-1 feature matrix: mean entropy, centered information metric, bias.
- Naive linear, Random Forest, KNN (`k=5`), and SVM detection models.
- Paper return-deviation gates and published accuracy tables.

The original PPO/ARS trajectories and trained policies were unavailable.
Executable results therefore use deterministic trajectory fixtures with the
paper's exact dimensions and injection protocol. Published and local results
are stored separately.
