# Public Benchmark Diagnostics

This deterministic fixture is a reconstruction stress test, not the original PPO/ARS experiment. `ranking_agrees_with_published=false` means the synthetic trajectory distribution does not reproduce the published best model/metric ordering.

| Environment | Local best | Published best | Agreement | Local accuracy range |
|---|---|---|---:|---:|
| bipedal_walker | knn + kl (0.943) | svm + entropy (0.978) | False | 0.028 |
| lunar_lander | knn + kl (0.983) | random_forest + entropy (0.971) | False | 0.114 |
