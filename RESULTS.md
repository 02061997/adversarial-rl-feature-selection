# Verified Results

## Provenance

This is a reconstructed reference implementation. It is not the original paper
code. Published-paper claims remain `NOT_RUN` until the manuscript and exact
experimental protocol are reviewed.

## Local reproduction

`make reproduce-results` completed locally on June 11, 2026.

| Feature method | Noise scale | Mean robustness score | Std. dev. |
|---|---:|---:|---:|
| All | 0.10 | 0.939 | 0.004 |
| All | 0.25 | 0.891 | 0.007 |
| All | 0.50 | 0.826 | 0.010 |
| Entropy | 0.10 | 0.824 | 0.003 |
| Entropy | 0.25 | 0.813 | 0.005 |
| Entropy | 0.50 | 0.776 | 0.003 |
| Kl Divergence | 0.10 | 0.816 | 0.004 |
| Kl Divergence | 0.25 | 0.804 | 0.004 |
| Kl Divergence | 0.50 | 0.771 | 0.009 |
| Mutual Information | 0.10 | 0.923 | 0.003 |
| Mutual Information | 0.25 | 0.878 | 0.004 |
| Mutual Information | 0.50 | 0.809 | 0.005 |

Mutual-information selection is consistently stronger than entropy and
KL-divergence selection in this synthetic perturbation benchmark, while using
all features remains strongest. These results validate this code path only.
