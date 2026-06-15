# Results

## Published paper results

The paper reports Lunar Lander's best detector as Random Forest with entropy
(97.05%) and Bipedal Walker's best detector as SVM with entropy (97.77%).
Complete published tables are stored in
`reports/latest/statistical_tests.json`.

## Local reference results

`make reproduce-results` completed on June 15, 2026 across both dimensional
profiles, every paper imposter count, Gaussian and uniform noise, three
information metrics, and four classifiers.

The fixture does **not** reproduce the paper's metric ordering: KL was strongest
for Lunar Lander-like synthetic trajectories, while the Bipedal profile was
less differentiated. This is a useful negative result showing that metric
ranking depends on the genuine trajectory distribution. Exact PPO 200,000-step
and ARS 1,000-iteration policy experiments remain `NOT_RUN` and are recorded
explicitly in `reports/latest/statistical_tests.json`.

The latest run also writes `reports/latest/benchmark_diagnostics.csv` and
`reports/latest/BENCHMARK_NOTE.md`, which compare local synthetic rankings
against the published best rows and mark any ordering mismatch directly.
