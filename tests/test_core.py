import numpy as np

from robustfeatures.core import (
    ENVIRONMENTS,
    deviation_gate,
    entropy,
    feature_descriptors,
    inject_imposters,
    joint_entropy,
    kl_divergence,
)


def test_imposters_are_appended_and_labeled():
    clean = np.ones((100, 8))
    injected, labels = inject_imposters(clean, 4, "uniform", 7)
    assert injected.shape == (100, 12)
    assert labels.tolist() == [-1] * 8 + [1] * 4


def test_information_measures_are_finite():
    rng = np.random.default_rng(7)
    left, right = rng.normal(size=1000), rng.uniform(size=1000)
    assert entropy(left) > 0
    assert joint_entropy(left, right) > 0
    assert np.isfinite(kl_divergence(left, right))


def test_algorithm_feature_matrix_and_gate():
    clean = np.random.default_rng(7).normal(size=(600, 8))
    injected, labels = inject_imposters(clean, 2, "gaussian", 7)
    descriptors = feature_descriptors(injected, labels)
    assert set(["mean_entropy", "centered_entropy", "entropy", "joint_entropy", "kl"]).issubset(
        descriptors
    )
    assert deviation_gate(301, ENVIRONMENTS["lunar_lander"])
    assert not deviation_gate(150, ENVIRONMENTS["lunar_lander"])
