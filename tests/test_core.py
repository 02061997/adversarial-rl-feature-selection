import numpy as np
from robustfeatures.core import attack, heuristic_action


def test_heuristic_direction():
    assert heuristic_action(np.array([0, 0, 0.2, 0])) == 1
    assert heuristic_action(np.array([0, 0, -0.2, 0])) == 0


def test_attack_is_seeded():
    x = np.arange(40, dtype=float).reshape(10, 4)
    assert np.array_equal(attack(x, 0.2, 7), attack(x, 0.2, 7))
