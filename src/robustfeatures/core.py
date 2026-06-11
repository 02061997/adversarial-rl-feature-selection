from __future__ import annotations

import gymnasium as gym
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def heuristic_action(observation: np.ndarray) -> int:
    position, velocity, angle, angular_velocity = observation
    score = angle + 0.4 * angular_velocity + 0.08 * position + 0.02 * velocity
    return int(score > 0)


def collect(samples: int = 6000, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    env = gym.make("CartPole-v1")
    observation, _ = env.reset(seed=seed)
    x, y = [], []
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        action = heuristic_action(observation)
        x.append(observation.copy())
        y.append(action)
        observation, _, terminated, truncated, _ = env.step(int(rng.integers(0, 2)))
        if terminated or truncated:
            observation, _ = env.reset()
    env.close()
    return np.asarray(x), np.asarray(y)


def attack(x: np.ndarray, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    attacked = x.copy()
    attacked += rng.normal(0, scale * np.std(x, axis=0, ddof=1), x.shape)
    glitches = rng.random(x.shape) < 0.02
    attacked += glitches * rng.normal(0, 5 * scale * np.std(x, axis=0, ddof=1), x.shape)
    return attacked


def rank_features(x: np.ndarray, y: np.ndarray, attacked: np.ndarray) -> dict[str, np.ndarray]:
    bins = 20
    entropy = []
    divergence = []
    for column in range(x.shape[1]):
        limits = np.quantile(x[:, column], np.linspace(0, 1, bins + 1))
        limits = np.unique(limits)
        clean_hist, _ = np.histogram(x[:, column], bins=limits, density=True)
        attack_hist, _ = np.histogram(attacked[:, column], bins=limits, density=True)
        clean_hist = clean_hist + 1e-9
        attack_hist = attack_hist + 1e-9
        clean_hist /= clean_hist.sum()
        attack_hist /= attack_hist.sum()
        entropy.append(-(clean_hist * np.log(clean_hist)).sum())
        divergence.append((clean_hist * np.log(clean_hist / attack_hist)).sum())
    return {
        "entropy": np.argsort(entropy)[::-1],
        "kl_divergence": np.argsort(divergence),
        "mutual_information": np.argsort(mutual_info_classif(x, y, random_state=7))[::-1],
        "all": np.arange(x.shape[1]),
    }


def evaluate(samples=6000, seeds=(1, 7, 19), scales=(0.1, 0.25, 0.5)) -> list[dict]:
    x, y = collect(samples)
    split = int(len(x) * 0.7)
    records = []
    for scale in scales:
        attacked = attack(x, scale, 100)
        rankings = rank_features(x[:split], y[:split], attacked[:split])
        for method, ranking in rankings.items():
            selected = ranking if method == "all" else ranking[:2]
            for seed in seeds:
                model = LogisticRegression(max_iter=2000, random_state=seed)
                model.fit(x[:split, selected], y[:split])
                prediction = model.predict(attack(x[split:, selected], scale, seed))
                records.append(
                    {
                        "method": method,
                        "scale": scale,
                        "seed": seed,
                        "features": selected.tolist(),
                        "accuracy": accuracy_score(y[split:], prediction),
                    }
                )
    return records
