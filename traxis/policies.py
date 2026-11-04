"""Nonlearning acquisition baselines; each sees only an Observation."""

import numpy as np

from traxis.bank import Observation


class RandomPolicy:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def choose(self, observation: Observation, reconstruction=None) -> int:
        available = np.flatnonzero(~observation.observed)
        if not len(available):
            raise ValueError("all records have been acquired")
        return int(self.rng.choice(available))


class FixedPolicy:
    """Opposing paths first, interleaving frequencies across tied path ranks.

    For the corner/edge eight-PZT layout this begins with opposite corner
    diagonals, then the opposing edge midpoints. Ring layouts use their
    longest chords. Tie breaks are stable catalog indices.
    """

    def choose(self, observation: Observation, reconstruction=None) -> int:
        frequencies = sorted({a.frequency_hz for a in observation.actions})
        ranking = []
        for i, a in enumerate(observation.actions):
            if observation.observed[i]:
                continue
            xy = observation.sensors_xy_m[[a.transmitter, a.receiver]]
            distance = float(np.linalg.norm(xy[1] - xy[0]))
            center = np.array([observation.plate.width_m, observation.plate.height_m]) / 2
            opposing = np.allclose(xy.mean(axis=0), center, rtol=0, atol=1e-9)
            ranking.append((0 if opposing else 1, -round(distance, 12), frequencies.index(a.frequency_hz), i))
        if not ranking:
            raise ValueError("all records have been acquired")
        return min(ranking)[-1]


def measurement_budgets(action_count: int, proposed=(10, 20, 40, 80, 160, 330)) -> tuple[int, ...]:
    if type(action_count) is not int or action_count < 1:
        raise ValueError("action count must be a positive integer")
    if any(type(v) is not int or v < 1 for v in proposed):
        raise ValueError("budgets must be positive integers")
    return tuple(sorted({v for v in proposed if v <= action_count} | {action_count}))
