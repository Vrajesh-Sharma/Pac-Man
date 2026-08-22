import pickle
import random
from collections import defaultdict

import numpy as np

from .base_agent import BaseAgent


class MonteCarloAgent(BaseAgent):
    """Every-visit Monte Carlo control. Learns only from *complete* episodes:
    plays a full episode with an epsilon-greedy policy, then walks the
    episode backwards accumulating the discounted return G and updates
    Q(s,a) towards the incremental average of all observed returns.
    No bootstrapping -- purely from actual outcomes, which tends to make it
    slower but sometimes more stable than TD methods on noisy rewards."""

    name = "montecarlo"

    def __init__(self, n_actions=4, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
        self.N = defaultdict(lambda: np.zeros(n_actions, dtype=np.int32))
        self._episode = []  # list of (s, a, r)

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def q_values(self, state):
        return np.asarray(self.Q[state], dtype=np.float32)

    def record(self, s, a, r):
        self._episode.append((s, a, r))

    def on_episode_end(self):
        G = 0.0
        for s, a, r in reversed(self._episode):
            G = r + self.gamma * G
            self.N[s][a] += 1
            self.Q[s][a] += (G - self.Q[s][a]) / self.N[s][a]
        self._episode = []

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"Q": dict(self.Q), "N": dict(self.N)}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32), data["Q"])
        self.N = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.int32), data["N"])
