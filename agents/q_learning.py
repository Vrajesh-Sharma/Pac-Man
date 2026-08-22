import pickle
import random
from collections import defaultdict

import numpy as np

from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Off-policy TD control: Q(s,a) += alpha * (r + gamma*max_a' Q(s',a') - Q(s,a))"""

    name = "qlearning"

    def __init__(self, n_actions=4, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def q_values(self, state):
        """Return this state's action-value estimates as a plain numpy
        array. Used by visualize_play.py to mask out wall-blocked actions
        during greedy playback without touching training behaviour."""
        return np.asarray(self.Q[state], dtype=np.float32)

    def update(self, s, a, r, s_next, done):
        best_next = 0.0 if done else np.max(self.Q[s_next])
        td_target = r + self.gamma * best_next
        self.Q[s][a] += self.alpha * (td_target - self.Q[s][a])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32), data)
