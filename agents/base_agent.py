from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Common interface so train.py / visualize_play.py can treat every
    algorithm identically."""

    name = "base"

    @abstractmethod
    def select_action(self, state, training=True):
        ...

    @abstractmethod
    def save(self, path):
        ...

    @abstractmethod
    def load(self, path):
        ...

    def on_episode_end(self):
        """Hook for agents that learn only at episode end (e.g. Monte Carlo,
        REINFORCE). No-op by default."""
        pass

    def decay_epsilon(self):
        if hasattr(self, "epsilon"):
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
