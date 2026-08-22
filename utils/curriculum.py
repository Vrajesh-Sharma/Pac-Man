"""
Dynamic-difficulty curriculum: watches a rolling *win rate* (not raw score)
and only bumps env.difficulty once the agent is reliably clearing the level,
so later episodes get harder only after the agent has actually mastered the
current one -- rather than a fixed schedule.

Note: raw score is a bad trigger for this -- even a semi-random agent racks
up a fair score just by wandering into pellets before eventually dying, so
gating on score alone advances the difficulty long before the agent can
really handle it (this was the actual bug: difficulty was hitting max
within a few hundred episodes, faster than epsilon had even decayed, so the
agent spent most of training fighting the hardest ghost roster while still
barely-trained). Gating on win rate ties advancement to genuine competence.
"""
from collections import deque


class DifficultyCurriculum:
    def __init__(self, window=100, win_rate_threshold=0.3, max_level=2, min_episodes_per_level=400):
        self.window = deque(maxlen=window)
        self.win_rate_threshold = win_rate_threshold
        self.max_level = max_level
        self.min_episodes_per_level = min_episodes_per_level
        self._episodes_at_level = 0

    def update(self, env, episode_result):
        self.window.append(1 if episode_result == "win" else 0)
        self._episodes_at_level += 1
        if (len(self.window) == self.window.maxlen
                and self._episodes_at_level >= self.min_episodes_per_level
                and sum(self.window) / len(self.window) >= self.win_rate_threshold
                and env.difficulty < self.max_level):
            env.set_difficulty(env.difficulty + 1)
            self._episodes_at_level = 0
            self.window.clear()
            return True
        return False
