import torch

from .dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """Same architecture and replay setup as DQN, but decouples action
    *selection* from action *evaluation* to fix DQN's overestimation bias:
    the online network picks the best next action, the target network
    evaluates it.  target = r + gamma * Q_target(s', argmax_a Q_online(s',a))
    """

    name = "double_dqn"

    def _target_value(self, s_next_t, done_t):
        with torch.no_grad():
            best_actions = self.online(s_next_t).argmax(dim=1, keepdim=True)
            next_q = self.target(s_next_t).gather(1, best_actions).squeeze(1)
            return next_q * (1 - done_t)
