import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent
from utils.replay_buffer import ReplayBuffer

# Speeds up convolution/linear algorithm selection on a fixed input shape --
# harmless no-op on CPU, free win on CUDA.
torch.backends.cudnn.benchmark = True


class QNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent(BaseAgent):
    """Vanilla Deep Q-Network: MLP over the full grid observation, experience
    replay, and a periodically-synced target network to stabilize the TD
    target. Uses max_a' Q_target(s', a') -- known to over-estimate Q-values,
    which is exactly the failure mode Double DQN fixes.

    GPU notes: pass device="cuda" (or leave device=None to auto-detect) to
    run on an NVIDIA GPU. The network here is small, so per-step action
    selection (batch size 1) sees only a modest speedup -- the real win is
    train_step(), which processes a full replay batch through the network
    and backprop every call. Bumping --batch_size up (e.g. 128-256) on a
    GPU is essentially free and gives more stable gradient estimates too."""

    name = "dqn"

    def __init__(self, input_dim, n_actions=4, gamma=0.95, lr=1e-3,
                 epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.05,
                 buffer_size=50000, batch_size=128, target_sync=500,
                 device=None, verbose_device=True):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose_device:
            if self.device.type == "cuda":
                print(f"[{self.name}] training on GPU: {torch.cuda.get_device_name(self.device)}")
            else:
                print(f"[{self.name}] training on CPU (no CUDA GPU detected)")

        self.online = QNetwork(input_dim, n_actions).to(self.device)
        self.target = QNetwork(input_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self._train_steps = 0

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(t)
            return int(torch.argmax(q, dim=1).item())

    def q_values(self, state):
        """Return this state's action-value estimates as a plain numpy
        array. Used by visualize_play.py to mask out wall-blocked actions
        during greedy playback without touching training behaviour."""
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.online(t).squeeze(0).cpu().numpy()

    def remember(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)

    def _target_value(self, s_next_t, done_t):
        with torch.no_grad():
            next_q = self.target(s_next_t).max(dim=1)[0]
            return next_q * (1 - done_t)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s_next, done = self.buffer.sample(self.batch_size)
        # non_blocking has an effect when the source tensor is pinned host
        # memory; harmless (falls back to a normal copy) otherwise.
        s_t = torch.as_tensor(s).to(self.device, non_blocking=True)
        a_t = torch.as_tensor(a).to(self.device, non_blocking=True)
        r_t = torch.as_tensor(r).to(self.device, non_blocking=True)
        s_next_t = torch.as_tensor(s_next).to(self.device, non_blocking=True)
        done_t = torch.as_tensor(done).to(self.device, non_blocking=True)

        q_vals = self.online(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        target = r_t + self.gamma * self._target_value(s_next_t, done_t)
        # Huber loss (smooth L1) instead of plain MSE: this environment's
        # rewards span a wide range (-200 death penalty to +1000 level
        # clear), and MSE's squared-error gradient on the rare huge-reward
        # transitions can dominate/destabilize training. Huber behaves like
        # MSE for small errors but like a bounded linear penalty for large
        # ones, which is the standard fix for this in DQN.
        loss = nn.functional.smooth_l1_loss(q_vals, target)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.opt.step()

        self._train_steps += 1
        if self._train_steps % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss.item()

    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.online.load_state_dict(state)
        self.target.load_state_dict(state)
