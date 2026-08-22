import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class ReinforceAgent(BaseAgent):
    """Monte-Carlo policy gradient (REINFORCE) with a moving-average
    baseline and an entropy bonus. Unlike the value-based methods above,
    this learns a *stochastic policy* directly -- interesting to compare
    because it tends to produce visibly different (sometimes more
    exploratory) behaviour than epsilon-greedy value methods.

    Note: REINFORCE has no epsilon -- its exploration comes from sampling
    its own probability distribution over actions, not from an
    epsilon-greedy schedule. `self.epsilon` below is a fixed placeholder
    kept only so train.py's logging code doesn't need a special case; it is
    never used to make decisions and train.py does not decay it.

    Without an entropy bonus, vanilla REINFORCE on a sparse-reward task
    like this one tends to collapse early onto a near-deterministic policy
    (probability mass dumped onto whatever action looked best in the first
    few noisy episodes) and then never explores its way out of a bad local
    optimum -- which is exactly the "always loses" behaviour. The entropy
    term below directly rewards keeping the action distribution spread out,
    counteracting that collapse."""

    name = "reinforce"

    def __init__(self, input_dim, n_actions=4, gamma=0.95, lr=1e-3,
                 entropy_coef=0.02, device=None, verbose_device=True):
        self.n_actions = n_actions
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose_device:
            if self.device.type == "cuda":
                print(f"[{self.name}] training on GPU: {torch.cuda.get_device_name(self.device)}")
            else:
                print(f"[{self.name}] training on CPU (no CUDA GPU detected)")
        self.policy = PolicyNetwork(input_dim, n_actions).to(self.device)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self._log_probs = []
        self._entropies = []
        self._rewards = []
        self._baseline = 0.0
        self.last_mean_entropy = None  # for logging, in place of "epsilon"
        # kept for interface parity with other agents (see docstring above)
        self.epsilon = 0.0

    def select_action(self, state, training=True):
        t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.policy(t)
        dist = torch.distributions.Categorical(probs)
        if training:
            action = dist.sample()
            self._log_probs.append(dist.log_prob(action))
            self._entropies.append(dist.entropy())
            return int(action.item())
        return int(torch.argmax(probs, dim=1).item())

    def q_values(self, state):
        """Return action *probabilities* (not true Q-values, but usable the
        same way by visualize_play.py's wall-avoidance masking)."""
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.policy(t).squeeze(0).cpu().numpy()

    def record_reward(self, r):
        self._rewards.append(r)

    def on_episode_end(self):
        if not self._rewards:
            return
        G, returns = 0.0, []
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        # normalize / baseline-subtract for variance reduction
        self._baseline = 0.95 * self._baseline + 0.05 * returns.mean().item()
        advantages = returns - self._baseline
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        entropies = torch.stack(self._entropies)
        self.last_mean_entropy = entropies.mean().item()

        policy_loss = -(torch.stack(self._log_probs) * advantages).sum()
        entropy_bonus = entropies.sum()
        loss = policy_loss - self.entropy_coef * entropy_bonus

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=5.0)
        self.opt.step()

        self._log_probs = []
        self._entropies = []
        self._rewards = []

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state)
