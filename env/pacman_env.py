"""
PacManEnv: a compact, gym-style (reset/step) environment used as a shared
benchmark for all RL algorithms in this project.

reset() -> (tabular_state, info)
step(action) -> (tabular_state, reward, terminated, truncated, info)

Two observation views are exposed so both tabular and deep-RL agents can
share the exact same environment:
  - get_tabular_state(): small hashable tuple (for Q-learning/SARSA/MC)
  - get_grid_state():    flat float32 vector (for DQN/DoubleDQN/REINFORCE)

Reward shaping (as specified, tuned so a decent policy nets a clearly
positive total -- see README "Reward tuning notes" for the reasoning):
  pellet                : +10
  power pellet          : +50
  eat frightened ghost  : +200
  lose a life           : -200
  clear the level       : +1000
  per-step time cost    : -0.5  (keeps agents from stalling forever)
"""
import copy
import random

import numpy as np

from .maze import DEFAULT_MAZE, GHOST_START_CANDIDATES, PATROL_ROUTE, ACTION_DELTA, N_ACTIONS
from .ghosts import Ghost, neighbors

REWARD_PELLET = 10
REWARD_POWER_PELLET = 50
REWARD_EAT_GHOST = 200
REWARD_DEATH = -200
REWARD_WIN = 1000
REWARD_STEP = -0.5
WALL_BUMP_PENALTY = -3     # extra penalty for walking into a wall (wastes a turn)
OSCILLATION_PENALTY = -3   # extra penalty for immediately reversing back to the cell
                           # you were in two steps ago (breaks back-and-forth loops)
POWER_DURATION = 30  # steps

# Ghost personality loadout per difficulty tier. Difficulty 0 is now a
# single, none-too-perfect chaser so a brand new agent can actually survive
# long enough to receive learning signal; the curriculum adds ghosts and
# sharpens their play as the agent proves it can handle more.
DIFFICULTY_LOADOUT = {
    0: ["chaser"],
    1: ["chaser", "random"],
    2: ["chaser", "interceptor", "random"],
    3: ["chaser", "interceptor", "patrol", "random"],
    4: ["chaser", "interceptor", "patrol", "random"],
}
# smart_prob = how often ghosts take the optimal move instead of a random one.
DIFFICULTY_SMART_PROB = {0: 0.30, 1: 0.40, 2: 0.50, 3: 0.65, 4: 0.80}
# double_move_prob = chance a ghost takes an extra step this turn (speed boost).
DIFFICULTY_DOUBLE_MOVE = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.15, 4: 0.3}
# how many power pellets remain active at higher difficulty (resource cut).
DIFFICULTY_MAX_POWER_PELLETS = {0: 4, 1: 4, 2: 4, 3: 3, 4: 2}


class PacManEnv:
    def __init__(self, maze=None, max_steps=400, difficulty=0, lives=4, seed=None):
        self.maze_lines = maze or DEFAULT_MAZE
        self.rows = len(self.maze_lines)
        self.cols = len(self.maze_lines[0])
        self.max_steps = max_steps
        self.start_lives = lives
        self.n_actions = N_ACTIONS
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.walls = [[c == "#" for c in row] for row in self.maze_lines]
        self.pacman_start = None
        base_pellets, base_power = set(), set()
        for r, row in enumerate(self.maze_lines):
            for c, ch in enumerate(row):
                if ch == "P":
                    self.pacman_start = (r, c)
                elif ch == "o":
                    base_power.add((r, c))
                elif ch == ".":
                    base_pellets.add((r, c))
        # ghost homes never hold pellets
        for gs in GHOST_START_CANDIDATES:
            base_pellets.discard(gs)
        self.base_pellets = base_pellets
        self.base_power = base_power
        self.total_pellets_full = len(base_pellets) + len(base_power)

        self.difficulty = difficulty
        self.ghosts = []
        self._build_ghosts()

        # runtime state (populated in reset)
        self.pacman_pos = self.pacman_start
        self.pacman_dir = None
        self.pellets = set()
        self.power_pellets = set()
        self.lives = lives
        self.score = 0
        self.power_timer = 0
        self.steps = 0
        self.last_action = None       # feeds the tabular state ("what did I just do")
        self._recent_positions = []   # last 2 cells, used to detect immediate reversals

    # ------------------------------------------------------------------ #
    # setup helpers
    # ------------------------------------------------------------------ #
    def _build_ghosts(self):
        behaviors = DIFFICULTY_LOADOUT[min(self.difficulty, max(DIFFICULTY_LOADOUT))]
        self.ghosts = []
        for i, behavior in enumerate(behaviors):
            start = GHOST_START_CANDIDATES[i % len(GHOST_START_CANDIDATES)]
            self.ghosts.append(Ghost(start, behavior, PATROL_ROUTE, name=f"{behavior}_{i}"))

    def set_difficulty(self, level):
        """Reconfigure ghost roster / speed / resources for curriculum training."""
        level = max(0, min(level, max(DIFFICULTY_LOADOUT)))
        self.difficulty = level
        self._build_ghosts()

    # ------------------------------------------------------------------ #
    # gym-style API
    # ------------------------------------------------------------------ #
    def reset(self):
        self.pacman_pos = self.pacman_start
        self.pacman_dir = None
        self.pellets = set(self.base_pellets)
        max_power = DIFFICULTY_MAX_POWER_PELLETS[self.difficulty]
        self.power_pellets = set(list(self.base_power)[:max_power])
        self.lives = self.start_lives
        self.score = 0
        self.power_timer = 0
        self.steps = 0
        self.last_action = None
        self._recent_positions = [self.pacman_start]
        for g in self.ghosts:
            g.reset()
        return self.get_tabular_state(), self._info()

    def step(self, action):
        assert 0 <= action < N_ACTIONS
        reward = REWARD_STEP
        terminated = False
        truncated = False

        # --- move pac-man ---
        old_pos = self.pacman_pos
        dr, dc = ACTION_DELTA[action]
        nr, nc = self.pacman_pos[0] + dr, self.pacman_pos[1] + dc
        moved = 0 <= nr < self.rows and 0 <= nc < self.cols and not self.walls[nr][nc]
        if moved:
            self.pacman_pos = (nr, nc)
            self.pacman_dir = action
        else:
            # walked straight into a wall -- wasted a turn, extra penalty so
            # the agent actually learns to avoid this instead of freezing
            # against a wall it hasn't explored its way around yet.
            reward += WALL_BUMP_PENALTY

        # immediate back-and-forth detection: if this move landed on the
        # cell we were in two steps ago, nudge against the loop. A single
        # legitimate backtrack (e.g. a dead end) isn't penalized twice in a
        # row unless the agent keeps oscillating.
        if moved and len(self._recent_positions) >= 2 and self.pacman_pos == self._recent_positions[-2]:
            reward += OSCILLATION_PENALTY
        self._recent_positions.append(self.pacman_pos)
        if len(self._recent_positions) > 3:
            self._recent_positions.pop(0)
        self.last_action = action

        # --- pellets ---
        if self.pacman_pos in self.pellets:
            self.pellets.remove(self.pacman_pos)
            reward += REWARD_PELLET
            self.score += REWARD_PELLET
        if self.pacman_pos in self.power_pellets:
            self.power_pellets.remove(self.pacman_pos)
            reward += REWARD_POWER_PELLET
            self.score += REWARD_POWER_PELLET
            self.power_timer = POWER_DURATION
            for g in self.ghosts:
                g.frightened = True

        # --- ghosts move ---
        smart_prob = DIFFICULTY_SMART_PROB[self.difficulty]
        double_move_p = DIFFICULTY_DOUBLE_MOVE[self.difficulty]
        for g in self.ghosts:
            moves = 2 if random.random() < double_move_p else 1
            for _ in range(moves):
                g.pos = g.choose_next_pos(self.walls, self.pacman_pos, self.pacman_dir, smart_prob)

        # --- power timer countdown ---
        if self.power_timer > 0:
            self.power_timer -= 1
            if self.power_timer == 0:
                for g in self.ghosts:
                    g.frightened = False

        # --- collisions ---
        for g in self.ghosts:
            if g.pos == self.pacman_pos:
                if g.frightened:
                    reward += REWARD_EAT_GHOST
                    self.score += REWARD_EAT_GHOST
                    g.reset()  # sent home, no longer frightened
                else:
                    self.lives -= 1
                    reward += REWARD_DEATH
                    if self.lives <= 0:
                        terminated = True
                    else:
                        self.pacman_pos = self.pacman_start
                        self.pacman_dir = None
                        self.power_timer = 0
                        self.last_action = None
                        self._recent_positions = [self.pacman_start]
                        for gg in self.ghosts:
                            gg.reset()
                    break  # one collision resolved per step

        # --- win condition ---
        if not self.pellets and not self.power_pellets and not terminated:
            reward += REWARD_WIN
            self.score += REWARD_WIN
            terminated = True

        self.steps += 1
        if self.steps >= self.max_steps and not terminated:
            truncated = True

        info = self._info()
        info["result"] = "win" if (terminated and not self.pellets and not self.power_pellets and self.lives > 0) \
            else ("lose" if terminated else ("timeout" if truncated else "ongoing"))
        return self.get_tabular_state(), reward, terminated, truncated, info

    def _info(self):
        return {
            "score": self.score,
            "lives": self.lives,
            "pellets_left": len(self.pellets) + len(self.power_pellets),
            "difficulty": self.difficulty,
            "steps": self.steps,
        }

    # ------------------------------------------------------------------ #
    # observations
    # ------------------------------------------------------------------ #
    def _nearest_signed_dir(self, pr, pc, targets):
        """Return (dr_sign, dc_sign) pointing from (pr,pc) toward the nearest
        of `targets`, or (0, 0) if targets is empty. Signed (-1/0/1) per axis
        instead of a single up/down/left/right label -- lets the agent see
        diagonal relationships instead of only the dominant axis."""
        if not targets:
            return 0, 0
        best = min(targets, key=lambda t: abs(t[0] - pr) + abs(t[1] - pc))
        dr, dc = best[0] - pr, best[1] - pc
        return (0 if dr == 0 else (1 if dr > 0 else -1),
                0 if dc == 0 else (1 if dc > 0 else -1))

    def get_tabular_state(self):
        """Compact but disambiguated feature state for tabular agents --
        the agent's "eyes": which directions are wall-blocked, where the
        nearest dangerous ghost is (direction + near/far), where the
        nearest huntable frightened ghost is (while powered), where the
        nearest pellet is, and what it just did last step.

        v1 used only coarse boolean "is there a threat somewhere within 3
        tiles in direction X" flags aggregated across every ghost, with no
        memory of the previous action. Two genuinely different situations
        (e.g. "ghost 1 tile north" vs "ghost 3 tiles north") collapsed to
        the same state, and with no notion of "what did I just do", the
        learned policy could get stuck alternating between two actions that
        both looked equally good from an identical-looking state -- that's
        the freezing/back-and-forth behaviour. This version disambiguates
        distance (near vs far) and adds last_action so the Q-table can
        learn "don't immediately reverse" as part of the state itself, on
        top of the explicit oscillation penalty in step().
        """
        pr, pc = self.pacman_pos
        walls_dir = []
        for a, (dr, dc) in ACTION_DELTA.items():
            nr, nc = pr + dr, pc + dc
            blocked = not (0 <= nr < self.rows and 0 <= nc < self.cols) or self.walls[nr][nc]
            walls_dir.append(int(blocked))

        # nearest dangerous (non-frightened) ghost
        threats = [g.pos for g in self.ghosts if not g.frightened]
        t_dr, t_dc = self._nearest_signed_dir(pr, pc, threats)
        if threats:
            t_dist = min(abs(g[0] - pr) + abs(g[1] - pc) for g in threats)
            t_near = int(t_dist <= 2)
            t_present = 1
        else:
            t_near, t_present = 0, 0

        # nearest huntable (frightened) ghost -- only meaningful while powered
        hunters = [g.pos for g in self.ghosts if g.frightened] if self.power_timer > 0 else []
        h_dr, h_dc = self._nearest_signed_dir(pr, pc, hunters)
        if hunters:
            h_dist = min(abs(g[0] - pr) + abs(g[1] - pc) for g in hunters)
            h_near = int(h_dist <= 3)
            h_present = 1
        else:
            h_near, h_present = 0, 0

        pellet_targets = list(self.pellets) + list(self.power_pellets)
        p_dr, p_dc = self._nearest_signed_dir(pr, pc, pellet_targets)

        return (
            tuple(walls_dir),                              # 16 combos
            (t_present, t_dr, t_dc, t_near),                # threat: presence, direction, near/far
            (h_present, h_dr, h_dc, h_near),                # hunt target: same shape
            int(self.power_timer > 0),
            (p_dr, p_dc),                                    # signed pellet direction (both axes)
            self.last_action if self.last_action is not None else 4,  # what we just did
        )

    def get_grid_state(self):
        """Flat float32 vector for neural-net agents: one channel per
        semantic layer of the maze, flattened."""
        walls = np.array(self.walls, dtype=np.float32)
        pellets = np.zeros((self.rows, self.cols), dtype=np.float32)
        for r, c in self.pellets:
            pellets[r, c] = 1.0
        power = np.zeros((self.rows, self.cols), dtype=np.float32)
        for r, c in self.power_pellets:
            power[r, c] = 1.0
        pac = np.zeros((self.rows, self.cols), dtype=np.float32)
        pac[self.pacman_pos] = 1.0
        ghosts_normal = np.zeros((self.rows, self.cols), dtype=np.float32)
        ghosts_frightened = np.zeros((self.rows, self.cols), dtype=np.float32)
        for g in self.ghosts:
            if g.frightened:
                ghosts_frightened[g.pos] = 1.0
            else:
                ghosts_normal[g.pos] = 1.0
        scalar = np.array([self.power_timer / POWER_DURATION, self.lives / self.start_lives], dtype=np.float32)
        stacked = np.stack([walls, pellets, power, pac, ghosts_normal, ghosts_frightened])
        return np.concatenate([stacked.flatten(), scalar]).astype(np.float32)

    @property
    def grid_state_dim(self):
        return self.rows * self.cols * 6 + 2

    def clone(self):
        return copy.deepcopy(self)
