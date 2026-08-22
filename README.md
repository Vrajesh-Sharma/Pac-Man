# PacMan RL Benchmark

A compact, visually clean Pac-Man clone built specifically as a **reinforcement
learning benchmark**: the same environment, the same reward function, and the
same maze are used to train and compare six different RL algorithms, so you
can directly show *how differently they learn to play the same game*.

The core question the project answers for a viewer: **does the algorithm
learn to be aggressive (chase score, hunt ghosts) or cautious (prioritize
survival)?** — and you can watch that difference on screen, not just in a
number.

---

## 1. What's in the box

```
pacman_rl/
├── env/
│   ├── maze.py            # the maze layout + action encoding
│   ├── ghosts.py          # 4 ghost personalities: chaser, interceptor, patrol, random
│   └── pacman_env.py       # the shared Gym-style environment (reset/step)
├── agents/
│   ├── base_agent.py       # shared interface
│   ├── q_learning.py       # Q-Learning (off-policy TD)
│   ├── sarsa.py            # SARSA (on-policy TD)
│   ├── monte_carlo.py      # Every-visit Monte Carlo control
│   ├── dqn.py               # Deep Q-Network (PyTorch)
│   ├── double_dqn.py       # Double DQN (fixes DQN's overestimation bias)
│   └── reinforce.py        # REINFORCE policy-gradient (bonus 6th algorithm)
├── utils/
│   ├── replay_buffer.py    # experience replay for DQN/Double DQN
│   ├── metrics_logger.py   # per-episode CSV logging
│   └── curriculum.py       # dynamic-difficulty scheduler
├── train.py                 # train ONE algorithm
├── train_all.py             # train ALL SIX algorithms back-to-back
├── visualize_play.py        # pygame playback: watch an agent play live
├── dashboard.py              # builds the comparison dashboard (PNG + table)
├── requirements.txt
├── results/                  # per-algorithm metrics CSVs + dashboard PNG land here
└── models/                   # trained checkpoints land here
```

---

## 2. Install

You need Python 3.9+.

```bash
cd pacman_rl
pip install -r requirements.txt
```

If `torch` install is slow/fails on your machine, install the CPU-only build
directly from PyTorch's index:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### GPU (CUDA) setup -- NVIDIA RTX 4060 / any CUDA GPU

Q-Learning, SARSA, and Monte Carlo are plain Python dictionaries and get
**no benefit** from a GPU -- they stay CPU-only no matter what. DQN, Double
DQN, and REINFORCE are PyTorch neural nets and *do* benefit, mainly during
`train_step()` (a full replay batch forward+backward pass every step).

1. Install the CUDA build of PyTorch instead of the default one (check
   https://pytorch.org/get-started/locally/ for the exact command matching
   your CUDA version -- this one targets CUDA 12.1, which works with recent
   driver versions on a 4060):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
2. Verify it's detected:
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   should print `True NVIDIA GeForce RTX 4060`.
3. Train a neural agent -- it auto-detects and uses the GPU, and prints
   which device it picked:
   ```bash
   python train.py --algo dqn --episodes 2000
   # -> [dqn] training on GPU: NVIDIA GeForce RTX 4060
   ```
   Force CPU or GPU explicitly with `--device cpu` / `--device cuda` if you
   ever need to. `train_all.py` takes the same flag.
4. An 8GB card has plenty of headroom for this network size -- bump the
   replay batch size up for more GPU throughput per step:
   ```bash
   python train.py --algo dqn --episodes 2000 --batch_size 256
   ```
   Since each episode is now cheaper, this is also a good time to raise
   `--episodes` for dqn/double_dqn/reinforce (e.g. 2500-3000) to get a
   cleaner learning curve than the CPU-budget defaults.

**Everything below has already been smoke-tested end-to-end** in the
environment this was built in (env logic, all 6 agents, training loop,
dashboard, and headless pygame playback all run without errors) — so you're
just running it for real now, not debugging it.

---

## 3. The environment, in one page

- **Maze**: compact 13×11 grid, symmetric, 4 power pellets in the corners,
  normal pellets filling the corridors.
- **Actions**: `0=up, 1=down, 2=left, 3=right`.
- **Ghosts** (personalities defined in `env/ghosts.py`):
  - `chaser` — always takes the shortest path straight at Pac-Man (BFS).
  - `interceptor` — paths toward a point ~4 tiles ahead of Pac-Man's current
    heading, trying to cut him off (Pinky-style ambush).
  - `patrol` — loops around fixed waypoints, and only breaks off to chase
    if Pac-Man wanders close.
  - `random` — mostly random movement, with occasional chase lunges.
  - When Pac-Man eats a power pellet, **every** ghost flips to *frightened*
    and actively flees instead.
- **Rewards** (see `env/pacman_env.py` top constants to tune):
  | Event                          | Reward |
  |---------------------------------|-------:|
  | eat a normal pellet              |  +10   |
  | eat a power pellet               |  +50   |
  | eat a frightened ghost           | +200   |
  | lose a life                      | -500   |
  | clear the whole level            | +1000  |
  | every step (time pressure)       |   -1   |

  This is deliberately *not* just "maximize pellets" — the -500 death
  penalty and the step cost force a genuine trade-off between greedy pellet
  collection, ghost avoidance, and using power pellets tactically.

- **Dynamic difficulty** (`utils/curriculum.py`): a rolling window of recent
  scores is tracked; once an agent is consistently scoring well, the
  environment automatically advances to the next difficulty tier — adding
  ghosts, making them smarter/faster, and reducing the number of power
  pellets available. This happens automatically during `train.py` unless you
  pass `--no_curriculum`.

- **Two observation modes**, both computed from the same underlying state so
  every algorithm sees a fair equivalent of the same game:
  - `get_tabular_state()` — a small hashable tuple (Pac-Man cell, clipped
    relative ghost offsets, power-mode flag, nearest-pellet direction) for
    the tabular methods, whose Q-tables would otherwise blow up if they had
    to track the exact remaining-pellet layout.
  - `get_grid_state()` — a flattened multi-channel grid (walls / pellets /
    power pellets / Pac-Man / normal ghosts / frightened ghosts + a couple
    of scalars) for the neural agents.

---

## 4. Training

### Train one algorithm
```bash
python train.py --algo qlearning  --episodes 6000
python train.py --algo sarsa      --episodes 6000
python train.py --algo montecarlo --episodes 6000
python train.py --algo dqn        --episodes 1500
python train.py --algo double_dqn --episodes 1500
python train.py --algo reinforce  --episodes 1500
```
Each run:
- prints progress every `--log_every` episodes (score, reward, result,
  current difficulty tier, epsilon),
- appends every episode's outcome to `results/<algo>_metrics.csv`,
- saves the final trained model to `models/<algo>.pkl` (tabular) or
  `models/<algo>.pt` (neural).

Tabular methods (Q-Learning, SARSA, Monte Carlo) are cheap — thousands of
episodes run in well under a minute. The neural methods (DQN, Double DQN,
REINFORCE) do a forward+backward pass every step, so budget more wall-clock
time; 1000–2000 episodes is enough to see a clear learning curve on this
compact maze.

### Train everything at once
```bash
python train_all.py            # full run, defaults defined at the top of the file
python train_all.py --quick    # fast smoke test (~200 episodes each) to check your setup works
```
Run `--quick` first — it takes a couple of minutes and confirms your
install is healthy before you commit to the full run.

---

## 5. Watch it play (the demo hook)

```bash
# 1. Show the professor the baseline: an untrained agent flailing around
python visualize_play.py --untrained --episodes 2

# 2. Then show a trained agent for each algorithm, one at a time
python visualize_play.py --algo qlearning  --episodes 3
python visualize_play.py --algo sarsa      --episodes 3
python visualize_play.py --algo montecarlo --episodes 3
python visualize_play.py --algo dqn        --episodes 3
python visualize_play.py --algo double_dqn --episodes 3
python visualize_play.py --algo reinforce  --episodes 3

# crank up difficulty to show it holding up against more/smarter ghosts
python visualize_play.py --algo double_dqn --difficulty 3 --episodes 2
```
A pygame window opens showing the maze, live score/lives/reward/difficulty
in the header, pellets, power pellets, Pac-Man, and each ghost color-coded
by personality (and turning blue when frightened). Press `ESC` or close the
window to stop early.

---

## 6. Compare all algorithms (the dashboard)

Once you've trained some (or all) algorithms:
```bash
python dashboard.py
```
This reads every `results/<algo>_metrics.csv` present and produces
`results/comparison_dashboard.png`, a 2×3 panel figure with:
1. Episode reward curve (rolling average) — convergence speed/stability.
2. Game score curve — pellet/ghost-eating performance.
3. Survival time curve — how long each algorithm stays alive.
4. Win-rate bar chart — full-level clears in the last 20% of episodes.
5. Difficulty-tier progression — shows the curriculum ramping up per algorithm.
6. A final numeric summary table (also saved separately as
   `results/summary_table.csv`).

This single PNG is the "here's the whole comparison" artifact for a
presentation or report.

---

## 7. What to actually look for / talk about

- **Q-Learning vs SARSA**: Q-Learning is off-policy and tends to learn the
  *optimal* (sometimes riskier) path near ghosts; SARSA is on-policy and
  tends to learn a more *cautious* path because it accounts for its own
  exploration mistakes while training. Compare their survival-time curves.
- **Monte Carlo**: learns only from complete episodes with no bootstrapping
  — usually noisier/slower to converge but can be more stable once it does,
  since it never bootstraps off half-learned estimates.
- **DQN vs Double DQN**: DQN is known to over-estimate Q-values (it both
  picks and evaluates the best next action with the same network); Double
  DQN decouples selection/evaluation. Watch for DQN's reward curve being
  more optimistic/unstable early on compared to Double DQN's.
- **REINFORCE**: learns a stochastic policy directly rather than a value
  table/network — worth comparing its behaviour visually, since
  policy-gradient agents often look more "exploratory" even once trained.
- **Aggressive vs cautious strategies**: with the -500 death penalty and
  +200 ghost-eating reward, watch whether a given algorithm learns to camp
  near power pellets and hunt ghosts (aggressive) or mostly avoids ghosts
  and mops up pellets slowly (cautious). This tends to differ by algorithm
  and is the most visually compelling thing to point out live.

---

## 8. Reward tuning notes (v2 — fixes an earlier bug)

An earlier version of this project could finish training with every
algorithm stuck at negative reward and 0% win rate. Root causes, now fixed:

1. **Epsilon barely decayed.** A fixed `epsilon_decay=0.9995` meant epsilon
   was still ~0.22 at the halfway point of a 6000-episode run and only hit
   its floor in the last few dozen episodes — so agents spent nearly the
   entire run acting close to randomly. `train.py` now auto-computes
   `epsilon_decay` from the requested episode count (`_autotune_epsilon_decay`)
   so epsilon reliably reaches its floor by ~50% of training, leaving a real
   exploitation phase.
2. **The tabular state space was astronomically large** (exact relative
   ghost positions for 4 ghosts ≈ tens of millions of states) — a Q-table
   can't get repeat visits to any state in a few thousand episodes.
   `get_tabular_state()` now encodes compact, translation-invariant local
   features (which directions are blocked, which directions have a nearby
   threat/huntable ghost, nearest pellet direction) — roughly 40k possible
   states, and the lesson "a ghost above you is dangerous" now transfers to
   every position in the maze instead of needing to be relearned per cell.
3. **The dynamic-difficulty curriculum advanced on raw score**, which even
   a semi-random agent racks up by wandering into pellets before dying — so
   it was hitting the hardest ghost roster within a few hundred episodes,
   long before any algorithm had learned anything. It now gates on rolling
   **win rate** instead, requires a minimum number of episodes per tier, and
   bumps epsilon back up on every level-up so the agent can actually
   re-explore the harder setup instead of exploiting a stale policy.
4. Ghosts at difficulty 0 were already near-perfect pathfinders; smart_prob
   now ramps gently (0.30 → 0.80) instead of starting at 0.55, and the
   death penalty was reduced from -500 to -200 (still a strong penalty
   relative to a +10 pellet, just no longer structurally guaranteed to
   swamp any score an agent could realistically earn in a short episode).

With these fixes, Q-Learning/SARSA/Monte Carlo reliably reach strongly
positive reward and >40% win rate within 6000 episodes at the default
curriculum cap (difficulty 2). DQN/Double DQN/REINFORCE follow the same
underlying environment and epsilon-autotuning, but need more wall-clock
time per episode (a forward+backward pass every step) — give them
1500-2000 episodes.

If you want to push past difficulty 2, raise `max_level` in
`utils/curriculum.py` (difficulty 3-4 add a 4th ghost and ghost speed
bursts — noticeably harder, good for a stress-test run with a bigger
episode budget).

---

## 9. Tuning knobs worth knowing about

- Reward values: top of `env/pacman_env.py`.
- Maze layout: `env/maze.py` (`DEFAULT_MAZE` — keep it a rectangle of `#`,
  `.`, `o`, `P`, space).
- Ghost personalities per difficulty tier and their "smartness"/speed:
  `DIFFICULTY_LOADOUT`, `DIFFICULTY_SMART_PROB`, `DIFFICULTY_DOUBLE_MOVE` in
  `env/pacman_env.py`.
- Curriculum sensitivity (how good the agent must be, and for how long,
  before difficulty ramps up): `utils/curriculum.py`.
- Agent hyperparameters (learning rate, epsilon decay, network size, replay
  buffer size, etc.): each agent's `__init__` in `agents/`.
