"""
Watch an agent play the maze in a pygame window. This is the "wow" demo:
run with --untrained to show random flailing, then run with --algo qlearning
(after training) to show the learned behaviour, side by side conceptually.

Usage:
    python visualize_play.py --untrained                       # random baseline
    python visualize_play.py --algo qlearning                  # loads models/qlearning.pkl
    python visualize_play.py --algo dqn --difficulty 2 --fps 8
    python visualize_play.py --algo double_dqn --episodes 3

By default the difficulty shown matches whatever difficulty that model's
training actually ended on (read from models/<algo>_meta.json, written by
train.py) -- pass --difficulty explicitly to override. This avoids the
confusing situation where a model trained up to difficulty 2 (3 ghosts) gets
played back at the default difficulty 0 (1 ghost) and looks like ghosts are
randomly appearing/disappearing between runs.

Controls: close the window or press ESC to quit early.
"""


import argparse
import os
import sys

import pygame

from env.pacman_env import PacManEnv
from env.maze import ACTION_DELTA

CELL = 42
MARGIN_TOP = 60
COLORS = {
    "bg": (10, 10, 30),
    "wall": (33, 33, 222),
    "pellet": (255, 222, 130),
    "power": (255, 120, 120),
    "pacman": (255, 222, 0),      # Pac-Man is always solid yellow
    "text": (255, 255, 255),
    "frightened": (70, 70, 255),   # all ghosts turn this blue when powered
}
# Fixed per-slot ghost colors (classic Blinky/Pinky/Inky/Clyde palette).
# Assigned by *slot index*, not by behaviour, so a ghost's color never
# changes as it moves -- only frightened state changes its color.
GHOST_SLOT_COLORS = [
    (220, 30, 30),    # red
    (255, 130, 190),  # pink
    (0, 210, 210),    # cyan
    (255, 165, 40),   # orange
]
# Small fixed per-slot pixel offset so ghosts sharing a cell (which happens
# often when they converge on Pac-Man) render as distinguishable circles
# instead of perfectly overlapping into what looks like one flickering blob.
GHOST_SLOT_JITTER = [(-8, -8), (8, -8), (-8, 8), (8, 8)]


def load_agent(algo, input_dim):
    if algo is None:
        return None, None
    from train import build_agent, model_path, load_meta
    agent = build_agent(algo, input_dim=input_dim)
    path = model_path(algo)
    meta = load_meta(algo)
    if not os.path.exists(path):
        print(f"WARNING: no saved model at {path} -- showing an UNTRAINED '{algo}' agent instead.")
        return agent, meta
    agent.load(path)
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0  # greedy playback
    return agent, meta


def valid_actions(env):
    """Actions that don't walk straight into a wall from Pac-Man's current
    cell. Always returns at least one action (falls back to all four in the
    pathological case every neighbor is somehow blocked)."""
    pr, pc = env.pacman_pos
    valid = []
    for a, (dr, dc) in ACTION_DELTA.items():
        nr, nc = pr + dr, pc + dc
        if 0 <= nr < env.rows and 0 <= nc < env.cols and not env.walls[nr][nc]:
            valid.append(a)
    return valid or [0, 1, 2, 3]


def pick_action(agent, state, env, deterministic):
    """Greedy playback with wall-avoidance masking: if the agent has a
    q_values()-style method, pick the best-scoring action among only the
    ones that don't immediately hit a wall, instead of trusting the raw
    argmax. This matters because a state the agent barely visited during
    training can have near-arbitrary Q-values, and without masking the
    agent can freeze bumping into the same wall repeatedly during a demo
    even though a clearly better move is available right next to it. This
    only affects playback -- training still uses each agent's own
    select_action() unmasked, since exploring "bad" moves is how it learns
    to avoid them in the first place."""
    if agent is None:
        import random
        return random.choice(valid_actions(env))
    if not deterministic or not hasattr(agent, "q_values"):
        return agent.select_action(state, training=False)
    scores = agent.q_values(state)
    valid = valid_actions(env)
    return max(valid, key=lambda a: scores[a])


def draw(screen, env, font, algo_label, episode, total_reward):
    screen.fill(COLORS["bg"])
    for r in range(env.rows):
        for c in range(env.cols):
            x, y = c * CELL, MARGIN_TOP + r * CELL
            if env.walls[r][c]:
                pygame.draw.rect(screen, COLORS["wall"], (x, y, CELL, CELL), border_radius=6)
            else:
                if (r, c) in env.pellets:
                    pygame.draw.circle(screen, COLORS["pellet"], (x + CELL // 2, y + CELL // 2), 4)
                if (r, c) in env.power_pellets:
                    pygame.draw.circle(screen, COLORS["power"], (x + CELL // 2, y + CELL // 2), 9)

    for i, g in enumerate(env.ghosts):
        jx, jy = GHOST_SLOT_JITTER[i % len(GHOST_SLOT_JITTER)]
        gx = g.pos[1] * CELL + CELL // 2 + jx
        gy = MARGIN_TOP + g.pos[0] * CELL + CELL // 2 + jy
        color = COLORS["frightened"] if g.frightened else GHOST_SLOT_COLORS[i % len(GHOST_SLOT_COLORS)]
        radius = CELL // 2 - 10  # smaller so jittered ghosts don't overlap each other
        pygame.draw.circle(screen, color, (gx, gy), radius)
        pygame.draw.circle(screen, (255, 255, 255), (gx, gy), radius, width=2)  # outline for separation

    px, py = env.pacman_pos[1] * CELL + CELL // 2, MARGIN_TOP + env.pacman_pos[0] * CELL + CELL // 2
    pygame.draw.circle(screen, COLORS["pacman"], (px, py), CELL // 2 - 3)
    pygame.draw.circle(screen, (0, 0, 0), (px, py), CELL // 2 - 3, width=2)

    hud = (f"{algo_label} | Episode {episode} | Score {env.score} | Lives {env.lives} "
           f"| Reward {total_reward:.0f} | Difficulty {env.difficulty}")
    screen.blit(font.render(hud, True, COLORS["text"]), (10, 10))

    legend_parts = [f"{g.behavior}" for g in env.ghosts]
    x = 10
    y = 32
    small_font = pygame.font.SysFont("consolas", 13)
    for i, label in enumerate(legend_parts):
        color = GHOST_SLOT_COLORS[i % len(GHOST_SLOT_COLORS)]
        pygame.draw.circle(screen, color, (x + 6, y + 7), 6)
        pygame.draw.circle(screen, (255, 255, 255), (x + 6, y + 7), 6, width=1)
        txt = small_font.render(label, True, COLORS["text"])
        screen.blit(txt, (x + 16, y))
        x += 16 + txt.get_width() + 14
    if not legend_parts:
        screen.blit(small_font.render("(no ghosts at this difficulty)", True, COLORS["text"]), (x, y))

    pygame.display.flip()


def run(algo, untrained, episodes, difficulty, fps, max_steps, deterministic=True):
    input_dim_probe = PacManEnv(difficulty=0, max_steps=max_steps).grid_state_dim
    agent, meta = (None, None) if untrained else load_agent(algo, input_dim_probe)

    if difficulty is None:
        # no explicit --difficulty given -- default to whatever this model
        # actually finished training at, so playback matches what it learned.
        difficulty = meta["final_difficulty"] if meta else 0

    env = PacManEnv(difficulty=difficulty, max_steps=max_steps)
    use_grid = algo in ("dqn", "double_dqn", "reinforce") and agent is not None

    pygame.init()
    width = env.cols * CELL
    height = env.rows * CELL + MARGIN_TOP
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("PacMan RL Benchmark - Playback")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    label = "RANDOM (untrained baseline)" if untrained else f"{algo} (trained)" if agent else f"{algo} (untrained - no checkpoint found)"
    matched = meta is not None and difficulty == meta.get("final_difficulty")
    print(f"Playing at difficulty {difficulty} "
          f"({'auto-matched to training' if matched else 'explicit/default'}) "
          f"-- ghost roster: {[g.behavior for g in env.ghosts]}")

    for ep in range(1, episodes + 1):
        env.reset()
        s = env.get_grid_state() if use_grid else env.get_tabular_state()
        done = False
        total_reward = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit(0)

            action = pick_action(agent, s, env, deterministic)

            _, r, term, trunc, info = env.step(action)
            s = env.get_grid_state() if use_grid else env.get_tabular_state()
            total_reward += r
            done = term or trunc

            draw(screen, env, font, label, ep, total_reward)
            clock.tick(fps)

        print(f"Episode {ep}: result={info['result']} score={info['score']} reward={total_reward:.0f}")

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["qlearning", "sarsa", "montecarlo", "dqn", "double_dqn", "reinforce"])
    parser.add_argument("--untrained", action="store_true", help="ignore --algo, play fully random baseline")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--difficulty", type=int, default=None,
                         help="Force a difficulty tier. Default: auto-match whatever "
                              "difficulty this model's training actually reached.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--no_masking", action="store_true",
                         help="Disable wall-avoidance action masking and show the "
                              "agent's raw greedy policy, warts and all.")
    args = parser.parse_args()

    if not args.untrained and not args.algo:
        parser.error("pass --algo <name> or --untrained")

    run(args.algo, args.untrained, args.episodes, args.difficulty, args.fps,
        args.max_steps, deterministic=not args.no_masking)
