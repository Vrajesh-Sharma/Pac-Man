"""
Ghost personalities. Each ghost picks its next grid cell every step using
a behaviour-specific target policy, then a shared BFS pathfinder converts
"target cell" into "next single step" on the maze graph.

Personalities:
  chaser       - always paths straight at Pac-Man's current cell (aggressive).
  interceptor  - paths toward a predicted cell a few tiles ahead of Pac-Man's
                 current heading (tries to cut him off, Pinky-style).
  patrol       - loops around a fixed waypoint route, ignoring Pac-Man
                 until he gets close, then behaves like a chaser.
  random       - mostly moves randomly; occasionally lunges at Pac-Man.

When Pac-Man is powered up (frightened mode) every ghost overrides its
normal target and instead tries to maximize distance from Pac-Man.
"""
import random
from collections import deque

from .maze import ACTION_DELTA, PATROL_ROUTE


def neighbors(pos, walls):
    r, c = pos
    rows, cols = len(walls), len(walls[0])
    out = []
    for a, (dr, dc) in ACTION_DELTA.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and not walls[nr][nc]:
            out.append((a, (nr, nc)))
    return out


def bfs_next_step(walls, start, target):
    """Return the first step (action, cell) of the shortest path start->target.
    If start == target or unreachable, returns None."""
    if start == target:
        return None
    visited = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == target:
            break
        for a, nxt in neighbors(cur, walls):
            if nxt not in visited:
                visited[nxt] = (cur, a)
                q.append(nxt)
    if target not in visited:
        return None
    # walk back from target to the cell adjacent to start
    step = target
    while visited[step] is not None and visited[step][0] != start:
        step = visited[step][0]
    if visited[step] is None:
        return None
    return visited[step][1], step  # action, resulting cell


class Ghost:
    def __init__(self, start, behavior, patrol_route=None, name=""):
        self.start = start
        self.pos = start
        self.behavior = behavior  # 'chaser' | 'interceptor' | 'patrol' | 'random'
        self.name = name or behavior
        self.patrol_route = patrol_route or PATROL_ROUTE
        self.patrol_idx = 0
        self.frightened = False
        self.eaten = False  # briefly true right after being eaten, then respawns

    def reset(self):
        self.pos = self.start
        self.patrol_idx = 0
        self.frightened = False
        self.eaten = False

    def _random_move(self, walls):
        opts = neighbors(self.pos, walls)
        return random.choice(opts)[1] if opts else self.pos

    def _target_move(self, walls, target):
        result = bfs_next_step(walls, self.pos, target)
        if result is None:
            return self._random_move(walls)
        return result[1]

    def _flee_move(self, walls, pacman_pos):
        opts = neighbors(self.pos, walls)
        if not opts:
            return self.pos
        # pick the neighbor that maximizes Manhattan distance from Pac-Man,
        # with a little randomness so fleeing isn't perfectly predictable.
        best = max(
            opts,
            key=lambda oc: abs(oc[1][0] - pacman_pos[0]) + abs(oc[1][1] - pacman_pos[1])
            + random.random() * 0.5,
        )
        return best[1]

    def choose_next_pos(self, walls, pacman_pos, pacman_dir, smart_prob=0.85):
        """Decide the next grid cell for this ghost."""
        if self.frightened:
            if random.random() < 0.7:
                return self._flee_move(walls, pacman_pos)
            return self._random_move(walls)

        if random.random() > smart_prob:
            # occasional mistake, scaled by difficulty via smart_prob
            return self._random_move(walls)

        if self.behavior == "chaser":
            return self._target_move(walls, pacman_pos)

        if self.behavior == "interceptor":
            dr, dc = ACTION_DELTA.get(pacman_dir, (0, 0))
            rows, cols = len(walls), len(walls[0])
            tr = min(max(pacman_pos[0] + dr * 4, 1), rows - 2)
            tc = min(max(pacman_pos[1] + dc * 4, 1), cols - 2)
            target = (tr, tc) if not walls[tr][tc] else pacman_pos
            return self._target_move(walls, target)

        if self.behavior == "patrol":
            dist_to_pacman = abs(self.pos[0] - pacman_pos[0]) + abs(self.pos[1] - pacman_pos[1])
            if dist_to_pacman <= 3:
                return self._target_move(walls, pacman_pos)
            waypoint = self.patrol_route[self.patrol_idx % len(self.patrol_route)]
            if self.pos == waypoint:
                self.patrol_idx += 1
                waypoint = self.patrol_route[self.patrol_idx % len(self.patrol_route)]
            return self._target_move(walls, waypoint)

        if self.behavior == "random":
            if random.random() < 0.25:
                return self._target_move(walls, pacman_pos)
            return self._random_move(walls)

        return self._random_move(walls)
