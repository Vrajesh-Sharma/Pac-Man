"""
Compact 13x11 maze definition.

Legend:
  #  wall
  .  normal pellet
  o  power pellet
  P  Pac-Man start position
  (space) empty floor (no pellet)

The maze is symmetric so ghost start corners are equidistant from the
power pellets and from Pac-Man's start in the centre.
"""

DEFAULT_MAZE = [
    "#############",
    "#...#...#...#",
    "#o#.#.#.#.#o#",
    "#...........#",
    "#.###.#.###.#",
    "#.....P.....#",
    "#.###.#.###.#",
    "#...........#",
    "#o#.#.#.#.#o#",
    "#...#...#...#",
    "#############",
]

# Four symmetric corners just inside the outer wall -- used as ghost homes.
GHOST_START_CANDIDATES = [(1, 1), (1, 11), (9, 1), (9, 11)]

# Patrol waypoint loop for the "patrol" ghost personality (clockwise lap).
PATROL_ROUTE = [(1, 1), (1, 11), (9, 11), (9, 1)]

ROWS = len(DEFAULT_MAZE)
COLS = len(DEFAULT_MAZE[0])

# Action encoding shared by env + all agents.
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_DELTA = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}
N_ACTIONS = 4
