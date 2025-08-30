
from dataclasses import dataclass
from typing import Tuple, List, Dict

@dataclass(frozen=True)
class State:
    r: int
    c: int

class GridMDP:
    """
    Deterministic Grid World MDP with walls and a single goal.
    Coordinates are zero-indexed with (0,0) at the TOP-LEFT of the grid.
    """
    def __init__(self, N: int, walls: set, start: Tuple[int,int], goal: Tuple[int,int],
                 step_reward: float = -1.0, wall_penalty: float = -2.0, goal_reward: float = 10.0,
                 gamma: float = 0.9):
        self.N = N
        self.walls = set((int(r), int(c)) for r,c in walls)
        self.start = State(int(start[0]), int(start[1]))
        self.goal = State(int(goal[0]), int(goal[1]))
        self.step_reward = float(step_reward)
        self.wall_penalty = float(wall_penalty)
        self.goal_reward = float(goal_reward)
        self.gamma = float(gamma)

        self.action_map = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1),
        }

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.N and 0 <= c < self.N

    def is_wall(self, r: int, c: int) -> bool:
        return (r, c) in self.walls

    def is_terminal(self, s: State) -> bool:
        return (s.r, s.c) == (self.goal.r, self.goal.c)

    def available_actions(self, s: State) -> List[str]:
        # Even in terminal, we can return actions, but the simulator will end the episode when terminal is reached.
        return list(self.action_map.keys())

    def transition(self, s: State, a: str) -> Tuple[State, float, bool]:
        """Deterministic transition. Invalid moves (off-grid/wall) keep the agent in place, adding wall_penalty."""
        if a not in self.action_map:
            raise ValueError(f"Unknown action: {a}")
        if self.is_terminal(s):
            # Episode should be ended by the simulator; if called, remain in goal with zero reward
            return s, 0.0, True

        dr, dc = self.action_map[a]
        nr, nc = s.r + dr, s.c + dc

        if not self.in_bounds(nr, nc) or self.is_wall(nr, nc):
            # bump into wall/edge: stay put + wall penalty
            ns = s
            reward = self.wall_penalty
            done = False
        else:
            ns = State(nr, nc)
            if self.is_terminal(ns):
                reward = self.goal_reward
                done = True
            else:
                reward = self.step_reward
                done = False
        return ns, reward, done

    def manhattan(self, s: State) -> int:
        return abs(s.r - self.goal.r) + abs(s.c - self.goal.c)
