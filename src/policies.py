
import random
from typing import List
from maze_env import GridMDP, State

class RandomPolicy:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def action(self, env: GridMDP, s: State) -> str:
        acts = env.available_actions(s)
        return self.rng.choice(acts)

class GreedyManhattanPolicy:
    """
    Greedy towards the goal by minimizing Manhattan distance.
    Tie-break preference order: U, R, D, L.
    If no move reduces distance (local obstacle), pick any valid move that does not increase distance by a lot;
    fallback to a random action among available.
    """
    def __init__(self, seed: int = 123):
        self.rng = random.Random(seed)
        self.pref_order = ['U','R','D','L']

    def action(self, env: GridMDP, s: State) -> str:
        if env.is_terminal(s):
            return 'U'

        best_acts = []
        best_dist = env.manhattan(s)
        # try moves that reduce distance first
        for a in self.pref_order:
            ns, _, _ = self._peek(env, s, a)
            if ns is None:
                continue
            d = env.manhattan(ns)
            if d < best_dist:
                best_acts.append(a)

        if best_acts:
            # follow preference order among reducing moves
            for a in self.pref_order:
                if a in best_acts: 
                    return a

        # Otherwise, pick any action that doesn't collide (peek returns None when collision)
        non_colliding = [a for a in self.pref_order if self._peek(env, s, a)[0] is not None]
        if non_colliding:
            return non_colliding[0]

        # Fallback
        return self.rng.choice(env.available_actions(s))

    def _peek(self, env: GridMDP, s: State, a: str):
        # Return next state without applying penalties, but observe collisions
        dr, dc = env.action_map[a]
        nr, nc = s.r + dr, s.c + dc
        if not env.in_bounds(nr, nc) or env.is_wall(nr, nc):
            return None, None, None
        return (type(s))(nr, nc), None, None
