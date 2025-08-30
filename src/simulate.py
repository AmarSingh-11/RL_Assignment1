
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import random

from maze_env import GridMDP, State

@dataclass
class EpisodeResult:
    steps: int
    total_reward: float
    reached_goal: bool
    trajectory: List[Tuple[State, str, float]]  # (state_before, action, reward)

def simulate_episode(env: GridMDP, policy, max_steps: int = 200, seed: int = 0) -> EpisodeResult:
    rng = random.Random(seed)
    s = env.start
    total_reward = 0.0
    traj = []

    for t in range(max_steps):
        if env.is_terminal(s):
            return EpisodeResult(steps=t, total_reward=total_reward, reached_goal=True, trajectory=traj)
        a = policy.action(env, s)
        ns, r, done = env.transition(s, a)
        traj.append((s, a, r))
        total_reward += r
        s = ns
        if done:
            return EpisodeResult(steps=t+1, total_reward=total_reward, reached_goal=True, trajectory=traj)
    # max steps reached
    return EpisodeResult(steps=max_steps, total_reward=total_reward, reached_goal=False, trajectory=traj)
