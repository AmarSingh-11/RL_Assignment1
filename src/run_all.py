
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from maze_env import GridMDP
from policies import RandomPolicy, GreedyManhattanPolicy
from simulate import simulate_episode

BASE = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

# ---------- Configuration (you can change these) ----------
N = 6
# Walls as (row, col) with (0,0) at TOP-LEFT. Keep a solvable layout.
WALLS = {
    (1,1),(1,2),(1,4),
    (2,4),
    (3,1),(3,2),(3,4),
    (4,4),
}
START = (N-1, 0)       # bottom-left
GOAL = (0, N-1)        # top-right
STEP_REWARD = -1.0
WALL_PENALTY = -2.0
GOAL_REWARD = 10.0
GAMMA = 0.9
EPISODES = 20
MAX_STEPS = 200
SEED = 7

def make_env():
    return GridMDP(N=N, walls=WALLS, start=START, goal=GOAL,
                   step_reward=STEP_REWARD, wall_penalty=WALL_PENALTY,
                   goal_reward=GOAL_REWARD, gamma=GAMMA)

def run_policy(name, policy_ctor):
    env = make_env()
    policy = policy_ctor()
    rows = []
    sample_traj = None
    successes = 0

    for ep in range(EPISODES):
        res = simulate_episode(env, policy, max_steps=MAX_STEPS, seed=SEED + ep)
        rows.append({
            "episode": ep+1,
            "steps": res.steps,
            "total_reward": res.total_reward,
            "reached_goal": int(res.reached_goal),
        })
        if sample_traj is None and res.reached_goal:
            sample_traj = res.trajectory
        if res.reached_goal:
            successes += 1

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, f"stats_{name}.csv"), index=False)

    # Save sample trajectory
    traj_path = os.path.join(RESULTS, f"sample_trajectory_{name}.txt")
    with open(traj_path, "w") as f:
        f.write(f"Sample trajectory for {name} policy (state_before -> action -> reward):\n\n")
        if sample_traj is None:
            f.write("(No successful episode to sample; showing first episode steps)\n")
            # Use first episode's trajectory
            env = make_env()
            policy = policy_ctor()
            res = simulate_episode(env, policy, max_steps=MAX_STEPS, seed=SEED)
            sample_traj = res.trajectory[:]
        for (s, a, r) in sample_traj:
            f.write(f"{(s.r, s.c)} -> {a} -> {r:.2f}\n")

    # Figures (do not set colors or styles)
    plt.figure()
    plt.title(f"Steps per Episode - {name}")
    plt.plot(df["episode"], df["steps"], marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f"steps_{name}.png"))
    plt.close()

    plt.figure()
    plt.title(f"Total Reward per Episode - {name}")
    plt.plot(df["episode"], df["total_reward"], marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f"rewards_{name}.png"))
    plt.close()

    summary = {
        "policy": name,
        "episodes": EPISODES,
        "successes": successes,
        "success_rate": successes / EPISODES,
        "steps_min": float(df["steps"].min()),
        "steps_max": float(df["steps"].max()),
        "steps_avg": float(df["steps"].mean()),
        "reward_min": float(df["total_reward"].min()),
        "reward_max": float(df["total_reward"].max()),
        "reward_avg": float(df["total_reward"].mean()),
    }
    with open(os.path.join(RESULTS, f"summary_{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return df, summary, traj_path

def save_grid_image(path):
    # Visualize grid layout
    import matplotlib.pyplot as plt
    import numpy as np

    grid = np.zeros((N, N))
    for (r, c) in WALLS:
        grid[r, c] = 1  # wall

    plt.figure(figsize=(4,4))
    plt.imshow(grid, interpolation="none")
    plt.title("Maze Layout (1=Wall, 0=Free)")
    plt.xticks(range(N))
    plt.yticks(range(N))
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    os.makedirs(RESULTS, exist_ok=True)
    grid_img = os.path.join(RESULTS, "grid.png")
    save_grid_image(grid_img)

    random_df, random_summary, random_traj = run_policy("random", lambda: RandomPolicy(seed=42))
    greedy_df, greedy_summary, greedy_traj = run_policy("greedy", lambda: GreedyManhattanPolicy(seed=123))

    # Combined CSV
    random_df["policy"] = "random"
    greedy_df["policy"] = "greedy"
    combined = pd.concat([random_df, greedy_df], ignore_index=True)
    combined.to_csv(os.path.join(RESULTS, "stats_combined.csv"), index=False)

    # Combined summaries
    combined_summary = {"random": random_summary, "greedy": greedy_summary}
    with open(os.path.join(RESULTS, "summary_combined.json"), "w") as f:
        json.dump(combined_summary, f, indent=2)

    print("Done. Results written to:", RESULTS)

if __name__ == "__main__":
    main()
