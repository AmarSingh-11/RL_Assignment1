Perfect 🚀 — here’s the updated **README.md template** with a **badges section** at the top. These badges make your GitHub repo look polished and professional:

```markdown
# RL Assignment 1 – Grid Maze as an MDP  

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)  
![Status](https://img.shields.io/badge/status-completed-success.svg)  
![License](https://img.shields.io/badge/license-MIT-green.svg)  

**Author:** Amar Singh (2022UG000009)  
**Course Outcome (CO1):** Apply core reinforcement learning concepts to solve real-world problems.  

---

## 📌 Project Overview
This project implements a **6×6 grid maze** environment formulated as a **Markov Decision Process (MDP)**.  
Two agents were implemented and compared:  
- **Random Policy Agent** (chooses actions uniformly at random)  
- **Greedy Manhattan Policy Agent** (chooses actions that reduce Manhattan distance to the goal)  

The project simulates **20 episodes per policy**, logs trajectories, computes statistics, and provides analysis of performance.  

---

## 🏗️ MDP Specification
- **States (S):** Each free cell in the grid, represented as `(row, col)`  
- **Actions (A):** Up, Down, Left, Right  
- **Transitions (p(s′|s,a)):** Deterministic  
  - Valid move → move to next cell  
  - Invalid move (off-grid or wall) → stay in same cell  
  - Goal → terminal absorbing state  
- **Rewards (r(s,a,s′)):**  
  - –1 per step  
  - –2 if bump into wall/off-grid  
  - +10 when reaching the goal  
- **Discount Factor (γ):** 0.9  
- **Start:** `(5,0)` (bottom-left)  
- **Goal:** `(0,5)` (top-right)  
- **Walls:** (1,1), (1,2), (1,4), (2,4), (3,1), (3,2), (3,4), (4,4)  

---

## 📂 Project Structure
```

RL\_Assignment1/
│
├── src/                  # Source code
│   ├── maze\_env.py       # Grid environment & MDP definition
│   ├── policies.py       # Random and Greedy policy implementations
│   ├── simulate.py       # Episode simulation functions
│   └── run\_all.py        # Main script to run experiments
│
├── results/              # Output of simulations
│   ├── stats\_random.csv          # Per-episode stats for Random policy
│   ├── stats\_greedy.csv          # Per-episode stats for Greedy policy
│   ├── summary\_random.json       # Min/Max/Avg stats (Random)
│   ├── summary\_greedy.json       # Min/Max/Avg stats (Greedy)
│   ├── sample\_trajectory\_random.txt
│   ├── sample\_trajectory\_greedy.txt
│   ├── steps\_random.png          # Plot: steps per episode (Random)
│   ├── steps\_greedy.png          # Plot: steps per episode (Greedy)
│   ├── rewards\_random.png        # Plot: rewards per episode (Random)
│   ├── rewards\_greedy.png        # Plot: rewards per episode (Greedy)
│   └── comparison\_charts.pdf     # Side-by-side plots (success rate, avg steps, avg rewards)
│
├── report/
│   ├── Assignment1\_Final\_Report.pdf
│   └── Assignment1\_Report\_Compact\_3pg.pdf
│
└── README.md             # This file

````

---

## ▶️ How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/AmarSingh-11/RL_Assignment1.git
   cd RL_Assignment1
````

2. **Run all experiments**

   ```bash
   python3 src/run_all.py
   ```

3. **Outputs will be saved under `results/`**

   * Per-episode stats in CSV
   * Summary stats in JSON
   * Plots in PNG
   * Sample trajectories in TXT

---

## 📊 Results Summary

### Random Policy (20 episodes)

* Success rate: **60% (12/20)**
* Steps: min = **34**, max = **200**, avg = **141.5**
* Rewards: min = **–287**, max = **–37**, avg = **–188.2**

### Greedy Policy (20 episodes)

* Success rate: **100% (20/20)**
* Steps: always **10**
* Rewards: always **+1**

---

## 🔍 Analysis & Exploration

* **Wall placement**: Forces detours; increases steps and lowers reward if direct greedy path is blocked.
* **Reward values**: Step penalty (–1) encourages efficiency, bump penalty (–2) discourages hitting walls, +10 goal reward ensures motivation.
* **Random policy**: Explores widely, sometimes finds paths, but inefficient (low reward, long steps).
* **Greedy policy**: Fast and reliable in this maze (always succeeds in 10 steps), but short-sighted in more complex layouts.

---

## 🚀 Suggested Improvements

* **ε-Greedy policy**: Mostly greedy, but sometimes random to escape traps.
* **Reward shaping**: Add small positive reward for reducing distance to goal.
* **Loop penalty**: Penalize revisiting states to avoid cycles.
* **Lookahead planning**: Consider next two steps to avoid dead ends.

---

## 📑 Deliverables

* ✅ Well-commented source code (`src/`)
* ✅ Simulation results (`results/`)
* ✅ Final compact report (`report/Assignment1_Final_Report.pdf`)

