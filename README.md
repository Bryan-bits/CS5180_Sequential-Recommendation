# CS5180 Reinforcement Learning-Based Fatigue-Aware Recommendation

A reinforcement learning-based movie recommendation system built on MovieLens 1M,
developed as the course project for CS5180 (Reinforcement Learning) at Northeastern University.

---

## Project Overview

This project frames sequential movie recommendation as a Markov Decision Process (MDP),
where an RL agent learns to recommend movie genres to maximize cumulative user satisfaction
over a session. We investigate whether a sequence-aware user simulator (GRU4Rec) captures
recommendation fatigue and whether RL agents can exploit that signal.

The User Simulator (GRU4Rec) is trained as part of the companion CS6140 (Machine Learning)
project, which provides `P(like | user, movie)` as the reward signal for RL training.

---

## Two-Course Architecture

```
Machine Learning (CS6140)                Reinforcement Learning (CS5180)
──────────────────────────               ──────────────────────────────
MovieLens 1M historical data             RL Agent (DQN / PPO)
        ↓                                          ↓
Train User Simulator:                    Recommends genre to user
  XGBoost (static baseline)                        ↓
  GRU4Rec (sequence-aware)   →→→→→     Simulator returns reward
        ↓                                sampled from Bernoulli(P(like))
P(like | user, movie)                              ↓
                                         Agent updates policy
                                                   ↓
                                         Next state = updated genre counts
```

---

## MDP Formulation

| Component      | Definition                                                         |
|----------------|--------------------------------------------------------------------|
| **State**      | 68-dim: SVD user embedding (50) + recent genre counts (18)        |
| **Action**     | 18 genre indices; env selects best movie within genre via history  |
| **Reward**     | Bernoulli(clip(P(like), 0.1, 0.9))                                |
| **Episode**    | T = 20 recommendation steps per user session                      |
| **Transition** | State updates via recent_genre_counts sliding window (w=10)       |

---

## Key Design Decisions

**Why genre as action (not movie)?**
Movie pool has 3,706 entries — too large for stable DQN. Genre (18) reduces the action
space while capturing the diversity signal. Movie sub-selection is handled separately
by the environment using historical user ratings.

**Why `recent_genre_counts` in state?**
SVD embedding mean compresses sequence info, making it hard for the Q-network to detect
repetition. `recent_genre_counts[i]` directly encodes how often genre i appeared in the
recent window — a linear, readable signal for fatigue avoidance.

**Why User Simulator instead of historical data?**
DQN needs 120,000+ interactions during training. Historical data can't answer counterfactual
questions ("what if we recommended X instead of Y?"). A simulator enables exploration of
unseen recommendation sequences.

---

## Results (T=20, 500 evaluation episodes)

### Bernoulli Reward (sampled)
```
Policy                    Mean     Std
----------------------------------------------
Random                   10.21 ±  2.46
Greedy-CTR               11.50 ±  2.78
DQN                      11.21 ±  2.73
PPO                      12.78 ±  2.42

DQN vs Greedy-CTR : -0.28  (p=0.106, not significant)
PPO vs Greedy-CTR : +1.29  (p<0.001, significant)
```

### Expected Reward (P(like) accumulated, no Bernoulli noise)
```
Policy                    Mean     Std
----------------------------------------------
Random                   10.12 ±  1.24
Greedy-CTR               11.39 ±  1.95
DQN                      11.11 ±  1.76
PPO                      12.90 ±  1.34

PPO vs Greedy-CTR (expected) : +1.51
```

### Ablation: State without Genre Counts (50-dim)
```
Policy                    Full State    No Genre Counts
-------------------------------------------------------
DQN                       11.21         11.96
PPO                       12.78         12.76
```

DQN performs *worse* with genre counts, suggesting it uses the signal
counterproductively. PPO's performance is unchanged, indicating it relies
entirely on user embeddings for genre selection.

---

## Key Findings

**GRU4Rec does not capture genre fatigue.**
Fatigue probing shows P(like) is flat across k=0 to k=4 same-genre repetitions
(~0.51 ± 0.15). Paired comparison (same-genre vs mixed-genre history) yields
a mean delta of -0.0007 — no fatigue signal exists in the simulator.

**DQN exhibits genre concentration (reward hacking).**
DQN allocates 80% of recommendations to Comedy, exploiting the genre's large
movie pool rather than learning a diversity strategy. This causes it to
underperform Greedy-CTR.

**PPO's advantage comes from preference learning, not fatigue avoidance.**
PPO significantly outperforms Greedy-CTR (+1.29), but the ablation study
shows this advantage is preserved without genre counts. PPO learns better
user-genre preferences from SVD embeddings alone.

**The simulator is the bottleneck.**
GRU4Rec achieves 0.72 AUC — modest for a reward model. The absence of a
fatigue signal means RL agents cannot learn fatigue avoidance regardless of
architecture. Improving the simulator (or adding explicit fatigue modeling)
is the highest-leverage next step.

---

## Repository Structure

```
CS5180-RL-Recommendation/
├── .gitignore
├── README.md
├── main.ipynb                  # 11-cell notebook: load → EDA → train → eval
│
├── data_setup.py               # Data pipeline (download, load, SVD, XGBoost)
├── eda.py                      # EDA functions + derived feature generation
├── gru4rec.py                  # GRU4Rec model, training, inference, fatigue eval
├── env.py                      # RL environment + ablation variant
├── baselines.py                # Random, Greedy-CTR, expected-reward helpers
├── evaluate.py                 # Evaluation pipeline, plots, save artifacts
│
├── experiments/
│   ├── __init__.py
│   └── tune_xgb.py            # Optuna hyperparameter tuning for XGBoost
│
├── eda/                        # Generated by run_all_eda()
│   ├── 01_rating_distribution.png
│   ├── ...
│   └── 09_policy_comparison.png
│
├── outputs/                    # Generated by save_all_artifacts()
│   ├── simulators/             # xgb_simulator.json, gru4rec_best.pt
│   ├── embeddings/             # U_embed.npy, M_embed.npy
│   └── agents/                 # dqn_v8.zip, ppo_v8.zip, ablation models
│
└── ml-1m/                      # MovieLens 1M (auto-downloaded, gitignored)
```

---

## Setup

**Quick Start**

```bash
pip install numpy pandas scipy xgboost scikit-learn gymnasium stable-baselines3 torch seaborn optuna
```

Then open `main.ipynb` and run cells 1–11. Cell 1 auto-downloads MovieLens 1M
if the `ml-1m/` folder doesn't exist.

**From Scratch (no pretrained weights)**

In Cell 4 of the notebook, use Option B to train GRU4Rec:
```python
from gru4rec import train_gru4rec, make_gru4rec_p_like
gru4rec_model  = train_gru4rec(ctx, save_dir="outputs/simulators", n_epochs=50)
gru4rec_p_like = make_gru4rec_p_like(gru4rec_model, movie2idx, T=20)
```

**With Pretrained Weights**

Place checkpoint files in `outputs/simulators/` and `outputs/embeddings/`,
then use Option A in Cell 4:
```python
from gru4rec import load_gru4rec, make_gru4rec_p_like
gru4rec_model  = load_gru4rec("outputs/simulators/gru4rec_best.pt", n_movies=len(movie2idx))
gru4rec_p_like = make_gru4rec_p_like(gru4rec_model, movie2idx, T=20)
```
