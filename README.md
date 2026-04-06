# CS5180 Reinforcement Learning-Based Fatigue-Aware Recommendation

A reinforcement learning-based movie recommendation system built on MovieLens 1M,
developed as the course project for CS5180 (Reinforcement Learning) at Northeastern University.

---

## Project Overview

This project frames sequential movie recommendation as a Markov Decision Process (MDP),
where an RL agent learns to recommend movie genres to maximize cumulative user satisfaction
over a session, while avoiding recommendation fatigue.

The User Simulator (GRU4Rec) is trained as part of the companion CS6140 (Machine Learning)
project, which provides `P(like | user, movie)` as the reward signal for RL training.

---

## Two-Course Architecture

```
Machine Learning                         Reinforcement Learning
─────────────────────────              ──────────────────────────────
MovieLens 1M historical data           RL Agent (DQN / PPO)
        ↓                                          ↓
Train User Simulator:                  Recommends genre to user
  XGBoost (static baseline)                        ↓
  GRU4Rec (sequence-aware)   →→→→→   Simulator returns reward
        ↓                              sampled from Bernoulli(P(like))
P(like | user, movie)                              ↓
                                       Agent updates policy
                                                   ↓
                                       Next state = updated genre counts
```

---

## MDP Formulation

| Component | Definition |
|---|---|
| **State** | 68-dim: SVD user embedding (50) + recent genre counts (18) |
| **Action** | 18 genre indices; env selects highest P(like) movie within genre |
| **Reward** | Bernoulli(P(like)) − fatigue penalty (hand-crafted linear rule) |
| **Episode** | T = 20 recommendation steps per user session |
| **Transition** | State updates via recent_genre_counts sliding window |

---

## Key Design Decisions

**Why genre as action (not movie)?**
- Movie pool has 3,706 entries — action space too large for stable DQN
- Genre (18) captures the diversity/fatigue signal that matters
- Movie sub-selection handled separately by env

**Why `recent_genre_counts` in state (not `liked_history_embed`)?**
- SVD embedding mean compresses sequence info → DQN can't decode fatigue signal
- `recent_genre_counts[i]` directly tells DQN how many times genre i appeared recently
- Linear, readable signal → Q-network learns fatigue avoidance easily

**Why User Simulator instead of directly using historical data?**
- DQN needs 120,000+ interactions during training
- Historical data can't answer counterfactual questions
  ("what if we recommended X instead of Y?")
- Simulator enables exploration of unseen recommendation sequences

---

## Results (T=20, 200 evaluation episodes)

### Bernoulli Reward (sampled)
```
Policy                         Mean     Std
------------------------------------------------------------
Random                        10.95 ±  2.20
Greedy-CTR                    12.28 ±  2.43
DQN                           12.44 ±  2.54

DQN vs Greedy-CTR : +0.16
```

### Expected Reward (p_like accumulated, no Bernoulli noise)
```
Policy                         Mean     Std
------------------------------------------------------------
Random                        10.978 ±  0.987
Greedy (expected)             12.163 ±  1.338
DQN (expected)                12.736 ±  1.396

DQN vs Greedy (expected) : +0.572
```

**Finding:** Bernoulli sampling introduces variance that masks DQN's true policy advantage.
Expected reward is the primary evaluation metric.

---

## Genre Preference Analysis (DQN vs Greedy-CTR, 500 steps)

```
Genre             DQN    CTR    Diff
----------------------------------------
Fantasy           194      0    +194  ←  DQN concentrates here
War               160     40    +120  ←
Film-Noir         111    120      -9
Animation           0    100    -100  ←  CTR concentrates here
Musical             0     60     -60  ←
Thriller            0     60     -60  ←
Adventure          16     40     -24
```

DQN exhibits genre concentration (reward hacking) — exploiting simulator bias
rather than learning a generalizable diversity strategy.

---

## Repository Structure

```
CS5180-RL-Recommendation/
├── README.md
├── .gitignore
├── notebooks/
│   └── cs5180_v6_final.ipynb     # Complete pipeline: env + baselines + DQN + PPO
└── data/
    └── README.md                 # Instructions to download MovieLens 1M
```

---

## Setup

**Dataset**

Download [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) and place
the `.dat` files in your Google Drive:

```
/content/drive/MyDrive/cs5180/
    movies.dat
    ratings.dat
    users.dat
```

**Dependencies**

```bash
pip install numpy pandas scipy xgboost scikit-learn gymnasium stable-baselines3 torch
```

**Running**

1. Mount Google Drive and load `movies`, `ratings`, `users` DataFrames
2. Run all `[DATA SETUP]` cells top to bottom (required on every Colab restart)
3. Load GRU4Rec simulator weights from Drive (`gru4rec_best.pt`)
4. Run `[MODEL/ALGO]` cells for DQN / PPO training and evaluation

---

## .gitignore

```
*.pkl
*.pt
*.zip
*.npy
*.dat
__pycache__/
.ipynb_checkpoints/
```
