# CS5180 Reinforcement Learning-Based Fatigue-Aware Recommendation

A reinforcement learning-based movie recommendation system built on MovieLens 1M,
developed as the course project for CS5180 (Reinforcement Learning) at Northeastern University.

---

## Project Overview

This project frames sequential movie recommendation as a Markov Decision Process (MDP),
where an RL agent learns to recommend movie genres to maximize cumulative user satisfaction
over a session while avoiding recommendation fatigue.

We investigate whether a sequence-aware user simulator (GRU4Rec) captures recommendation
fatigue, design an explicit fatigue penalty when it doesn't, and compare how value-based
(DQN) and policy-gradient (PPO) methods handle diversity-requiring reward signals.

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
| **Reward**     | Bernoulli(clip(P(like) − fatigue_penalty, 0.1, 0.9))             |
| **Episode**    | T = 20 recommendation steps per user session                      |
| **Transition** | State updates via recent_genre_counts sliding window (w=10)       |

### Fatigue Penalty Design

The reward includes two penalty components:

**Window penalty:** penalizes recent same-genre repetition within a sliding window.
`window_penalty = min(0.05 × same_genre_count_in_window, 0.25)`

**Concentration penalty:** penalizes session-level genre dominance above 25% share.
`concentration_penalty = 0.3 × max(genre_fraction − 0.25, 0)`

These were added after probing revealed GRU4Rec does not inherently capture fatigue.

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
DQN needs 200,000+ interactions during training. Historical data can't answer counterfactual
questions ("what if we recommended X instead of Y?"). A simulator enables exploration of
unseen recommendation sequences.

**Why explicit fatigue penalty?**
Fatigue probing (see findings below) showed GRU4Rec does not encode genre fatigue.
Without explicit penalty, RL agents exploit high-like-rate genres rather than diversifying.

---

## Results (T=20, 500 evaluation episodes)

### Bernoulli Reward (sampled)
```
Policy                    Mean     Std
----------------------------------------------
Random                    7.66 ±  2.41
Greedy-CTR                5.67 ±  2.51
DQN                       7.22 ±  2.95
PPO                       9.98 ±  2.54

DQN vs Greedy-CTR : +1.55  (p<0.001, significant)
PPO vs Greedy-CTR : +4.31  (p<0.001, significant)
```

### Expected Reward (P(like) accumulated, no Bernoulli noise)
```
Policy                    Mean     Std
----------------------------------------------
Random                    7.95 ±  1.27
Greedy-CTR                5.68 ±  1.64
DQN                       6.84 ±  1.87
PPO                       9.91 ±  1.47

PPO vs Greedy-CTR (expected) : +4.23
```

### Ablation: State without Genre Counts (50-dim)
```
Policy                    Full State    No Genre Counts    Δ
-------------------------------------------------------------
DQN                         7.22            5.17         +2.05
PPO                         9.98            5.55         +4.43
```

Both agents rely heavily on genre counts. PPO's larger drop (+4.43)
shows it uses the fatigue signal more effectively than DQN (+2.05).

### Fatigue Probe
```
k (same-genre in history)    P(like)
--------------------------------------
0                            0.435
1                            0.391
2                            0.333
3                            0.267
4                            0.206

Δ (k=0 → k=4): 0.228
Paired probe: 97.5% of pairs show mixed > same history
Mean delta: 0.231
```

### Genre Selection Distribution
```
Genre           DQN%     PPO%
---------------------------------
Children's     46.0%     1.8%   ← DQN concentrates
War            25.6%    19.7%
Film-Noir       8.3%    27.0%
Documentary     0.3%    31.0%   ← PPO favors
Thriller        9.4%     0.9%
(12 others)    10.4%    19.6%
```

---

## Reward Shaping Experiment

We tested four penalty conditions to study how reward design affects agent behavior.
Full experiment in `experiments/reward_shaping.ipynb`.

### Cross-Condition Summary (T=20, 500 episodes)
```
Condition               Random  Greedy     DQN     PPO  PPO vs Greedy
-----------------------------------------------------------------------
No penalty                9.67   11.31   13.14   13.30         +1.99
Window only               7.91    7.47    9.53   10.58         +3.11
Window + concentration    7.77    5.72    9.73    9.96         +4.23
Aggressive                6.77    3.63    8.04    9.00         +5.37
```

PPO's advantage over Greedy-CTR grows monotonically with penalty strength
(+1.99 → +5.37), confirming it adapts to diversity constraints rather than
just exploiting base like-rates.

### Fatigue Signal by Condition
```
Condition                P(like) k=0   P(like) k=4    Δ
----------------------------------------------------------
No penalty                 0.489         0.479       0.009
Window only                0.445         0.293       0.152
Window + concentration     0.451         0.223       0.228
Aggressive                 0.405         0.115       0.289
```

No penalty: flat (no fatigue signal). Each successive penalty creates a steeper
decline, giving RL agents a stronger signal to learn from.

### Genre Concentration by Condition (max genre fraction)
```
Condition               DQN      PPO
----------------------------------------
No penalty              63.3%    62.0%   ← both exploit Film-Noir
Window only             28.3%    29.5%   ← diversifies to 3 genres
Window + concentration  35.9%    31.0%   ← further spread
Aggressive              21.2%    25.6%   ← most diverse
```

---

## Key Findings

**1. GRU4Rec does not capture genre fatigue natively.**
Without explicit penalty, P(like) is flat across same-genre repetitions (Δ=0.009).
Both agents exploit Film-Noir at 63%. The explicit fatigue penalty is necessary
to create a learnable diversity signal.

**2. PPO significantly outperforms all baselines across all conditions.**
PPO beats Greedy-CTR in every condition, and its relative advantage grows as
the penalty strengthens (+1.99 → +3.11 → +4.23 → +5.37). This confirms PPO
genuinely learns to handle diversity constraints, not just exploit base like-rates.

**3. DQN improves with penalty but remains structurally limited.**
DQN beats Greedy-CTR in all conditions but concentrates more than PPO. As a
value-based method, it converges to a deterministic policy and cannot express
stochastic genre mixing. Under the strongest penalty (Condition 3 in main.ipynb),
DQN can even underperform Random when concentration triggers are severe.

**4. Diversity-requiring objectives favor stochastic policies.**
This is the core structural insight: value-based methods (DQN) converge to a single
best action per state and cannot express "Film-Noir 20%, War 15%, Documentary 15%."
Policy-gradient methods (PPO) maintain stochastic policies that naturally distribute
selections across genres while favoring high-value ones. This makes PPO structurally
better suited for recommendation settings with fatigue or diversity constraints.

**5. Reward design matters more than algorithm choice.**
Across four penalty conditions, the choice of penalty had a larger impact on agent
behavior than the choice of algorithm. Greedy-CTR collapses from 11.31 to 3.63
as penalty strengthens, while PPO adapts from 13.30 to 9.00. Simulator quality
and reward shaping determine the ceiling of RL performance.

---

## Repository Structure

```
CS5180-RL-Recommendation/
├── .gitignore
├── README.md
├── main.ipynb                  # 14-cell notebook: load → EDA → train → eval → analysis
│
├── data_setup.py               # Data pipeline (download, load, SVD, XGBoost)
├── eda.py                      # EDA functions + derived feature generation
├── gru4rec.py                  # GRU4Rec model, training, inference, fatigue eval
├── env.py                      # RL environment + ablation variant
├── baselines.py                # Random, Greedy-CTR, expected-reward helpers
├── evaluate.py                 # Evaluation pipeline, plots, save artifacts
│
├── experiments/
│   ├── reward_shaping.ipynb    # 4-condition penalty comparison experiment
│   ├── exp1_no_penalty.png
│   ├── exp2_window_only.png
│   ├── exp3_window_concentration.png
│   ├── exp4_aggressive.png
│   ├── exp_fatigue_comparison.png
│   └── exp_summary.png
│
├── utils/
│   └── tune_xgb.py            # Optuna hyperparameter tuning for XGBoost
│
├── eda/                        # Generated by run_all_eda() + analysis cells
│   ├── 01_rating_distribution.png
│   ├── ...
│   ├── 09_policy_comparison.png
│   ├── 10_genre_distribution.png
│   ├── 11_ablation_comparison.png
│   └── 12_fatigue_analysis.png
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

Then open `main.ipynb` and run cells 1–14. Cell 1 auto-downloads MovieLens 1M
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