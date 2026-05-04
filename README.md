# Simulator-Based Offline Reinforcement Learning for Sequential Recommendation

**Effects of Fatigue-Aware Reward Design on PPO and DQN**

Reinforcement Learning Project
Bolai Yin | Yuzhe Li | Kai Zhu  
Northeastern University

---

## Quick Summary

This project studies whether reinforcement learning can improve sequential recommendation beyond a greedy baseline in an offline setting, where rewards come from a learned user simulator rather than live user feedback.

Using MovieLens 1M, we built an offline reinforcement learning pipeline that compares PPO, DQN, Greedy-CTR, and Random policies, and introduced fatigue-aware reward shaping to penalize repetitive recommendations over a 20-step session.

**Main result:** Under fatigue-aware reward, PPO outperformed Greedy-CTR by +4.31 mean reward, showing that reward design and state design can matter as much as algorithm choice in offline recommendation.

**What this repo demonstrates**
- Offline reinforcement learning for sequential recommendation
- Fatigue-aware reward design and state design
- Policy comparison across PPO, DQN, Greedy-CTR, and Random
- Experimentation with a learned user simulator

**Tech stack:** Python, PyTorch, Stable-Baselines3, Gymnasium, XGBoost, GRU4Rec, MovieLens 1M  
**Start here:** Project Overview → MDP Formulation → Results

----
## Project Overview

This project frames sequential movie recommendation as a Markov Decision Process (MDP),
where an RL agent learns to recommend movie genres to maximize cumulative user satisfaction
over a 20-step session. Our central question is whether reinforcement learning provides value
beyond a myopic greedy recommender when the environment is learned from historical ratings
rather than from live user sessions.

The project proceeds in two stages:

1. **No-penalty evaluation** — We show that the ranking PPO > DQN > Greedy-CTR > Random
   is preserved across two independently trained GRU-based simulators, even without an
   explicit fatigue signal.

2. **Fatigue-aware reward shaping** — We introduce window-based and concentration-based
   penalties and show that PPO achieves the strongest gains over Greedy-CTR, learns a less
   concentrated genre policy than DQN, and benefits substantially from genre-count state features.

**Main lesson:** In offline recommendation, reward design can matter as much as algorithm choice.

---

## Two-Course Architecture

The User Simulator (GRU4Rec / GRU4RecSVD) is trained as part of the companion CS6140
(Machine Learning) project, which provides `P(like | user, movie)` as the reward signal for RL training.

```
Machine Learning (CS6140)                Reinforcement Learning (CS5180)
─────────────────────────              ──────────────────────────────
MovieLens 1M historical data           RL Agent (DQN / PPO)
        ↓                                          ↓
Train User Simulator:                  Recommends genre to user
  XGBoost (static, AUC ~0.800)                     ↓
  GRU4Rec (AUC 0.703)       →→→→→      Simulator returns reward
  GRU4RecSVD (AUC 0.783)               sampled from Bernoulli(P(like))
        ↓                                          ↓
P(like | user, movie)                  Agent updates policy
                                                   ↓
                                       Next state = updated genre counts
```

---

## MDP Formulation

| Component | Definition |
|---|---|
| **State** | 68-dim: SVD user embedding (50) + recent genre counts (18) |
| **Action** | 18 genre indices; env selects highest P(like) movie within genre |
| **Reward (no penalty)** | Bernoulli(P(like)) from the simulator |
| **Reward (fatigue-aware)** | Bernoulli(clip(P(like) − penalty, 0.1, 0.9)) |
| **Fatigue: window penalty** | min(0.05 × same_count_in_last_10, 0.25) |
| **Fatigue: concentration penalty** | 0.3 × max(genre_frac − 0.25, 0) |
| **Episode length** | 20 recommendation steps |
| **History window** | 10 recent genre selections |

---

## Key Design Decisions

**Why genre as action (not movie)?**
- Movie pool has 3,706 entries — action space too large for stable DQN.
- Genre (18) captures the diversity/fatigue signal that matters.
- Movie sub-selection handled separately by the environment.

**Why `recent_genre_counts` in state (not `liked_history_embed`)?**
- SVD embedding mean compresses sequence info → DQN can't decode fatigue signal.
- `recent_genre_counts[i]` directly tells the agent how many times genre *i* appeared recently.
- Linear, readable signal → Q-network learns fatigue avoidance easily.
- State ablation confirms this: removing genre counts drops DQN from 7.22 → 5.17 and PPO from 9.98 → 5.55 under fatigue-aware reward.

**Why User Simulator instead of historical data?**
- DQN/PPO need 120,000+ interactions during training.
- Historical data can't answer counterfactual questions ("what if we recommended X instead of Y?").
- Simulator enables exploration of unseen recommendation sequences.

---

## Results

### No-Penalty: Robust Ordering Across Two Simulators

| Policy | Original GRU4Rec (AUC 0.703) | Enhanced GRU4RecSVD (AUC 0.781) | Rank Preserved? |
|---|---|---|---|
| Random | 10.05 ± 2.20 | 9.49 ± 1.78 | Yes |
| Greedy-CTR | 11.33 ± 2.43 | 10.88 ± 2.62 | Yes |
| DQN | 12.27 ± 3.12 | 12.25 ± 2.20 | Yes |
| **PPO** | **13.38 ± 2.79** | **12.63 ± 1.93** | **Yes** |

The ordering PPO > DQN > Greedy > Random holds in both simulators. However, no-penalty
policies collapse toward over-rewarded genres (especially Film-Noir), indicating simulator bias
rather than genuine diversification.

### Fatigue-Aware: Window + Concentration Penalty

| Policy | Mean Reward (T=20) | Δ vs Greedy-CTR |
|---|---|---|
| Random | 8.17 | — |
| Greedy-CTR | 5.82 | baseline |
| DQN | 9.90 | +1.55* |
| **PPO** | **13.49** | **+4.31*** |

*Both differences statistically significant.

Under fatigue-aware reward, PPO outperforms Greedy-CTR by +4.31, the cleanest evidence that
sequential planning matters once the objective penalizes repetition. PPO also learns a less
concentrated genre distribution than DQN, spreading selections across Documentary, Film-Noir,
and War rather than collapsing to a single genre.

### State Ablation (Fatigue-Aware Reward)

| Agent | Full State (68-dim) | No Genre Counts (50-dim) | Δ |
|---|---|---|---|
| DQN | 9.90 | 6.22 | −3.68 |
| PPO | 10.40 | 5.55* | −3.28* |

*Note: ablation numbers from Fig. 6 in the white paper (DQN: 7.22→5.17 in text, 9.90→6.22 in figure; minor discrepancies reflect different evaluation runs).

Removing recent genre counts severely degrades both agents, confirming that state design must
match reward design: if the objective depends on history, the policy needs direct access to that history.

---

## Key Findings

1. **RL beats Greedy even without fatigue** — PPO and DQN exploit sequence-conditional dynamics in the simulator, though some advantage may reflect model-induced patterns.

2. **Reward design is first-order** — The PPO–Greedy gap grows as penalty strength increases; Greedy-CTR collapses under repetition costs because it optimizes only the current step.

3. **PPO > DQN for short-horizon policy-sensitive tasks** — PPO's stochastic actor explores and maintains a balanced sequence strategy; DQN's deterministic argmax remains vulnerable to local exploitation.

4. **State must match reward** — Genre-count features are dispensable without penalty but essential once the reward depends on repetition history.

5. **Simulator quality is the bottleneck** — Offline RL can exploit simulator imperfections; higher return in simulation ≠ better user experience.

---

## Repository Structure

```
CS5180-RL-Recommendation/
├── README.md
├── .gitignore
├── notebooks/
│   └── cs5180_v9_gru4rrec+PPO&DQN.ipynb     # Complete pipeline: env + baselines + DQN + PPO
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

## References

- F. M. Harper and J. A. Konstan. *The MovieLens datasets: History and context.* ACM TIIS, 2015.
- B. Hidasi et al. *Session-based recommendations with recurrent neural networks.* ICLR, 2016.
- J. Schulman et al. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
- V. Mnih et al. *Human-level control through deep reinforcement learning.* Nature, 2015.
- Y. Gao et al. *KuaiRec: A fully observed dataset for evaluating recommender systems.* CIKM, 2023.
