# Project Abstract

## Simulator-Based Offline Reinforcement Learning for Sequential Recommendation

**Subtitle:** Effects of Fatigue-Aware Reward Design on PPO and DQN

**Academic Project:** Reinforcement Learning · Northeastern University  
**Team:** Bryan Yin · Yuzhe Li · Kai Zhu  
**Dataset:** [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)

---

## Abstract

This project studies whether reinforcement learning can improve sequential recommendation beyond a greedy baseline in an offline setting, where rewards come from a learned user simulator rather than live user interaction.

Using MovieLens 1M, we built an offline reinforcement learning pipeline in which an agent recommends movie genres over a 20-step session, receives rewards from a learned simulator, and updates its policy through repeated interaction with the simulated environment. We compare PPO, DQN, Greedy-CTR, and Random policies, and introduce fatigue-aware reward shaping to penalize repetitive recommendations.

Our central question is whether sequential planning becomes more valuable once the objective reflects user fatigue rather than only immediate preference.

---

## Research Question

> Does reinforcement learning provide meaningful value beyond a myopic greedy recommender when recommendations are evaluated in a learned offline environment, especially once repetitive recommendations are penalized?

---

## System Design

This project combines two components:

1. **User Simulator**  
   A learned reward model from the companion CS6140 project, which estimates `P(like | user, movie)` from historical MovieLens ratings.

2. **Offline RL Environment**  
   A recommendation environment where the RL agent selects a **genre** rather than a specific movie, and the environment chooses the highest-probability movie within that genre.

### MDP Formulation

| Component | Definition |
|---|---|
| **State** | 68-dim: SVD user embedding (50) + recent genre counts (18) |
| **Action** | 18 genre indices |
| **Reward (no penalty)** | Bernoulli(`P(like)`) from the simulator |
| **Reward (fatigue-aware)** | Bernoulli(clip(`P(like) − penalty`, 0.1, 0.9)) |
| **Episode length** | 20 recommendation steps |
| **History window** | 10 recent genre selections |

---

## Why This Design

### Why use **genre** as the action instead of movie?
- The movie space is too large for stable Deep Q-Network training.
- Genre better captures the diversity and repetition signal relevant to fatigue.
- Movie choice is delegated to the environment.

### Why include `recent_genre_counts` in the state?
- Fatigue depends on repetition history, so the policy needs direct access to recent genre usage.
- A compressed history embedding was too indirect for Deep Q-Network to reliably learn fatigue avoidance.
- Genre-count features provide a simple, interpretable signal for sequential decision-making.

### Why use a user simulator at all?
- Offline reinforcement learning requires many interactions during training.
- Historical logs cannot answer counterfactual questions such as “what if a different genre had been recommended?”
- The simulator makes controlled offline experimentation possible.

---

## Experiments

We evaluate two stages:

### 1. No-Penalty Evaluation
We first test whether policy ordering is stable when rewards are sampled directly from the learned simulator without any fatigue penalty.

### 2. Fatigue-Aware Reward Shaping
We then introduce:
- **Window-based penalty** for repeated recent genres
- **Concentration-based penalty** for overly narrow genre distributions

This allows us to test whether reinforcement learning becomes more useful when repetition is explicitly discouraged.

---

## Key Results

### No-Penalty Setting
The policy ranking is consistent across simulators:

**PPO > DQN > Greedy-CTR > Random**

This suggests that reinforcement learning can exploit sequence-conditional structure in the simulator even before fatigue penalties are added.

### Fatigue-Aware Setting

| Policy | Mean Reward (T=20) | Δ vs Greedy-CTR |
|---|---|---|
| Random | 8.17 | — |
| Greedy-CTR | 5.82 | baseline |
| DQN | 9.90 | +1.55* |
| **PPO** | **13.49** | **+4.31*** |

\* Both improvements over Greedy-CTR are statistically significant.

**Main result:** Under fatigue-aware reward, PPO outperformed Greedy-CTR by **+4.31** mean reward, providing the clearest evidence that sequential planning matters once repetitive recommendations are penalized.

### State Ablation

Removing recent genre-count features severely reduced performance:

| Agent | Full State | No Genre Counts | Drop |
|---|---|---|---|
| DQN | 9.90 | 6.22 | −3.68 |
| PPO | 10.40 | 5.55 | −3.28 |

This confirms that **state design must match reward design**: if the objective depends on history, the agent must observe that history directly.

---

## Main Conclusions

1. **Reinforcement learning can beat a greedy baseline even without explicit fatigue penalties**, though some gains may reflect simulator-specific dynamics.

2. **Reward design is first-order.** Once repetition is penalized, the gap between PPO and Greedy-CTR grows substantially.

3. **PPO outperforms DQN in this short-horizon, policy-sensitive setting.** Its stochastic policy better supports balanced sequential behavior, while Deep Q-Network remains more vulnerable to local exploitation.

4. **State representation matters as much as algorithm choice.** Genre-count features are essential once fatigue becomes part of the objective.

5. **Simulator quality remains the core bottleneck.** Stronger performance in simulation does not automatically guarantee better real-world recommendation quality.

---

## Limitations

- The environment is based on a learned simulator rather than real interactive feedback.
- Offline reinforcement learning can exploit simulator imperfections.
- MovieLens 1M is not a naturally fatigue-rich dataset, so fatigue effects are partly introduced through reward shaping rather than observed directly.
- Some ablation values vary slightly across runs, reflecting the instability typical of offline RL experiments.

---

## Why This Project Matters

This project demonstrates more than just reinforcement learning algorithms. It shows how to:

- formulate a sequential decision problem from historical recommendation data,
- connect a learned predictive model to a reinforcement learning environment,
- design reward functions that better reflect long-term objectives,
- and test whether policy learning meaningfully outperforms myopic optimization.

More broadly, it highlights a practical lesson for recommender systems:  
**better sequential behavior may depend as much on reward and state design as on the specific RL algorithm.**
