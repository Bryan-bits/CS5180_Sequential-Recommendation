"""
baselines.py — CS5180 RL Recommendation: Baselines & Helpers

Contains:
    - EpisodeRewardCallback (SB3 training callback)
    - Random baseline runner
    - Greedy-CTR baseline (genre selection + runner)
    - Generic expected-reward episode runner

Usage:
    from baselines import EpisodeRewardCallback, run_random, run_greedy_ctr
    from baselines import run_expected_episode
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# ============================================================
# 1. SB3 Training Callback
# ============================================================

class EpisodeRewardCallback(BaseCallback):
    """Track per-episode cumulative reward during SB3 training."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._cur = 0.0

    def _on_step(self):
        self._cur += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._cur)
            self._cur = 0.0
        return True


# ============================================================
# 2. Random Baseline
# ============================================================

def run_random(env, n_eval=200, eval_users=None):
    """Random genre + random movie."""
    rewards = []
    users = eval_users if eval_users is not None else [None] * n_eval

    for user in users:
        opts = {"user_id": user} if user else None
        obs, _ = env.reset(options=opts)
        ep, done = 0.0, False
        while not done:
            obs, r, done, _, _ = env.step_random(env.action_space.sample())
            ep += r
        rewards.append(ep)

    return rewards


# ============================================================
# 3. Greedy-CTR Baseline
# ============================================================

def greedy_ctr_action(env):
    """
    Greedy-CTR genre selection:
    Pick the genre with highest historical mean rating for this user.
    No simulator access — uses only user's past ratings.
    """
    user_ratings = (
        env.ratings_df[env.ratings_df["userId"] == env.current_user]
        .set_index("movieId")["rating"]
        .to_dict()
    )
    user_global_mean = (np.mean(list(user_ratings.values()))
                        if user_ratings else 3.0)

    best_action, best_score = 0, -1.0

    for i, genre in enumerate(env.all_genres):
        genre_movies = set(env.genre_to_movies.get(genre, []))
        genre_scores = [user_ratings[m] for m in genre_movies
                        if m in user_ratings]
        score = (np.mean(genre_scores) if genre_scores
                 else user_global_mean)

        if score > best_score:
            best_score = score
            best_action = i

    return best_action


def run_greedy_ctr(env, n_eval=200, eval_users=None):
    """Greedy-CTR: historical genre preference + history-based movie selection."""
    rewards = []
    users = eval_users if eval_users is not None else [None] * n_eval

    for user in users:
        opts = {"user_id": user} if user else None
        obs, _ = env.reset(options=opts)
        ep, done = 0.0, False
        while not done:
            obs, r, done, _, _ = env.step_ctr(greedy_ctr_action(env))
            ep += r
        rewards.append(ep)

    return rewards


# ============================================================
# 4. Model Evaluation (Bernoulli reward)
# ============================================================

def run_model_eval(env, model, n_eval=200, eval_users=None):
    """
    Evaluate a trained SB3 model (DQN or PPO) using Bernoulli-sampled reward.

    Args:
        env:        the RL environment
        model:      trained SB3 model with .predict()
        n_eval:     number of episodes
        eval_users: list of user IDs (None = random)

    Returns:
        list of episode rewards
    """
    rewards = []
    users = eval_users if eval_users is not None else [None] * n_eval

    for user in users:
        opts = {"user_id": user} if user else None
        obs, _ = env.reset(options=opts)
        ep, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, _ = env.step(action)
            ep += r
        rewards.append(ep)

    return rewards


# ============================================================
# 5. Expected Reward Evaluation (no Bernoulli noise)
# ============================================================

def run_expected_episode(env, policy_fn, movie_select_fn):
    """
    Run one episode accumulating expected reward (p_like) instead of
    sampled Bernoulli. Avoids duplicating _apply_step logic for each policy.

    Args:
        env:             the RL environment
        policy_fn:       callable(env, obs) → action (int)
        movie_select_fn: callable(env, genre) → movie_id

    Returns:
        float total expected reward for the episode
    """
    obs, _ = env.reset()
    ep_reward = 0.0

    while env.step_count < env.T:
        action = policy_fn(env, obs)
        genre = env.all_genres[int(action)]
        movie_id = movie_select_fn(env, genre)

        p_like = env._dynamic_p(movie_id)
        ep_reward += p_like

        # Manually advance env state (no Bernoulli sampling)
        env.all_recommended.append(movie_id)
        env.recent_genres.append(genre)
        if len(env.recent_genres) > env.window:
            env.recent_genres.pop(0)
        env.step_count += 1
        obs = env._get_state()

    return ep_reward


def run_expected_eval(env, policy_fn, movie_select_fn, n_eval=200):
    """
    Run n expected-reward episodes.

    Args:
        env:             the RL environment
        policy_fn:       callable(env, obs) → action
        movie_select_fn: callable(env, genre) → movie_id
        n_eval:          number of episodes

    Returns:
        list of expected rewards
    """
    return [run_expected_episode(env, policy_fn, movie_select_fn)
            for _ in range(n_eval)]


# ============================================================
# 6. Policy functions (for use with run_expected_eval)
# ============================================================

def make_model_policy(model):
    """Wrap an SB3 model into a policy_fn for run_expected_eval."""
    def policy_fn(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return policy_fn


def random_policy(env, obs):
    """Random genre selection."""
    return env.action_space.sample()


def greedy_policy(env, obs):
    """Greedy-CTR genre selection."""
    return greedy_ctr_action(env)


def history_movie_select(env, genre):
    """History-based movie selection (used by DQN, Greedy-CTR)."""
    return env._select_movie_history(genre)


def random_movie_select(env, genre):
    """Random movie selection (used by Random baseline)."""
    return env._select_movie_random(genre)
