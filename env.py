"""
env.py — CS5180 RL Recommendation Environment

Gymnasium environment for fatigue-aware sequential movie recommendation.

State  (68-dim): user_embed(50) | recent_genre_counts(18)
Action (18):     genre index
Reward:          Bernoulli( clip(p_like, 0.1, 0.9) )
T = 20 steps per episode

Usage:
    from env import FinalMovieRecEnvV6, EnvNoGenreCounts
    env = FinalMovieRecEnvV6(ctx, gru4rec_p_like=gru4rec_p_like)
"""

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FinalMovieRecEnvV6(gym.Env):
    """
    Movie recommendation environment with genre-level actions.

    Movie selection methods by policy:
        _select_movie_random  → Random baseline   (no info)
        _select_movie_history → DQN + Greedy-CTR  (user history only)
        _select_movie_oracle  → Oracle             (simulator P(like))

    Args:
        ctx:             dict from data_setup.load_all()
        gru4rec_p_like:  closure from make_gru4rec_p_like() — sequence-aware reward
        T:               episode length
        window:          sliding window for recent genre counts
    """

    def __init__(self, ctx, gru4rec_p_like, T=20, window=10):
        super().__init__()

        # Unpack from ctx
        self.user_ids       = sorted(ctx["user2idx"].keys())
        self.movie_ids      = sorted(ctx["movie2idx"].keys())
        self.user2idx       = ctx["user2idx"]
        self.movie2idx      = ctx["movie2idx"]
        self.U_embed        = ctx["U_embed"]
        self.M_embed        = ctx["M_embed"]
        self.ratings_df     = ctx["ratings"]
        self.genre_to_movies = ctx["genre_to_movies"]
        self.movie_genres   = ctx["movie_genres"]
        self.all_genres     = sorted(ctx["genre2idx"].keys())
        self.genre2idx      = ctx["genre2idx"]
        self.n_genres       = len(self.all_genres)

        # Helpers
        self.get_p_like     = ctx["helpers"]["get_p_like"]
        self.gru4rec_p_like = gru4rec_p_like

        # Env config
        self.T              = T
        self.window         = window
        self.embed_dim      = self.U_embed.shape[1]       # 50
        self.state_dim      = self.embed_dim + self.n_genres  # 68

        self.action_space      = spaces.Discrete(self.n_genres)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32)

        # Episode state
        self.current_user    = None
        self.all_recommended = []
        self.recent_genres   = []
        self.step_count      = 0

    # ── state ────────────────────────────────────────────────

    def _get_state(self):
        u_idx = self.user2idx[self.current_user]
        user_emb = self.U_embed[u_idx].astype(np.float32)
        genre_counts = np.zeros(self.n_genres, dtype=np.float32)
        for g in self.recent_genres:
            if g in self.genre2idx:
                genre_counts[self.genre2idx[g]] += 1.0
        if self.window > 0:
            genre_counts /= self.window
        return np.concatenate([user_emb, genre_counts])

    # ── movie selection (three versions) ─────────────────────

    def _get_pool(self, genre):
        """Unseen movies in genre."""
        seen = set(self.all_recommended)
        pool = [m for m in self.genre_to_movies.get(genre, [])
                if m not in seen and m in self.movie2idx]
        if not pool:
            pool = [m for m in self.movie_ids if m not in seen]
        return pool if pool else [self.movie_ids[0]]

    def _select_movie_random(self, genre):
        """Random: no information used."""
        return random.choice(self._get_pool(genre))

    def _select_movie_history(self, genre):
        """
        History-based: uses user's own past ratings.
        No simulator access — same info as Greedy-CTR.
        """
        pool = self._get_pool(genre)
        sample = random.sample(pool, min(20, len(pool)))

        user_ratings = (
            self.ratings_df[self.ratings_df["userId"] == self.current_user]
            .set_index("movieId")["rating"]
            .to_dict()
        )
        user_global_mean = (np.mean(list(user_ratings.values()))
                            if user_ratings else 3.0)

        return max(sample,
                   key=lambda m: user_ratings.get(m, user_global_mean))

    def _select_movie_oracle(self, genre):
        """Oracle: uses simulator P(like). Not realistic."""
        pool = self._get_pool(genre)
        sample = random.sample(pool, min(20, len(pool)))
        return max(sample,
                   key=lambda m: self.get_p_like(self.current_user, m))

    # ── reward ────────────────────────────────────────────────

    def _dynamic_p(self, movieId):
        base_p = self.gru4rec_p_like(
            self.current_user, movieId, self.all_recommended)

        # Window penalty (recent repetition)
        target_genres = set(self.movie_genres.get(movieId, []))
        recent = self.all_recommended[-self.window:]
        same_count = sum(1 for mid in recent
                        if target_genres & set(self.movie_genres.get(mid, [])))
        window_penalty = min(0.05 * same_count, 0.25)

        # Concentration penalty (session-level)
        if self.all_recommended:
            genre_picks = []
            for mid in self.all_recommended:
                genre_picks.extend(self.movie_genres.get(mid, []))
            total = len(genre_picks) if genre_picks else 1
            genre_frac = sum(1 for g in genre_picks if g in target_genres) / total
            concentration_penalty = 0.3 * max(genre_frac - 0.25, 0)  # kick in above 25%
        else:
            concentration_penalty = 0

        penalty = window_penalty + concentration_penalty
        return float(np.clip(base_p - penalty, 0.1, 0.9))

    # ── shared step logic ─────────────────────────────────────

    def _apply_step(self, genre, movie_id):
        p_like = self._dynamic_p(movie_id)
        reward = float(np.random.binomial(1, p_like))

        self.all_recommended.append(movie_id)
        self.recent_genres.append(genre)
        if len(self.recent_genres) > self.window:
            self.recent_genres.pop(0)

        self.step_count += 1
        done = self.step_count >= self.T

        info = dict(user=self.current_user, genre=genre,
                    movie=movie_id, p_like=p_like, reward=reward)
        return self._get_state(), reward, done, False, info

    # ── gymnasium API ─────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_user = (
            options["user_id"]
            if options and "user_id" in options
            else random.choice(self.user_ids)
        )
        self.all_recommended = []
        self.recent_genres = []
        self.step_count = 0
        return self._get_state(), {"user": self.current_user}

    def step(self, action):
        """DQN/PPO step — history-based movie selection."""
        genre = self.all_genres[int(action)]
        return self._apply_step(genre, self._select_movie_history(genre))

    def step_random(self, action):
        """Random baseline step."""
        genre = self.all_genres[int(action)]
        return self._apply_step(genre, self._select_movie_random(genre))

    def step_ctr(self, action):
        """Greedy-CTR step — history-based movie selection."""
        genre = self.all_genres[int(action)]
        return self._apply_step(genre, self._select_movie_history(genre))


# ============================================================
# Ablation: state without genre counts
# ============================================================

class EnvNoGenreCounts(FinalMovieRecEnvV6):
    """
    Ablation variant: state = user_embed only (50-dim).
    Removes genre_counts to test whether DQN/PPO actually
    use the fatigue signal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_dim = self.embed_dim  # 50
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32)

    def _get_state(self):
        u_idx = self.user2idx[self.current_user]
        return self.U_embed[u_idx].astype(np.float32)
