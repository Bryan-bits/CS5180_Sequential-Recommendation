"""
data_setup.py — CS5180 RL Recommendation: Data Pipeline

Consolidates: dataset download, DataFrame loading, ID/genre mappings,
train/test split, SVD embeddings, XGBoost simulator training, and
helper functions.

Usage:
    from data_setup import load_all
    ctx = load_all(data_dir="ml-1m")
    # ctx is a dict with everything the RL notebook needs
"""

import os
import urllib.request
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# ============================================================
# 1. Download & Extract MovieLens 1M (skip if already exists)
# ============================================================

def download_movielens(data_dir: str = "ml-1m") -> None:
    """Download and unzip MovieLens 1M if the folder doesn't exist."""
    if os.path.isdir(data_dir):
        print(f"[download] '{data_dir}/' already exists — skipping.")
        return

    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = "ml-1m.zip"

    print(f"[download] Fetching {url} ...")
    urllib.request.urlretrieve(url, zip_path)

    print(f"[download] Extracting to '{data_dir}/' ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(".")

    os.remove(zip_path)
    print(f"[download] Done. Files: {os.listdir(data_dir)}")


# ============================================================
# 2. Load DataFrames
# ============================================================

def load_dataframes(data_dir: str = "ml-1m"):
    """Load ratings, movies, users from .dat files."""
    ratings = pd.read_csv(
        os.path.join(data_dir, "ratings.dat"),
        sep="::", engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        os.path.join(data_dir, "movies.dat"),
        sep="::", engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1",
    )
    users = pd.read_csv(
        os.path.join(data_dir, "users.dat"),
        sep="::", engine="python",
        names=["userId", "gender", "age", "occupation", "zip"],
    )

    # Binary label used by XGBoost simulator
    ratings["like"] = (ratings["rating"] >= 4).astype(int)

    print(f"[load] ratings: {len(ratings):,}  "
          f"movies: {len(movies):,}  users: {len(users):,}")
    return ratings, movies, users


# ============================================================
# 3. ID Mappings & Genre Lookups
# ============================================================

def build_mappings(ratings, movies):
    """Build user/movie/genre index dicts and genre-to-movie lists."""
    user_ids = sorted(ratings["userId"].unique())
    movie_ids = sorted(ratings["movieId"].unique())

    user2idx = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}

    # Extract genres from the dataset
    all_genres = sorted({
        g for genres in movies["genres"] for g in genres.split("|")
    })
    genre2idx = {g: i for i, g in enumerate(all_genres)}

    # genre_to_movies: only movies that appear in ratings (have embeddings)
    genre_to_movies = defaultdict(list)
    movie_genres = {}

    for _, row in movies.iterrows():
        mid = int(row["movieId"])
        if mid not in movie2idx:
            continue
        genres = [g for g in row["genres"].split("|") if g in genre2idx]
        movie_genres[mid] = genres
        for g in genres:
            genre_to_movies[g].append(mid)

    print(f"[mappings] users: {len(user2idx):,}  "
          f"movies: {len(movie2idx):,}  genres: {len(all_genres)}")
    return user2idx, movie2idx, genre2idx, genre_to_movies, movie_genres


# ============================================================
# 4. Train / Test Split (chronological 80/20)
# ============================================================

def split_train_test(ratings, user2idx, movie2idx):
    """Chronological 80/20 split → sparse rating matrix."""
    sorted_df = ratings.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(sorted_df) * 0.8)
    train_df = sorted_df.iloc[:split_idx]
    test_df = sorted_df.iloc[split_idx:]

    rows = [user2idx[u] for u in train_df["userId"]]
    cols = [movie2idx[m] for m in train_df["movieId"]]
    vals = train_df["rating"].values.astype(np.float32)

    R_train = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(user2idx), len(movie2idx)),
    )

    print(f"[split] train: {len(train_df):,}  test: {len(test_df):,}  "
          f"R_train nnz: {R_train.nnz:,}")
    return train_df, test_df, R_train


# ============================================================
# 5. SVD Embeddings
# ============================================================

def compute_svd(R_train, k: int = 50):
    """SVD on the training rating matrix → user & movie embeddings."""
    print(f"[svd] Computing SVD (k={k}) ...")
    U_raw, sigma, Vt_raw = svds(R_train.astype(np.float64), k=k)
    sqrt_sigma = np.diag(np.sqrt(sigma))

    U_embed = (U_raw @ sqrt_sigma).astype(np.float32)      # (n_users,  k)
    M_embed = (Vt_raw.T @ sqrt_sigma).astype(np.float32)   # (n_movies, k)

    print(f"[svd] U_embed: {U_embed.shape}  M_embed: {M_embed.shape}")
    return U_embed, M_embed


# ============================================================
# 6. XGBoost User Simulator
# ============================================================

def build_feature_matrix(df, user2idx, movie2idx, U_embed, M_embed):
    """Concatenate user + movie SVD embeddings → feature matrix."""
    u_idx = [user2idx[u] for u in df["userId"]]
    m_idx = [movie2idx[m] for m in df["movieId"]]
    X = np.hstack([U_embed[u_idx], M_embed[m_idx]])
    y = (df["rating"].values >= 4).astype(int)
    return X, y


def train_xgb_simulator(X_train, y_train, X_test, y_test):
    """Train XGBoost P(like | user, movie) simulator."""
    print("[xgb] Training XGBoost simulator ...")
    model = XGBClassifier(
        n_estimators=120, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"[xgb] Accuracy: {acc:.3f}  AUC-ROC: {auc:.3f}")
    return model


# ============================================================
# 7. Helper Functions (used by the RL environment)
# ============================================================

def make_helpers(user2idx, movie2idx, movie_genres, U_embed, M_embed, xgb_model):
    """
    Returns a dict of helper functions that close over the data.
    Keeps the RL notebook free of global state.
    """

    def get_p_like(userId, movieId):
        """P(like | user, movie) from XGBoost."""
        if userId not in user2idx or movieId not in movie2idx:
            return 0.5
        feat = np.hstack([
            U_embed[user2idx[userId]],
            M_embed[movie2idx[movieId]],
        ]).reshape(1, -1)
        return float(xgb_model.predict_proba(feat)[0][1])

    def get_movie_genres(movieId):
        return movie_genres.get(movieId, [])

    def count_same_genres(recent_movie_ids, target_genres):
        target_set = set(target_genres)
        return sum(
            1 for mid in recent_movie_ids
            if target_set & set(movie_genres.get(mid, []))
        )

    return {
        "get_p_like": get_p_like,
        "get_movie_genres": get_movie_genres,
        "count_same_genres": count_same_genres,
    }


# ============================================================
# 8. Main Entry Point
# ============================================================

def load_all(data_dir: str = "ml-1m", svd_k: int = 50, save_dir: str = "models"):
    """
    Run the full data pipeline and return everything the RL notebook needs.

    Returns a dict with keys:
        ratings, movies, users,
        train_df, test_df,
        user2idx, movie2idx, genre2idx,
        genre_to_movies, movie_genres,
        U_embed, M_embed,
        xgb_model,
        helpers  (dict of get_p_like, get_movie_genres, count_same_genres)
    """
    # Download if needed
    download_movielens(data_dir)

    # Load
    ratings, movies, users = load_dataframes(data_dir)

    # Mappings
    user2idx, movie2idx, genre2idx, genre_to_movies, movie_genres = \
        build_mappings(ratings, movies)

    # Split
    train_df, test_df, R_train = split_train_test(ratings, user2idx, movie2idx)

    # SVD
    U_embed, M_embed = compute_svd(R_train, k=svd_k)

    # XGBoost
    X_train, y_train = build_feature_matrix(
        train_df, user2idx, movie2idx, U_embed, M_embed)
    X_test, y_test = build_feature_matrix(
        test_df, user2idx, movie2idx, U_embed, M_embed)
    xgb_model = train_xgb_simulator(X_train, y_train, X_test, y_test)

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    xgb_path = os.path.join(save_dir, "xgb_simulator.json")
    xgb_model.save_model(xgb_path)
    print(f"[save] XGBoost → {xgb_path}")

    # Helpers
    helpers = make_helpers(
        user2idx, movie2idx, movie_genres, U_embed, M_embed, xgb_model)

    print("\n[load_all] Pipeline complete.")

    return {
        "ratings": ratings, "movies": movies, "users": users,
        "train_df": train_df, "test_df": test_df,
        "user2idx": user2idx, "movie2idx": movie2idx, "genre2idx": genre2idx,
        "genre_to_movies": genre_to_movies, "movie_genres": movie_genres,
        "U_embed": U_embed, "M_embed": M_embed,
        "xgb_model": xgb_model,
        "helpers": helpers,
    }


if __name__ == "__main__":
    ctx = load_all()
    print(f"\nReturned keys: {list(ctx.keys())}")