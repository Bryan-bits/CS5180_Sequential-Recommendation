"""
eda.py — CS5180 RL Recommendation: Exploratory Data Analysis

Callable functions for plotting and saving derived features.

Usage:
    from data_setup import load_all
    from eda import run_all_eda  # or call individual functions

    ctx = load_all()
    run_all_eda(ctx)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

OUT_DIR = "eda"


def _ensure_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def _savefig(name):
    _ensure_dir()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")


def _explode_genres(ratings, movies):
    """Merge ratings with genres and explode to one row per genre."""
    merged = ratings.merge(movies[["movieId", "genres"]], on="movieId")
    return merged.assign(genre=merged["genres"].str.split("|")).explode("genre")


# ============================================================
# Individual plot + save functions
# ============================================================

def plot_rating_distribution(ctx):
    ratings = ctx["ratings"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    counts = ratings["rating"].value_counts().sort_index()
    axes[0].bar(counts.index, counts.values, color="steelblue", width=0.6)
    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Rating Distribution")
    for r, c in counts.items():
        axes[0].text(r, c + 5000, f"{c / len(ratings):.1%}", ha="center", fontsize=8)

    like_counts = ratings["like"].value_counts().sort_index()
    axes[1].bar(["Dislike (1-3)", "Like (4-5)"], like_counts.values,
                color=["#e74c3c", "#2ecc71"], width=0.5)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Like vs Dislike (threshold = 4)")
    for i, c in enumerate(like_counts.values):
        axes[1].text(i, c + 5000, f"{c:,}", ha="center", fontsize=9)

    plt.tight_layout()
    _savefig("01_rating_distribution.png")
    plt.show()


def plot_user_activity(ctx):
    ratings = ctx["ratings"]
    rpu = ratings.groupby("userId").size()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(rpu, bins=80, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Number of Ratings")
    axes[0].set_ylabel("Number of Users")
    axes[0].set_title("Ratings per User")
    axes[0].axvline(rpu.median(), color="red", ls="--",
                    label=f"Median={rpu.median():.0f}")
    axes[0].legend()

    axes[1].hist(rpu, bins=80, color="steelblue", edgecolor="white", log=True)
    axes[1].set_xlabel("Number of Ratings")
    axes[1].set_ylabel("Number of Users (log)")
    axes[1].set_title("Ratings per User (log scale)")

    plt.tight_layout()
    _savefig("02_user_activity.png")
    plt.show()

    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p:2d}: {np.percentile(rpu, p):.0f}")


def plot_movie_popularity(ctx):
    ratings = ctx["ratings"]
    movies = ctx["movies"]
    rpm = ratings.groupby("movieId").size().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(rpm, bins=80, color="coral", edgecolor="white")
    axes[0].set_xlabel("Number of Ratings")
    axes[0].set_ylabel("Number of Movies")
    axes[0].set_title("Ratings per Movie")

    axes[1].plot(range(len(rpm)), rpm.values, color="coral", lw=0.8)
    axes[1].set_xlabel("Movie Rank")
    axes[1].set_ylabel("Number of Ratings")
    axes[1].set_title("Movie Popularity Long Tail")
    axes[1].set_yscale("log")

    plt.tight_layout()
    _savefig("03_movie_popularity.png")
    plt.show()

    print(f"Movies with ≤10 ratings : {(rpm <= 10).sum()}")
    print(f"Movies with ≤50 ratings : {(rpm <= 50).sum()}")
    print("\nTop 10 most rated:")
    for mid, count in rpm.head(10).items():
        title = movies.loc[movies["movieId"] == mid, "title"].values[0]
        print(f"  {title:<45s} {count:,}")


def plot_genre_analysis(ctx):
    """
    Genre-level stats plot.
    Saves: eda/genre_stats.csv
    Returns: genre_stats DataFrame
    """
    ratings = ctx["ratings"]
    movies = ctx["movies"]
    exploded = _explode_genres(ratings, movies)

    genre_stats = exploded.groupby("genre").agg(
        n_ratings=("rating", "size"),
        n_movies=("movieId", "nunique"),
        avg_rating=("rating", "mean"),
        like_rate=("like", "mean"),
    ).sort_values("n_ratings", ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].barh(genre_stats.index[::-1], genre_stats["n_ratings"][::-1],
                     color="steelblue")
    axes[0, 0].set_xlabel("Number of Ratings")
    axes[0, 0].set_title("Ratings per Genre")

    axes[0, 1].barh(genre_stats.index[::-1], genre_stats["n_movies"][::-1],
                     color="coral")
    axes[0, 1].set_xlabel("Number of Movies")
    axes[0, 1].set_title("Movies per Genre")

    sorted_by_like = genre_stats.sort_values("like_rate")
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(sorted_by_like)))
    axes[1, 0].barh(sorted_by_like.index, sorted_by_like["like_rate"], color=colors)
    axes[1, 0].set_xlabel("Like Rate (rating ≥ 4)")
    axes[1, 0].set_title("Like Rate per Genre ← RL reward signal")
    axes[1, 0].axvline(ratings["like"].mean(), color="black", ls="--", lw=0.8,
                        label="Overall")
    axes[1, 0].legend()

    sorted_by_rating = genre_stats.sort_values("avg_rating")
    axes[1, 1].barh(sorted_by_rating.index, sorted_by_rating["avg_rating"],
                     color="mediumpurple")
    axes[1, 1].set_xlabel("Average Rating")
    axes[1, 1].set_title("Average Rating per Genre")
    axes[1, 1].set_xlim(2.5, 4.5)

    plt.suptitle("Genre Analysis — 18 actions in the MDP", fontsize=13, y=1.02)
    plt.tight_layout()
    _savefig("04_genre_analysis.png")
    plt.show()

    # Save
    _ensure_dir()
    genre_stats.to_csv(os.path.join(OUT_DIR, "genre_stats.csv"))
    print(f"Saved → {OUT_DIR}/genre_stats.csv")

    print(genre_stats.to_string(float_format="%.3f"))
    print(f"\nLike-rate range: {genre_stats['like_rate'].min():.3f} – "
          f"{genre_stats['like_rate'].max():.3f}")
    print("→ This spread drives DQN genre concentration.")
    return genre_stats


def plot_genre_cooccurrence(ctx):
    """
    Genre co-occurrence heatmap.
    Saves: eda/genre_cooccurrence.npy, eda/genre_order.csv
    Returns: (cooccur_norm, all_genres_sorted)
    """
    genre2idx = ctx["genre2idx"]
    movie_genres = ctx["movie_genres"]
    all_genres_sorted = sorted(genre2idx.keys())
    n = len(all_genres_sorted)

    cooccur = np.zeros((n, n), dtype=int)
    for mid, genres in movie_genres.items():
        idxs = [all_genres_sorted.index(g) for g in genres
                if g in all_genres_sorted]
        for i in idxs:
            for j in idxs:
                cooccur[i, j] += 1

    row_sums = cooccur.diagonal()[:, None]
    cooccur_norm = np.divide(cooccur, row_sums, where=row_sums > 0,
                             out=np.zeros_like(cooccur, dtype=float))
    np.fill_diagonal(cooccur_norm, 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cooccur_norm, xticklabels=all_genres_sorted,
                yticklabels=all_genres_sorted, cmap="YlOrRd",
                annot=True, fmt=".2f", linewidths=0.3, ax=ax,
                annot_kws={"fontsize": 6})
    ax.set_title("Genre Co-occurrence: P(column | row)")
    _savefig("05_genre_cooccurrence.png")
    plt.show()

    # Save
    _ensure_dir()
    np.save(os.path.join(OUT_DIR, "genre_cooccurrence.npy"), cooccur_norm)
    pd.Series(all_genres_sorted, name="genre").to_csv(
        os.path.join(OUT_DIR, "genre_order.csv"), index=False)
    print(f"Saved → {OUT_DIR}/genre_cooccurrence.npy")
    print(f"Saved → {OUT_DIR}/genre_order.csv")

    avg_g = np.mean([len(g) for g in movie_genres.values()])
    print(f"Average genres per movie: {avg_g:.2f}")
    return cooccur_norm, all_genres_sorted


def plot_temporal_analysis(ctx):
    ratings = ctx["ratings"].copy()
    movies = ctx["movies"]
    train_df = ctx["train_df"]
    test_df = ctx["test_df"]

    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["year_month"] = ratings["datetime"].dt.to_period("M")
    split_ts = ratings.sort_values("timestamp").iloc[
        int(len(ratings) * 0.8)]["datetime"]

    exploded = _explode_genres(ratings, movies)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    monthly = ratings.groupby("year_month").size()
    axes[0, 0].plot(monthly.index.astype(str), monthly.values,
                     color="steelblue", lw=1)
    axes[0, 0].set_title("Rating Volume Over Time")
    axes[0, 0].tick_params(axis="x", rotation=45, labelsize=7)
    axes[0, 0].axvline(str(split_ts.to_period("M")), color="red", ls="--",
                        label="Train/Test split")
    axes[0, 0].legend(fontsize=8)

    monthly_like = ratings.groupby("year_month")["like"].mean()
    axes[0, 1].plot(monthly_like.index.astype(str), monthly_like.values,
                     color="green", lw=1)
    axes[0, 1].set_title("Like Rate Over Time")
    axes[0, 1].tick_params(axis="x", rotation=45, labelsize=7)
    axes[0, 1].axvline(str(split_ts.to_period("M")), color="red", ls="--",
                        label="Train/Test split")
    axes[0, 1].axhline(ratings["like"].mean(), color="gray", ls=":", lw=0.8)
    axes[0, 1].legend(fontsize=8)

    train_g = exploded.loc[exploded.index.isin(train_df.index)]
    test_g = exploded.loc[exploded.index.isin(test_df.index)]
    train_dist = train_g["genre"].value_counts(normalize=True).sort_index()
    test_dist = test_g["genre"].value_counts(normalize=True).sort_index()

    x = np.arange(len(train_dist))
    w = 0.35
    axes[1, 0].bar(x - w/2, train_dist.values, w, label="Train",
                    color="steelblue")
    axes[1, 0].bar(x + w/2, test_dist.reindex(train_dist.index).values, w,
                    label="Test", color="coral")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(train_dist.index, rotation=45, ha="right",
                                fontsize=7)
    axes[1, 0].set_title("Genre Distribution: Train vs Test")
    axes[1, 0].legend(fontsize=8)

    train_like = train_g.groupby("genre")["like"].mean().sort_index()
    test_like = test_g.groupby("genre")["like"].mean().sort_index()
    axes[1, 1].bar(x - w/2, train_like.values, w, label="Train",
                    color="steelblue")
    axes[1, 1].bar(x + w/2, test_like.reindex(train_like.index).values, w,
                    label="Test", color="coral")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(train_like.index, rotation=45, ha="right",
                                fontsize=7)
    axes[1, 1].set_title("Like Rate per Genre: Train vs Test")
    axes[1, 1].legend(fontsize=8)

    plt.suptitle("Temporal Analysis — Chronological Split Check",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    _savefig("06_temporal_analysis.png")
    plt.show()

    shift = (test_dist.reindex(train_dist.index) - train_dist).abs()
    print(f"Genre distribution shift (L1): {shift.sum():.4f}")
    print("Top shifted genres:")
    for g in shift.nlargest(5).index:
        print(f"  {g:<15s}: Δ={shift[g]:.4f}")


def plot_user_diversity(ctx):
    """
    User genre diversity (entropy + dominant fraction).
    Saves: eda/user_genre_diversity.csv
    Returns: user_diversity DataFrame
    """
    ratings = ctx["ratings"]
    movies = ctx["movies"]
    genre2idx = ctx["genre2idx"]
    n_genres = len(genre2idx)

    exploded = _explode_genres(ratings, movies)
    user_genre_counts = exploded.groupby(
        ["userId", "genre"]).size().unstack(fill_value=0)

    def entropy(row):
        p = row / row.sum()
        p = p[p > 0]
        return -(p * np.log2(p)).sum()

    user_entropy = user_genre_counts.apply(entropy, axis=1)
    dominant_frac = user_genre_counts.max(axis=1) / user_genre_counts.sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(user_entropy, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Genre Entropy (bits)")
    axes[0].set_ylabel("Number of Users")
    axes[0].set_title("User Genre Diversity (Shannon Entropy)")
    axes[0].axvline(user_entropy.median(), color="red", ls="--",
                    label=f"Median={user_entropy.median():.2f}")
    axes[0].legend()

    axes[1].hist(dominant_frac, bins=50, color="coral", edgecolor="white")
    axes[1].set_xlabel("Fraction of Most-Watched Genre")
    axes[1].set_ylabel("Number of Users")
    axes[1].set_title("How Concentrated Are Users?")

    plt.tight_layout()
    _savefig("07_user_genre_diversity.png")
    plt.show()

    # Save
    _ensure_dir()
    user_diversity = pd.DataFrame({
        "userId": user_entropy.index,
        "genre_entropy": user_entropy.values,
        "dominant_genre_frac": dominant_frac.values,
    })
    user_diversity.to_csv(os.path.join(OUT_DIR, "user_genre_diversity.csv"),
                          index=False)
    print(f"Saved → {OUT_DIR}/user_genre_diversity.csv")

    print(f"Entropy  — Mean: {user_entropy.mean():.2f}  "
          f"Median: {user_entropy.median():.2f}  "
          f"Max possible: {np.log2(n_genres):.2f}")
    print(f"Dom frac — Mean: {dominant_frac.mean():.2%}  "
          f"Median: {dominant_frac.median():.2%}")
    return user_diversity


def plot_svd_embeddings(ctx):
    U_embed = ctx["U_embed"]
    M_embed = ctx["M_embed"]
    ratings = ctx["ratings"]

    pca = PCA(n_components=2)
    U_pca = pca.fit_transform(U_embed)

    user_activity = ratings.groupby("userId").size().reindex(
        sorted(ratings["userId"].unique())
    ).values

    movie_avg_rating = ratings.groupby("movieId")["rating"].mean().reindex(
        sorted(ratings["movieId"].unique())
    ).values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc = axes[0].scatter(U_pca[:, 0], U_pca[:, 1], c=np.log1p(user_activity),
                         cmap="viridis", s=2, alpha=0.4)
    plt.colorbar(sc, ax=axes[0], label="log(1 + n_ratings)")
    axes[0].set_title("User Embeddings (PCA)")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    M_pca = PCA(n_components=2).fit_transform(M_embed)
    sc2 = axes[1].scatter(M_pca[:, 0], M_pca[:, 1], c=movie_avg_rating,
                          cmap="RdYlGn", s=2, alpha=0.4, vmin=1, vmax=5)
    plt.colorbar(sc2, ax=axes[1], label="Avg Rating")
    axes[1].set_title("Movie Embeddings (PCA)")

    plt.tight_layout()
    _savefig("08_svd_embeddings.png")
    plt.show()

    print(f"User PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")


def print_dataset_summary(ctx):
    ratings = ctx["ratings"]
    n_users = len(ctx["user2idx"])
    n_movies = len(ctx["movie2idx"])
    n_genres = len(ctx["genre2idx"])
    sparsity = 1 - len(ratings) / (n_users * n_movies)

    print(f"Ratings      : {len(ratings):,}")
    print(f"Users        : {n_users:,}")
    print(f"Movies       : {n_movies:,}")
    print(f"Genres       : {n_genres}")
    print(f"Sparsity     : {sparsity:.2%}")
    print(f"Avg rat/user : {len(ratings) / n_users:.1f}")
    print(f"Avg rat/movie: {len(ratings) / n_movies:.1f}")
    print(f"Like rate    : {ratings['like'].mean():.2%}")


def print_saved_summary():
    _ensure_dir()
    print("=" * 50)
    print(f"All EDA outputs in '{OUT_DIR}/':")
    print("=" * 50)
    plots = [f for f in sorted(os.listdir(OUT_DIR)) if f.endswith(".png")]
    data = [f for f in sorted(os.listdir(OUT_DIR)) if not f.endswith(".png")]
    if plots:
        print("\nPlots:")
        for f in plots:
            print(f"  {f}")
    if data:
        print("\nDerived features:")
        for f in data:
            print(f"  {f}")


# ============================================================
# Convenience: run everything
# ============================================================

def run_all_eda(ctx):
    """Run all EDA plots and save all derived features."""
    print_dataset_summary(ctx)
    plot_rating_distribution(ctx)
    plot_user_activity(ctx)
    plot_movie_popularity(ctx)
    plot_genre_analysis(ctx)
    plot_genre_cooccurrence(ctx)
    plot_temporal_analysis(ctx)
    plot_user_diversity(ctx)
    plot_svd_embeddings(ctx)
    print_saved_summary()


# ============================================================
# Load saved features (for downstream use)
# ============================================================

def load_genre_stats():
    return pd.read_csv(os.path.join(OUT_DIR, "genre_stats.csv"), index_col=0)

def load_genre_cooccurrence():
    matrix = np.load(os.path.join(OUT_DIR, "genre_cooccurrence.npy"))
    order = pd.read_csv(os.path.join(OUT_DIR, "genre_order.csv"))["genre"].tolist()
    return matrix, order

def load_user_diversity():
    return pd.read_csv(os.path.join(OUT_DIR, "user_genre_diversity.csv"))