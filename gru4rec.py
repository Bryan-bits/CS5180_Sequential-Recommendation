"""
gru4rec.py — GRU4Rec User Simulator

Sequence-aware simulator that predicts P(like | viewing_history, next_movie).
Trained as part of the CS6140 companion project, used as the reward model
for the CS5180 RL recommendation agent.

Usage:
    from gru4rec import load_gru4rec, make_gru4rec_p_like, train_gru4rec

    # Option A: load pretrained
    model = load_gru4rec("outputs/simulators/gru4rec_best.pt", n_movies, device)
    gru4rec_p_like = make_gru4rec_p_like(model, movie2idx, T=20)

    # Option B: train from scratch
    model = train_gru4rec(ctx, save_dir="outputs/simulators", n_epochs=30)
"""

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


# ============================================================
# 1. Model Definition
# ============================================================

class GRU4RecSimulator(nn.Module):
    """
    Predicts P(like | viewing_history, next_movie).

    Note: movie indices are stored as (movie2idx[m] + 1) because
    padding_idx=0 is reserved. All data prep must apply this +1 offset.
    """

    def __init__(self, n_movies, embed_dim=50, hidden_dim=128, T=20, dropout=0.3):
        super().__init__()
        self.T = T
        self.movie_embed = nn.Embedding(n_movies + 1, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim,
                          batch_first=True, dropout=0.0)  # single layer, no GRU dropout
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim + embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, sequence, next_movie):
        """
        Args:
            sequence:   (batch, T) int tensor, 0-padded, 1-indexed movie ids
            next_movie: (batch,)   int tensor, 1-indexed movie id
        Returns:
            (batch,) float tensor of P(like)
        """
        seq_embed = self.embed_drop(self.movie_embed(sequence))
        _, hidden = self.gru(seq_embed)
        hidden = hidden.squeeze(0)
        next_embed = self.movie_embed(next_movie)
        combined = torch.cat([hidden, next_embed], dim=1)
        return self.fc(combined).squeeze(1)


# ============================================================
# 2. Training
# ============================================================

def _build_sequence_dataset(ratings_df, movie2idx, T=20):
    """
    Sliding window over each user's chronological ratings.
    Returns list of {sequence, next_movie, label} dicts.
    Indices are +1 offset (0 = padding).
    """
    data = []
    sorted_df = ratings_df.sort_values(["userId", "timestamp"])

    for userId, group in sorted_df.groupby("userId"):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue

        recent = group.tail(50)

        for i in range(1, len(recent)):
            current = recent.iloc[i]
            if current["movieId"] not in movie2idx:
                continue

            start = max(0, i - T)
            history = recent.iloc[start:i]["movieId"].tolist()
            history_idx = [movie2idx[m] + 1 for m in history
                           if m in movie2idx]
            padded = [0] * (T - len(history_idx)) + history_idx

            data.append({
                "sequence": padded,
                "next_movie": movie2idx[current["movieId"]] + 1,
                "label": int(current["rating"] >= 4),
            })

    return data


class _SeqDataset(Dataset):
    def __init__(self, data):
        self.sequences = torch.tensor(
            [d["sequence"] for d in data], dtype=torch.long)
        self.next_movies = torch.tensor(
            [d["next_movie"] for d in data], dtype=torch.long)
        self.labels = torch.tensor(
            [d["label"] for d in data], dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.next_movies[idx], self.labels[idx]


def _evaluate_auc(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for seq, next_movie, label in loader:
            pred = model(seq.to(device), next_movie.to(device))
            preds.extend(pred.cpu().numpy())
            labels.extend(label.numpy())
    return roc_auc_score(labels, preds)


def train_gru4rec(ctx, save_dir="outputs/simulators", n_epochs=50, patience=8,
                  batch_size=512, lr=5e-4, weight_decay=1e-4,
                  embed_dim=50, hidden_dim=128, dropout=0.3, T=20):
    """
    Train GRU4Rec from scratch using ctx from load_all().

    Args:
        ctx:          dict from data_setup.load_all()
        save_dir:     directory to save best checkpoint
        n_epochs:     max training epochs
        patience:     early stopping patience
        weight_decay: L2 regularization

    Returns:
        trained GRU4RecSimulator in eval mode
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gru4rec] Device: {device}")

    movie2idx = ctx["movie2idx"]
    train_df = ctx["train_df"]
    test_df = ctx["test_df"]

    # Build datasets
    print("[gru4rec] Building sequence datasets ...")
    train_data = _build_sequence_dataset(train_df, movie2idx, T)
    test_data = _build_sequence_dataset(test_df, movie2idx, T)
    print(f"[gru4rec] Train sequences: {len(train_data):,}  "
          f"Test sequences: {len(test_data):,}")

    train_loader = DataLoader(_SeqDataset(train_data), batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(_SeqDataset(test_data), batch_size=batch_size,
                             shuffle=False, num_workers=0)

    # Init model
    model = GRU4RecSimulator(len(movie2idx), embed_dim, hidden_dim, T,
                             dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)
    criterion = nn.BCELoss()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "gru4rec_best.pt")

    best_auc = 0
    no_improve = 0

    print(f"[gru4rec] Training ({n_epochs} epochs, patience={patience}, "
          f"dropout={dropout}, wd={weight_decay}) ...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        for seq, next_movie, label in train_loader:
            seq = seq.to(device)
            next_movie = next_movie.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred = model(seq, next_movie)
            loss = criterion(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        auc = _evaluate_auc(model, test_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1:2d}/{n_epochs}  loss={avg_loss:.4f}  "
              f"AUC={auc:.4f}  lr={current_lr:.1e}")

        scheduler.step(auc)

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Reload best checkpoint
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    final_auc = _evaluate_auc(model, test_loader, device)
    print(f"[gru4rec] Training complete. Best AUC: {final_auc:.4f}")
    print(f"[gru4rec] Saved → {save_path}")
    return model


# ============================================================
# 3. Loading & Inference
# ============================================================

def load_gru4rec(path, n_movies, device=None, embed_dim=50, hidden_dim=128,
                 T=20, dropout=0.3):
    """Load a pretrained GRU4Rec model from disk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRU4RecSimulator(n_movies, embed_dim, hidden_dim, T,
                             dropout=dropout).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"[gru4rec] Loaded from {path} → {device}")
    return model


def make_gru4rec_p_like(model, movie2idx, T=20):
    """
    Create a closure that predicts P(like) for use during RL episodes.

    Returns:
        gru4rec_p_like(userId, movieId, episode_sequence) → float
    """
    device = next(model.parameters()).device

    def gru4rec_p_like(userId, movieId, episode_sequence):
        """
        Args:
            userId:           unused (kept for API consistency with get_p_like)
            movieId:          target movie to score
            episode_sequence: list of movieIds recommended so far this episode
        Returns:
            float P(like)
        """
        if movieId not in movie2idx:
            return 0.5

        # +1 offset: index 0 is reserved for padding
        seq_indices = [movie2idx[m] + 1 for m in episode_sequence[-T:]
                       if m in movie2idx]
        padded = [0] * (T - len(seq_indices)) + seq_indices

        seq_tensor = torch.tensor([padded], dtype=torch.long, device=device)
        movie_tensor = torch.tensor([movie2idx[movieId] + 1],
                                    dtype=torch.long, device=device)

        with torch.no_grad():
            p = model(seq_tensor, movie_tensor)
        return float(p.cpu())

    return gru4rec_p_like


# ============================================================
# 4. Fatigue Evaluation
# ============================================================

def _sample_movie_from_genre(genre, genre_to_movies, exclude=None):
    pool = genre_to_movies.get(genre, [])
    if exclude is not None:
        pool = [m for m in pool if m != exclude]
    return random.choice(pool) if pool else None


def _build_history_with_k_repeats(target_genre, length, k, all_genres,
                                  genre_to_movies, exclude_movie=None):
    """
    Build a history of `length` movies where exactly k are from target_genre
    and the rest are from other genres.
    """
    seq = []

    for _ in range(k):
        m = _sample_movie_from_genre(target_genre, genre_to_movies, exclude_movie)
        if m is not None:
            seq.append(m)

    other_genres = [g for g in all_genres if g != target_genre
                    and len(genre_to_movies.get(g, [])) > 0]
    for _ in range(length - len(seq)):
        if not other_genres:
            break
        g = random.choice(other_genres)
        m = _sample_movie_from_genre(g, genre_to_movies, exclude_movie)
        if m is not None:
            seq.append(m)

    random.shuffle(seq)
    return seq[:length]


def _gru_p_given_history(env, user_id, history_movie_ids, target_movie):
    """Temporarily override env state to compute P(like) with a given history."""
    old_user = env.current_user
    old_hist = list(env.all_recommended)

    env.current_user = user_id
    env.all_recommended = list(history_movie_ids)
    p = env._dynamic_p(target_movie)

    env.current_user = old_user
    env.all_recommended = old_hist
    return p


def probe_fatigue_curve(env, ctx, n_users=30, n_targets_per_user=5, hist_len=4):
    """
    Probe whether GRU4Rec captures genre fatigue.

    For each (user, target_movie, k), builds a history with k same-genre
    movies and measures P(like). If fatigue exists, P(like) should
    decrease as k increases.

    Args:
        env:  the RL environment instance (needs _dynamic_p method)
        ctx:  dict from load_all()

    Returns:
        DataFrame with columns: userId, target_movie, target_genre, k_same_genre, p_like
    """
    user_ids = sorted(ctx["user2idx"].keys())
    movie_ids = sorted(ctx["movie2idx"].keys())
    movie_genres = ctx["movie_genres"]
    genre_to_movies = ctx["genre_to_movies"]
    all_genres = sorted(ctx["genre2idx"].keys())

    rows = []
    sampled_users = random.sample(user_ids, min(n_users, len(user_ids)))

    for user_id in sampled_users:
        targets = random.sample(movie_ids, min(n_targets_per_user, len(movie_ids)))

        for target_movie in targets:
            target_genres = movie_genres.get(target_movie, [])
            if not target_genres:
                continue
            target_genre = random.choice(target_genres)

            for k in range(hist_len + 1):
                hist = _build_history_with_k_repeats(
                    target_genre, hist_len, k, all_genres,
                    genre_to_movies, exclude_movie=target_movie,
                )
                if len(hist) < hist_len:
                    continue

                p = _gru_p_given_history(env, user_id, hist, target_movie)
                rows.append({
                    "userId": user_id,
                    "target_movie": target_movie,
                    "target_genre": target_genre,
                    "k_same_genre": k,
                    "p_like": p,
                })

    return pd.DataFrame(rows)


def paired_fatigue_probe(env, ctx, n_pairs=200, hist_len=4):
    """
    Paired comparison: same-genre history vs mixed-genre history.

    If GRU4Rec captures fatigue, P(like | mixed_history) > P(like | same_history).

    Args:
        env:  the RL environment instance
        ctx:  dict from load_all()

    Returns:
        DataFrame with columns: userId, target_movie, target_genre,
                                p_same, p_mix, delta_mix_minus_same
    """
    user_ids = sorted(ctx["user2idx"].keys())
    movie_ids = sorted(ctx["movie2idx"].keys())
    movie_genres = ctx["movie_genres"]
    genre_to_movies = ctx["genre_to_movies"]
    all_genres = sorted(ctx["genre2idx"].keys())

    deltas = []

    for _ in range(n_pairs):
        user_id = random.choice(user_ids)
        target_movie = random.choice(movie_ids)

        target_genres = movie_genres.get(target_movie, [])
        if not target_genres:
            continue
        target_genre = random.choice(target_genres)

        # All same-genre history
        hist_same = _build_history_with_k_repeats(
            target_genre, hist_len, hist_len, all_genres,
            genre_to_movies, exclude_movie=target_movie,
        )
        # All different-genre history
        hist_mix = _build_history_with_k_repeats(
            target_genre, hist_len, 0, all_genres,
            genre_to_movies, exclude_movie=target_movie,
        )

        if len(hist_same) < hist_len or len(hist_mix) < hist_len:
            continue

        p_same = _gru_p_given_history(env, user_id, hist_same, target_movie)
        p_mix = _gru_p_given_history(env, user_id, hist_mix, target_movie)

        deltas.append({
            "userId": user_id,
            "target_movie": target_movie,
            "target_genre": target_genre,
            "p_same": p_same,
            "p_mix": p_mix,
            "delta_mix_minus_same": p_mix - p_same,
        })

    return pd.DataFrame(deltas)