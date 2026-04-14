"""
experiments/tune_xgb.py — Optuna hyperparameter search for XGBoost simulator

Usage:
    from data_setup import load_all
    from experiments.tune_xgb import run_xgb_tuning

    ctx = load_all()
    best_params, study = run_xgb_tuning(ctx, n_trials=50)
"""

import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score


def _build_features(df, user2idx, movie2idx, U_embed, M_embed):
    u_idx = [user2idx[u] for u in df["userId"]]
    m_idx = [movie2idx[m] for m in df["movieId"]]
    X = np.hstack([U_embed[u_idx], M_embed[m_idx]])
    y = (df["rating"].values >= 4).astype(int)
    return X, y


def run_xgb_tuning(ctx, n_trials=50, cv_folds=3):
    """
    Run Optuna hyperparameter search for XGBoost P(like) simulator.

    Args:
        ctx: dict from load_all()
        n_trials: number of Optuna trials
        cv_folds: number of cross-validation folds

    Returns:
        (best_params, study)
    """
    X_train, y_train = _build_features(
        ctx["train_df"], ctx["user2idx"], ctx["movie2idx"],
        ctx["U_embed"], ctx["M_embed"],
    )
    X_test, y_test = _build_features(
        ctx["test_df"], ctx["user2idx"], ctx["movie2idx"],
        ctx["U_embed"], ctx["M_embed"],
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds, scoring="roc_auc", n_jobs=-1,
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest CV AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Retrain with best params and evaluate on held-out test set
    best_model = XGBClassifier(
        **study.best_params,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    best_model.fit(X_train, y_train, verbose=False)
    test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    print(f"Test AUC (best params): {test_auc:.4f}")

    return study.best_params, study