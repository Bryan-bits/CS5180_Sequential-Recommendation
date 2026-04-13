"""
evaluate.py — CS5180 RL Recommendation: Evaluation & Visualization

Contains:
    - run_full_evaluation: evaluate all policies (Bernoulli + expected)
    - plot_comparison: learning curves + reward distributions
    - run_ttest: statistical significance test
    - save_all_artifacts: save models, embeddings, simulators

Usage:
    from evaluate import run_full_evaluation, plot_comparison, save_all_artifacts
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from baselines import (
    run_random, run_greedy_ctr, run_model_eval,
    run_expected_eval, make_model_policy, random_policy,
    greedy_policy, history_movie_select, random_movie_select,
)


# ============================================================
# 1. Full Evaluation Pipeline
# ============================================================

def run_full_evaluation(env, model_dqn, model_ppo, n_eval=500, seed=42):
    """
    Evaluate all four policies with the same set of users.

    Args:
        env:       the RL environment
        model_dqn: trained DQN model
        model_ppo: trained PPO model
        n_eval:    number of eval episodes
        seed:      random seed for user sampling

    Returns:
        dict with keys: eval_users, random, greedy, dqn, ppo
        Each value is a list of episode rewards.
    """
    user_ids = env.user_ids
    eval_users = list(
        np.random.RandomState(seed).choice(user_ids, size=n_eval, replace=True)
    )

    print(f"[eval] Running Bernoulli evaluation ({n_eval} episodes) ...")

    results = {
        "eval_users": eval_users,
        "random": run_random(env, eval_users=eval_users),
        "greedy": run_greedy_ctr(env, eval_users=eval_users),
        "dqn": run_model_eval(env, model_dqn, eval_users=eval_users),
        "ppo": run_model_eval(env, model_ppo, eval_users=eval_users),
    }

    for name in ["random", "greedy", "dqn", "ppo"]:
        r = results[name]
        print(f"  {name:<10s}: {np.mean(r):.2f} ± {np.std(r):.2f}")

    return results


def run_expected_evaluation(env, model_dqn, model_ppo, n_eval=200):
    """
    Evaluate all four policies using expected reward (no Bernoulli noise).

    Returns:
        dict with keys: random, greedy, dqn, ppo
    """
    print(f"[eval] Running expected-reward evaluation ({n_eval} episodes) ...")

    results = {
        "random": run_expected_eval(
            env, random_policy, random_movie_select, n_eval),
        "greedy": run_expected_eval(
            env, greedy_policy, history_movie_select, n_eval),
        "dqn": run_expected_eval(
            env, make_model_policy(model_dqn), history_movie_select, n_eval),
        "ppo": run_expected_eval(
            env, make_model_policy(model_ppo), history_movie_select, n_eval),
    }

    for name in ["random", "greedy", "dqn", "ppo"]:
        r = results[name]
        print(f"  {name:<10s}: {np.mean(r):.3f} ± {np.std(r):.3f}")

    return results


# ============================================================
# 2. Statistical Tests
# ============================================================

def run_ttest(results, policy_a="dqn", policy_b="greedy"):
    """Welch's t-test between two policies."""
    a = results[policy_a]
    b = results[policy_b]
    t_stat, p_val = stats.ttest_ind(a, b)
    sig = "significant" if p_val < 0.05 else "not significant"
    print(f"[t-test] {policy_a} vs {policy_b}: "
          f"t={t_stat:.3f}, p={p_val:.4f} ({sig})")
    return t_stat, p_val


# ============================================================
# 3. Comparison Plots
# ============================================================

def _smooth(x, w=100):
    return [np.mean(x[max(0, i - w // 2):min(len(x), i + w // 2 + 1)])
            for i in range(len(x))]


def plot_comparison(results, callback_dqn=None, callback_ppo=None,
                    T=20, save_path=None):
    """
    Two-panel figure: learning curves (left) + reward distribution (right).

    Args:
        results:      dict from run_full_evaluation
        callback_dqn: EpisodeRewardCallback from DQN training (optional)
        callback_ppo: EpisodeRewardCallback from PPO training (optional)
        T:            episode length (for axis labels)
        save_path:    if provided, save figure to this path
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # ── Left: Learning Curves ──
    ax = axes[0]

    if callback_dqn is not None:
        ep = callback_dqn.episode_rewards
        ax.plot(ep, alpha=0.10, color="steelblue", lw=0.8)
        ax.plot(_smooth(ep), color="steelblue", lw=2, label="DQN (smoothed)")

    if callback_ppo is not None:
        ep = callback_ppo.episode_rewards
        ax.plot(ep, alpha=0.10, color="mediumorchid", lw=0.8)
        ax.plot(_smooth(ep), color="mediumorchid", lw=2, label="PPO (smoothed)")

    colors_line = {"random": "gray", "greedy": "orange",
                   "dqn": "navy", "ppo": "purple"}
    styles = {"random": "--", "greedy": "--", "dqn": "-", "ppo": "-"}

    for name in ["random", "greedy", "dqn", "ppo"]:
        r = results[name]
        ax.axhline(np.mean(r), color=colors_line[name], ls=styles[name],
                   lw=1.5, label=f"{name:<12s} {np.mean(r):.2f}")

    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Cumulative reward (T={T})")
    ax.set_title("Learning Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Right: Reward Distribution ──
    ax2 = axes[1]

    policy_names = ["random", "greedy", "dqn", "ppo"]
    labels = ["Random", "Greedy-CTR", "DQN", "PPO"]
    colors_box = ["#aaaaaa", "#ff9800", "#1565c0", "#7b1fa2"]

    data = [results[name] for name in policy_names]

    bp = ax2.boxplot(data, patch_artist=True, widths=0.45,
                     medianprops=dict(color="white", lw=2))
    for patch, c in zip(bp["boxes"], colors_box):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    for i, (d, c) in enumerate(zip(data, colors_box), 1):
        jitter = np.random.uniform(-0.15, 0.15, len(d))
        ax2.scatter(i + jitter, d, alpha=0.2, s=8, color=c, zorder=3)

    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel(f"Episode reward (T={T})")
    ax2.set_title(f"Reward Distribution ({len(data[0])} episodes)")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "4-Policy Comparison\n"
        "DQN / PPO / Greedy-CTR use identical information; "
        "difference = sequential policy only",
        fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved → {save_path}")

    plt.show()


def print_results_table(results, label="BERNOULLI"):
    """Print a formatted comparison table."""
    n = len(results["random"])

    print(f"\n{'=' * 60}")
    print(f"FINAL COMPARISON — {label}  (T=20, n={n})")
    print(f"{'=' * 60}")
    print(f"{'Policy':<20s} {'Mean':>7s}   {'Std':>5s}")
    print("-" * 40)

    for name, display in [("random", "Random"), ("greedy", "Greedy-CTR"),
                           ("dqn", "DQN"), ("ppo", "PPO")]:
        r = results[name]
        print(f"{display:<20s} {np.mean(r):7.3f} ± {np.std(r):5.3f}")

    print()
    for name in ["dqn", "ppo"]:
        r = results[name]
        diff_random = np.mean(r) - np.mean(results["random"])
        diff_greedy = np.mean(r) - np.mean(results["greedy"])
        print(f"{name.upper()} vs Random     : {diff_random:+.3f}")
        print(f"{name.upper()} vs Greedy-CTR : {diff_greedy:+.3f}  ← key metric")


# ============================================================
# 4. Ablation Evaluation
# ============================================================

def run_ablation_eval(env_ablation, model, eval_users, label="DQN"):
    """Evaluate an ablation model and return rewards."""
    rewards = run_model_eval(env_ablation, model, eval_users=eval_users)
    print(f"[ablation] {label} (no genre counts): "
          f"{np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    return rewards


# ============================================================
# 5. Save All Artifacts
# ============================================================

def save_all_artifacts(output_dir, model_dqn, model_ppo,
                       gru4rec_model, xgb_model, U_embed, M_embed,
                       model_dqn_ablation=None, model_ppo_ablation=None):
    """
    Save all trained models and embeddings to organized subdirectories.

    Directory structure:
        output_dir/
        ├── simulators/
        │   ├── xgb_simulator.json
        │   └── gru4rec_best.pt
        ├── embeddings/
        │   ├── U_embed.npy
        │   └── M_embed.npy
        └── agents/
            ├── dqn_v8.zip
            ├── ppo_v8.zip
            ├── dqn_ablation_v8.zip  (if provided)
            └── ppo_ablation_v8.zip  (if provided)
    """
    import torch

    sim_dir = os.path.join(output_dir, "simulators")
    emb_dir = os.path.join(output_dir, "embeddings")
    agt_dir = os.path.join(output_dir, "agents")

    for d in [sim_dir, emb_dir, agt_dir]:
        os.makedirs(d, exist_ok=True)

    # Simulators
    xgb_model.save_model(os.path.join(sim_dir, "xgb_simulator.json"))
    torch.save(gru4rec_model.state_dict(),
               os.path.join(sim_dir, "gru4rec_best.pt"))

    # Embeddings
    np.save(os.path.join(emb_dir, "U_embed.npy"), U_embed)
    np.save(os.path.join(emb_dir, "M_embed.npy"), M_embed)

    # Agents
    model_dqn.save(os.path.join(agt_dir, "dqn_v8"))
    model_ppo.save(os.path.join(agt_dir, "ppo_v8"))

    if model_dqn_ablation is not None:
        model_dqn_ablation.save(os.path.join(agt_dir, "dqn_ablation_v8"))
    if model_ppo_ablation is not None:
        model_ppo_ablation.save(os.path.join(agt_dir, "ppo_ablation_v8"))

    print(f"[save] All artifacts saved to '{output_dir}/':")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (level + 1)
        for f in sorted(files):
            size = os.path.getsize(os.path.join(root, f)) / (1024 * 1024)
            print(f"{sub_indent}{f:<30s} {size:.2f} MB")
