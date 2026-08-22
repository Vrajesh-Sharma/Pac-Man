"""
Builds a comparison dashboard across every algorithm that has been trained
(i.e. has a results/<algo>_metrics.csv file). Produces:
  - results/comparison_dashboard.png  (reward/score/survival/winrate curves)
  - a printed summary table (also saved to results/summary_table.csv)

Usage:
    python dashboard.py
    python dashboard.py --smooth 50      # rolling-average window size
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALGO_ORDER = ["qlearning", "sarsa", "montecarlo", "dqn", "double_dqn", "reinforce"]
ALGO_LABELS = {
    "qlearning": "Q-Learning", "sarsa": "SARSA", "montecarlo": "Monte Carlo",
    "dqn": "DQN", "double_dqn": "Double DQN", "reinforce": "REINFORCE",
}
COLORS = {
    "qlearning": "#1f77b4", "sarsa": "#ff7f0e", "montecarlo": "#2ca02c",
    "dqn": "#d62728", "double_dqn": "#9467bd", "reinforce": "#8c564b",
}


def load_all(results_dir="results"):
    data = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_metrics.csv"))):
        algo = os.path.basename(path).replace("_metrics.csv", "")
        df = pd.read_csv(path)
        if len(df):
            data[algo] = df
    return data


def rolling(series, window):
    return series.rolling(window=max(1, window), min_periods=1).mean()


def build_dashboard(results_dir="results", out_path=None, smooth=50):
    data = load_all(results_dir)
    if not data:
        print("No metrics CSVs found in results/. Train at least one algorithm first.")
        return
    out_path = out_path or os.path.join(results_dir, "comparison_dashboard.png")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("RL Algorithm Comparison on the Shared PacMan Benchmark", fontsize=16, fontweight="bold")

    def color(algo):
        return COLORS.get(algo, None)

    # 1. episode reward curve
    ax = axes[0, 0]
    for algo, df in data.items():
        ax.plot(df["episode"], rolling(df["total_reward"], smooth), label=ALGO_LABELS.get(algo, algo), color=color(algo))
    ax.set_title("Episode Reward (rolling avg)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total reward")
    ax.legend(fontsize=8)

    # 2. score curve
    ax = axes[0, 1]
    for algo, df in data.items():
        ax.plot(df["episode"], rolling(df["score"], smooth), label=ALGO_LABELS.get(algo, algo), color=color(algo))
    ax.set_title("Game Score (rolling avg)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Score")

    # 3. survival time curve
    ax = axes[0, 2]
    for algo, df in data.items():
        ax.plot(df["episode"], rolling(df["steps_survived"], smooth), label=ALGO_LABELS.get(algo, algo), color=color(algo))
    ax.set_title("Survival Time (rolling avg)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps survived")

    # 4. win rate bar chart (over last 20% of episodes = "trained" performance)
    ax = axes[1, 0]
    algos, win_rates, bar_colors = [], [], []
    for algo in ALGO_ORDER:
        if algo not in data:
            continue
        df = data[algo]
        tail = df.tail(max(1, len(df) // 5))
        win_rates.append((tail["result"] == "win").mean() * 100)
        algos.append(ALGO_LABELS.get(algo, algo))
        bar_colors.append(color(algo))
    ax.bar(algos, win_rates, color=bar_colors)
    ax.set_title("Win Rate - last 20% of episodes")
    ax.set_ylabel("Win rate (%)")
    ax.tick_params(axis='x', rotation=30)

    # 5. difficulty progression (shows curriculum ramping)
    ax = axes[1, 1]
    for algo, df in data.items():
        ax.plot(df["episode"], df["difficulty"], label=ALGO_LABELS.get(algo, algo), color=color(algo))
    ax.set_title("Dynamic Difficulty Level Over Training")
    ax.set_xlabel("Episode"); ax.set_ylabel("Difficulty tier")

    # 6. final summary table rendered as text
    ax = axes[1, 2]
    ax.axis("off")
    rows = []
    for algo in ALGO_ORDER:
        if algo not in data:
            continue
        df = data[algo]
        tail = df.tail(max(1, len(df) // 5))
        rows.append([
            ALGO_LABELS.get(algo, algo),
            f"{tail['total_reward'].mean():.0f}",
            f"{tail['score'].mean():.0f}",
            f"{tail['steps_survived'].mean():.0f}",
            f"{(tail['result'] == 'win').mean()*100:.0f}%",
        ])
    table = ax.table(cellText=rows,
                      colLabels=["Algorithm", "Avg Reward", "Avg Score", "Avg Survival", "Win Rate"],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    ax.set_title("Final Performance Summary (last 20% of episodes)", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    print(f"Dashboard saved -> {out_path}")

    # also dump the summary table as CSV for the report/slides
    summary_df = pd.DataFrame(rows, columns=["Algorithm", "Avg Reward", "Avg Score", "Avg Survival", "Win Rate"])
    summary_csv = os.path.join(results_dir, "summary_table.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary table saved -> {summary_csv}")
    print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--smooth", type=int, default=50, help="rolling average window")
    args = parser.parse_args()
    build_dashboard(args.results_dir, smooth=args.smooth)
